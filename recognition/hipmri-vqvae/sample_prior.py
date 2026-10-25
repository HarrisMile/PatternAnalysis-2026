"""Sample an unconditional PixelCNN prior and decode VQ grids into MRI slices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import PixelCNNPrior, VQVAE
from train import (
    configure_reproducibility,
    describe_device,
    positive_int,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("predictions/formal-prior-samples")
    )
    parser.add_argument("--num-samples", type=positive_int, default=16)
    parser.add_argument(
        "--samples-per-seed",
        type=positive_int,
        help="Samples drawn for every value passed to --seeds.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=positive_int)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Multiple independent sampling seeds; requires --samples-per-seed.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if not overwrite and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Sample artifacts already exist in {output}. "
            "Use a new directory or pass --overwrite explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def save_panel(
    path: Path,
    images: np.ndarray,
    sample_seeds: np.ndarray,
    sample_indices_within_seed: np.ndarray,
) -> None:
    ordered_seeds = list(dict.fromkeys(int(seed) for seed in sample_seeds))
    per_seed = 16 if len(ordered_seeds) == 1 else 4
    shown_indices = np.concatenate(
        [np.flatnonzero(sample_seeds == seed)[:per_seed] for seed in ordered_seeds]
    )
    columns = min(4, len(shown_indices))
    rows = math.ceil(len(shown_indices) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.0 * columns, 4.2 * rows), squeeze=False
    )
    for axis_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if axis_index < len(shown_indices):
            sample_index = int(shown_indices[axis_index])
            axis.imshow(images[sample_index], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(
                f"seed {int(sample_seeds[sample_index])}, "
                f"sample {int(sample_indices_within_seed[sample_index]) + 1}"
            )
    figure.suptitle("Uncurated unconditional VQ-VAE samples across seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def code_statistics(code_indices: np.ndarray, num_embeddings: int) -> dict[str, Any]:
    counts = np.bincount(
        code_indices.reshape(-1).astype(np.int64), minlength=num_embeddings
    )
    probabilities = counts.astype(np.float64) / counts.sum()
    nonzero = probabilities > 0
    perplexity = float(
        np.exp(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum())
    )
    return {
        "total_assignments": int(counts.sum()),
        "active_codes": int(nonzero.sum()),
        "active_code_fraction": float(nonzero.mean()),
        "perplexity": perplexity,
        "code_counts": counts.tolist(),
    }


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def generate(args: argparse.Namespace) -> dict[str, Any]:
    prior_path = args.prior_checkpoint.resolve()
    vqvae_path = args.vqvae_checkpoint.resolve()
    output = args.output.resolve()
    if not prior_path.is_file():
        raise FileNotFoundError(f"Prior checkpoint not found: {prior_path}")
    if not vqvae_path.is_file():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vqvae_path}")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.seeds is None:
        seeds = [int(args.seed)]
        samples_per_seed = int(args.num_samples)
    else:
        if args.samples_per_seed is None:
            raise ValueError("--samples-per-seed is required when --seeds is used.")
        seeds = [int(seed) for seed in args.seeds]
        if len(set(seeds)) != len(seeds):
            raise ValueError("--seeds must not contain duplicate values.")
        samples_per_seed = int(args.samples_per_seed)
    total_samples = len(seeds) * samples_per_seed
    prepare_output(output, args.overwrite)

    prior_checkpoint = torch.load(prior_path, map_location="cpu", weights_only=True)
    if prior_checkpoint.get("model_name") != "pixelcnn_prior":
        raise ValueError("The prior checkpoint is not a PixelCNN prior.")
    model_config = prior_checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Prior checkpoint has no model configuration.")
    vocabulary = np.asarray(prior_checkpoint.get("vocabulary"), dtype=np.int64)
    grid_shape = tuple(int(value) for value in prior_checkpoint.get("grid_shape", []))
    if vocabulary.ndim != 1 or len(vocabulary) != int(
        model_config.get("num_tokens", -1)
    ):
        raise ValueError("Prior checkpoint vocabulary is invalid.")
    if len(grid_shape) != 2 or min(grid_shape) <= 0:
        raise ValueError("Prior checkpoint grid shape is invalid.")
    if args.top_k is not None and args.top_k > len(vocabulary):
        raise ValueError("--top-k cannot exceed the compact prior vocabulary.")

    vqvae_sha256 = sha256_file(vqvae_path)
    expected_vqvae_sha256 = prior_checkpoint.get("provenance", {}).get(
        "source_vqvae_sha256"
    )
    if expected_vqvae_sha256 and vqvae_sha256 != expected_vqvae_sha256:
        raise ValueError(
            "The supplied VQ-VAE checkpoint does not match the checkpoint used to export prior data."
        )
    vqvae_checkpoint = torch.load(vqvae_path, map_location="cpu", weights_only=True)
    if vqvae_checkpoint.get("model_name") != "vqvae":
        raise ValueError("The decoder checkpoint is not a VQ-VAE.")

    configure_reproducibility(seeds[0])
    device = select_device(args.device)
    prior = PixelCNNPrior(**model_config)
    prior.load_state_dict(prior_checkpoint["model_state"])
    prior = prior.to(device).eval()
    vqvae = VQVAE()
    vqvae.load_state_dict(vqvae_checkpoint["model_state"])
    vqvae = vqvae.to(device).eval()
    if not bool(vqvae.quantizer.codebook_initialized.item()):
        raise ValueError("The VQ-VAE codebook is not marked as initialised.")
    if (
        int(vocabulary.min()) < 0
        or int(vocabulary.max()) >= vqvae.quantizer.num_embeddings
    ):
        raise ValueError(
            "Prior vocabulary contains an index outside the VQ-VAE codebook."
        )

    vocabulary_tensor = torch.from_numpy(vocabulary).to(device=device, dtype=torch.long)

    original_batches: list[np.ndarray] = []
    image_batches: list[np.ndarray] = []
    seed_batches: list[np.ndarray] = []
    within_seed_batches: list[np.ndarray] = []
    per_seed_profiles: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronise(device)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        for seed in seeds:
            configure_reproducibility(seed)
            if device.type in {"cpu", "cuda"}:
                random_generator: torch.Generator | None = torch.Generator(
                    device=device
                )
                random_generator.manual_seed(seed)
            else:
                random_generator = None
            synchronise(device)
            seed_start = time.perf_counter()
            for start in range(0, samples_per_seed, args.batch_size):
                current_batch = min(args.batch_size, samples_per_seed - start)
                compact_indices = prior.sample(
                    batch_size=current_batch,
                    grid_shape=grid_shape,
                    device=device,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    generator=random_generator,
                )
                original_indices = vocabulary_tensor[compact_indices]
                quantized = vqvae.quantizer.embedding(original_indices)
                quantized = quantized.permute(0, 3, 1, 2).contiguous()
                images = vqvae.decoder(quantized)
                if images.ndim != 4 or images.shape[1] != 1:
                    raise RuntimeError(
                        f"Unexpected decoded image shape: {tuple(images.shape)}"
                    )
                if (
                    not torch.isfinite(images).all()
                    or float(images.min()) < 0
                    or float(images.max()) > 1
                ):
                    raise RuntimeError("Decoded samples contain invalid intensities.")
                original_batches.append(
                    original_indices.cpu().numpy().astype(np.uint16, copy=False)
                )
                image_batches.append(
                    images[:, 0].cpu().numpy().astype(np.float32, copy=False)
                )
                seed_batches.append(np.full(current_batch, seed, dtype=np.int64))
                within_seed_batches.append(
                    np.arange(start, start + current_batch, dtype=np.int32)
                )
                print(
                    f"seed {seed}: generated {start + current_batch}/{samples_per_seed} samples",
                    flush=True,
                )
            synchronise(device)
            seed_seconds = time.perf_counter() - seed_start
            per_seed_profiles.append(
                {
                    "seed": seed,
                    "samples": samples_per_seed,
                    "seconds": seed_seconds,
                    "milliseconds_per_image": 1000.0 * seed_seconds / samples_per_seed,
                }
            )
    synchronise(device)
    generation_seconds = time.perf_counter() - generation_start

    code_indices = np.concatenate(original_batches, axis=0)
    images = np.concatenate(image_batches, axis=0)
    sample_seeds = np.concatenate(seed_batches)
    sample_indices_within_seed = np.concatenate(within_seed_batches)
    if len(images) != total_samples or code_indices.shape[1:] != grid_shape:
        raise RuntimeError("Generated artifacts have an unexpected shape.")

    samples_path = output / "samples.npz"
    panel_path = output / "samples.png"
    summary_path = output / "summary.json"
    write_npz(
        samples_path,
        images=images,
        code_indices=code_indices,
        sample_id=np.arange(total_samples, dtype=np.int32),
        sample_seed=sample_seeds,
        sample_index_within_seed=sample_indices_within_seed,
    )
    save_panel(panel_path, images, sample_seeds, sample_indices_within_seed)
    summary = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_type": "unconditional_autoregressive",
        "num_samples": total_samples,
        "seeds": seeds,
        "samples_per_seed": samples_per_seed,
        "batch_size": args.batch_size,
        "image_shape": list(images.shape[1:]),
        "code_grid_shape": list(grid_shape),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": seeds[0] if len(seeds) == 1 else None,
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "name": describe_device(device),
        },
        "generation_profile": {
            "seconds": generation_seconds,
            "milliseconds_per_image": 1000.0 * generation_seconds / total_samples,
            "images_per_second": total_samples / generation_seconds,
            "peak_accelerator_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
            "per_seed": per_seed_profiles,
        },
        "prior_checkpoint": {
            "path": str(prior_path),
            "sha256": sha256_file(prior_path),
            "epoch": int(prior_checkpoint["epoch"]),
            "best_epoch": int(prior_checkpoint["best_epoch"]),
            "validation_nll": float(prior_checkpoint["best_validation_nll"]),
        },
        "vqvae_checkpoint": {
            "path": str(vqvae_path),
            "sha256": vqvae_sha256,
            "epoch": int(vqvae_checkpoint["epoch"]),
            "best_epoch": int(vqvae_checkpoint["best_epoch"]),
        },
        "compact_vocabulary": vocabulary.tolist(),
        "sample_code_statistics": code_statistics(
            code_indices,
            int(vqvae.quantizer.num_embeddings),
        ),
        "artifacts": {
            "samples": samples_path.name,
            "panel": panel_path.name,
            "panel_selection_rule": (
                "First four generated samples per seed in declared seed order; no visual selection"
                if len(seeds) > 1
                else "First sixteen generated samples in sample order; no visual selection"
            ),
        },
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    summary = generate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
