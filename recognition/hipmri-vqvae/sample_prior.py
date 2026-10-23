"""Sample an unconditional PixelCNN prior and decode VQ grids into MRI slices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import PixelCNNPrior, VQVAE
from train import configure_reproducibility, describe_device, positive_int, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions/formal-prior-samples"))
    parser.add_argument("--num-samples", type=positive_int, default=16)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=positive_int)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
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


def save_panel(path: Path, images: np.ndarray) -> None:
    shown = min(len(images), 16)
    columns = min(4, shown)
    rows = math.ceil(shown / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 4.2 * rows), squeeze=False)
    for axis_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if axis_index < shown:
            axis.imshow(images[axis_index], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(f"Random sample {axis_index + 1}")
    figure.suptitle("Unconditional VQ-VAE samples from the PixelCNN prior")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def code_statistics(code_indices: np.ndarray, num_embeddings: int) -> dict[str, Any]:
    counts = np.bincount(code_indices.reshape(-1).astype(np.int64), minlength=num_embeddings)
    probabilities = counts.astype(np.float64) / counts.sum()
    nonzero = probabilities > 0
    perplexity = float(np.exp(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()))
    return {
        "total_assignments": int(counts.sum()),
        "active_codes": int(nonzero.sum()),
        "active_code_fraction": float(nonzero.mean()),
        "perplexity": perplexity,
        "code_counts": counts.tolist(),
    }


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
    prepare_output(output, args.overwrite)

    prior_checkpoint = torch.load(prior_path, map_location="cpu", weights_only=True)
    if prior_checkpoint.get("model_name") != "pixelcnn_prior":
        raise ValueError("The prior checkpoint is not a PixelCNN prior.")
    model_config = prior_checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Prior checkpoint has no model configuration.")
    vocabulary = np.asarray(prior_checkpoint.get("vocabulary"), dtype=np.int64)
    grid_shape = tuple(int(value) for value in prior_checkpoint.get("grid_shape", []))
    if vocabulary.ndim != 1 or len(vocabulary) != int(model_config.get("num_tokens", -1)):
        raise ValueError("Prior checkpoint vocabulary is invalid.")
    if len(grid_shape) != 2 or min(grid_shape) <= 0:
        raise ValueError("Prior checkpoint grid shape is invalid.")
    if args.top_k is not None and args.top_k > len(vocabulary):
        raise ValueError("--top-k cannot exceed the compact prior vocabulary.")

    vqvae_sha256 = sha256_file(vqvae_path)
    expected_vqvae_sha256 = prior_checkpoint.get("provenance", {}).get("source_vqvae_sha256")
    if expected_vqvae_sha256 and vqvae_sha256 != expected_vqvae_sha256:
        raise ValueError(
            "The supplied VQ-VAE checkpoint does not match the checkpoint used to export prior data."
        )
    vqvae_checkpoint = torch.load(vqvae_path, map_location="cpu", weights_only=True)
    if vqvae_checkpoint.get("model_name") != "vqvae":
        raise ValueError("The decoder checkpoint is not a VQ-VAE.")

    configure_reproducibility(args.seed)
    device = select_device(args.device)
    prior = PixelCNNPrior(**model_config)
    prior.load_state_dict(prior_checkpoint["model_state"])
    prior = prior.to(device).eval()
    vqvae = VQVAE()
    vqvae.load_state_dict(vqvae_checkpoint["model_state"])
    vqvae = vqvae.to(device).eval()
    if not bool(vqvae.quantizer.codebook_initialized.item()):
        raise ValueError("The VQ-VAE codebook is not marked as initialised.")
    if int(vocabulary.min()) < 0 or int(vocabulary.max()) >= vqvae.quantizer.num_embeddings:
        raise ValueError("Prior vocabulary contains an index outside the VQ-VAE codebook.")

    vocabulary_tensor = torch.from_numpy(vocabulary).to(device=device, dtype=torch.long)
    if device.type in {"cpu", "cuda"}:
        random_generator: torch.Generator | None = torch.Generator(device=device)
        random_generator.manual_seed(args.seed)
    else:
        random_generator = None

    original_batches: list[np.ndarray] = []
    image_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, args.num_samples, args.batch_size):
            current_batch = min(args.batch_size, args.num_samples - start)
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
                raise RuntimeError(f"Unexpected decoded image shape: {tuple(images.shape)}")
            if not torch.isfinite(images).all() or float(images.min()) < 0 or float(images.max()) > 1:
                raise RuntimeError("Decoded samples contain invalid intensities.")
            original_batches.append(original_indices.cpu().numpy().astype(np.uint16, copy=False))
            image_batches.append(images[:, 0].cpu().numpy().astype(np.float32, copy=False))
            print(f"generated {start + current_batch}/{args.num_samples} samples", flush=True)

    code_indices = np.concatenate(original_batches, axis=0)
    images = np.concatenate(image_batches, axis=0)
    if len(images) != args.num_samples or code_indices.shape[1:] != grid_shape:
        raise RuntimeError("Generated artifacts have an unexpected shape.")

    samples_path = output / "samples.npz"
    panel_path = output / "samples.png"
    summary_path = output / "summary.json"
    write_npz(
        samples_path,
        images=images,
        code_indices=code_indices,
        sample_id=np.arange(args.num_samples, dtype=np.int32),
    )
    save_panel(panel_path, images)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_type": "unconditional_autoregressive",
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "image_shape": list(images.shape[1:]),
        "code_grid_shape": list(grid_shape),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "name": describe_device(device),
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
        },
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    summary = generate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
