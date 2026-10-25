"""Draw unconditional ConvVAE baseline samples from the standard-normal latent prior."""

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

from modules import ConvVAE
from train import (
    configure_reproducibility,
    describe_device,
    positive_int,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("predictions/formal-vae-samples")
    )
    parser.add_argument("--samples-per-seed", type=positive_int, default=16)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3710, 3711, 3712])
    parser.add_argument("--latent-height", type=positive_int, default=32)
    parser.add_argument("--latent-width", type=positive_int, default=18)
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


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def save_panel(
    path: Path,
    images: np.ndarray,
    sample_seeds: np.ndarray,
    sample_indices_within_seed: np.ndarray,
) -> None:
    ordered_seeds = list(dict.fromkeys(int(seed) for seed in sample_seeds))
    shown_indices = np.concatenate(
        [np.flatnonzero(sample_seeds == seed)[:4] for seed in ordered_seeds]
    )
    columns = min(4, len(shown_indices))
    rows = math.ceil(len(shown_indices) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 4.2 * rows),
        squeeze=False,
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
    figure.suptitle("Uncurated unconditional ConvVAE baseline samples across seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    seeds = [int(seed) for seed in args.seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicate values.")
    prepare_output(output, args.overwrite)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "vae":
        raise ValueError("The supplied checkpoint is not a ConvVAE checkpoint.")
    model = ConvVAE()
    model.load_state_dict(checkpoint["model_state"])
    device = select_device(args.device)
    model = model.to(device).eval()
    latent_channels = int(model.mu_head.out_channels)
    total_samples = len(seeds) * args.samples_per_seed

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
            for start in range(0, args.samples_per_seed, args.batch_size):
                current_batch = min(args.batch_size, args.samples_per_seed - start)
                latents = torch.randn(
                    (
                        current_batch,
                        latent_channels,
                        args.latent_height,
                        args.latent_width,
                    ),
                    device=device,
                    generator=random_generator,
                )
                images = model.decoder(latents)
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
                image_batches.append(
                    images[:, 0].cpu().numpy().astype(np.float32, copy=False)
                )
                seed_batches.append(np.full(current_batch, seed, dtype=np.int64))
                within_seed_batches.append(
                    np.arange(start, start + current_batch, dtype=np.int32)
                )
                print(
                    f"seed {seed}: generated {start + current_batch}/{args.samples_per_seed} samples",
                    flush=True,
                )
            synchronise(device)
            seed_seconds = time.perf_counter() - seed_start
            per_seed_profiles.append(
                {
                    "seed": seed,
                    "samples": args.samples_per_seed,
                    "seconds": seed_seconds,
                    "milliseconds_per_image": 1000.0
                    * seed_seconds
                    / args.samples_per_seed,
                }
            )
    synchronise(device)
    generation_seconds = time.perf_counter() - generation_start

    images = np.concatenate(image_batches, axis=0)
    sample_seeds = np.concatenate(seed_batches)
    sample_indices_within_seed = np.concatenate(within_seed_batches)
    if len(images) != total_samples:
        raise RuntimeError("Generated artifacts have an unexpected sample count.")

    samples_path = output / "samples.npz"
    panel_path = output / "samples.png"
    summary_path = output / "summary.json"
    write_npz(
        samples_path,
        images=images,
        sample_id=np.arange(total_samples, dtype=np.int32),
        sample_seed=sample_seeds,
        sample_index_within_seed=sample_indices_within_seed,
    )
    save_panel(panel_path, images, sample_seeds, sample_indices_within_seed)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_type": "unconditional_standard_normal_latent",
        "num_samples": total_samples,
        "seeds": seeds,
        "samples_per_seed": args.samples_per_seed,
        "batch_size": args.batch_size,
        "latent_shape": [latent_channels, args.latent_height, args.latent_width],
        "image_shape": list(images.shape[1:]),
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
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "best_epoch": int(checkpoint["best_epoch"]),
        },
        "artifacts": {
            "samples": samples_path.name,
            "panel": panel_path.name,
            "panel_selection_rule": (
                "First four generated samples per seed in declared seed order; no visual selection"
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
