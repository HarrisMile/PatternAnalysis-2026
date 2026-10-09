"""Evaluate a saved HipMRI reconstruction model and visualise its worst cases."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset
from metrics import reconstruction_metrics, region_reconstruction_metrics
from train import build_model, positive_int, select_device


METRIC_NAMES = (
    "mse",
    "mae",
    "psnr",
    "ssim",
    "foreground_mse",
    "foreground_mae",
    "foreground_psnr",
    "foreground_ssim",
    "background_mse",
    "background_mae",
    "background_psnr",
    "background_ssim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("predictions/run"))
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-batches", type=positive_int)
    parser.add_argument("--num-visualisations", type=int, default=6)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean_metrics(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        finite = np.isfinite(values)
        summary[name] = float(values[finite].mean()) if finite.any() else None
        if name.startswith(("foreground_", "background_")):
            summary[f"{name}_valid_samples"] = int(finite.sum())
    return summary


def _codebook_statistics(code_counts: Optional[Tensor]) -> Dict[str, float]:
    if code_counts is None:
        return {}
    total = code_counts.sum().clamp_min(1.0)
    probabilities = code_counts / total
    nonzero = probabilities > 0
    perplexity = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    active_codes = int(nonzero.sum().item())
    num_codes = int(code_counts.numel())
    return {
        "codebook_perplexity": float(perplexity.item()),
        "active_codes": active_codes,
        "dead_codes": num_codes - active_codes,
        "active_code_fraction": active_codes / num_codes,
    }


def _write_metrics(path: Path, rows: list[Dict[str, Any]]) -> None:
    fieldnames = [
        "subject_id",
        "week",
        "slice_index",
        "image_path",
        "mask_path",
        *METRIC_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_visualisation(path: Path, samples: list[Dict[str, Any]]) -> None:
    if not samples:
        return
    figure, axes = plt.subplots(
        len(samples),
        4,
        figsize=(12, 3.2 * len(samples)),
        squeeze=False,
    )
    column_titles = ("Original", "Reconstruction", "Absolute error", "Segmentation mask")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)

    for row_index, sample in enumerate(samples):
        original = sample["original"]
        reconstruction = sample["reconstruction"]
        error = np.abs(reconstruction - original)
        mask = sample["mask"]
        axes[row_index, 0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].imshow(reconstruction, cmap="gray", vmin=0.0, vmax=1.0)
        error_limit = max(float(np.percentile(error, 99.5)), 1e-6)
        axes[row_index, 2].imshow(error, cmap="magma", vmin=0.0, vmax=error_limit)
        axes[row_index, 3].imshow(mask, cmap="viridis", vmin=0, vmax=5)
        axes[row_index, 0].set_ylabel(
            f"subject {sample['subject_id']}\nweek {sample['week']}, "
            f"slice {sample['slice_index']}\nMSE {sample['mse']:.5f}"
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Highest-MSE reconstruction cases", y=1.002)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.num_visualisations < 0:
        raise ValueError("--num-visualisations cannot be negative.")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output}. Use --overwrite to reuse it."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_name = checkpoint["model_name"]
    if model_name not in {"vae", "vqvae"}:
        raise ValueError(f"Unsupported model in checkpoint: {model_name}")
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = HipMRISliceDataset(args.manifest, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    metrics_rows: list[Dict[str, Any]] = []
    worst_samples: list[tuple[float, int, Dict[str, Any]]] = []
    code_counts: Optional[Tensor] = None
    arrays_directory = args.output / "arrays"
    if args.save_arrays:
        arrays_directory.mkdir(parents=True, exist_ok=True)

    sequence = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            masks = batch["mask"].to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            reconstructions = outputs["reconstruction"]
            batch_metrics = reconstruction_metrics(images, reconstructions)
            batch_metrics.update(region_reconstruction_metrics(images, reconstructions, masks))

            if model_name == "vqvae":
                indices = outputs["encoding_indices"].reshape(-1)
                num_embeddings = int(model.quantizer.num_embeddings)
                counts = torch.bincount(indices, minlength=num_embeddings).to(torch.float64).cpu()
                code_counts = counts if code_counts is None else code_counts + counts

            originals_cpu = images.cpu().numpy()
            reconstructions_cpu = reconstructions.cpu().numpy()
            masks_cpu = batch["mask"].numpy()
            metric_values = {name: values.cpu().numpy() for name, values in batch_metrics.items()}

            for item_index in range(images.shape[0]):
                row: Dict[str, Any] = {
                    "subject_id": batch["subject_id"][item_index],
                    "week": int(batch["week"][item_index]),
                    "slice_index": int(batch["slice_index"][item_index]),
                    "image_path": batch["image_path"][item_index],
                    "mask_path": batch["mask_path"][item_index],
                }
                row.update(
                    {
                        name: float(metric_values[name][item_index])
                        for name in METRIC_NAMES
                    }
                )
                metrics_rows.append(row)
                sample = {
                    **row,
                    "original": originals_cpu[item_index, 0],
                    "reconstruction": reconstructions_cpu[item_index, 0],
                    "mask": masks_cpu[item_index],
                }
                if args.num_visualisations > 0:
                    entry = (row["mse"], sequence, sample)
                    if len(worst_samples) < args.num_visualisations:
                        heapq.heappush(worst_samples, entry)
                    elif row["mse"] > worst_samples[0][0]:
                        heapq.heapreplace(worst_samples, entry)

                if args.save_arrays:
                    filename = (
                        f"subject-{row['subject_id']}_week-{row['week']}_"
                        f"slice-{row['slice_index']}.npz"
                    )
                    np.savez_compressed(
                        arrays_directory / filename,
                        original=sample["original"],
                        reconstruction=sample["reconstruction"],
                        mask=sample["mask"],
                    )
                sequence += 1

    if not metrics_rows:
        raise RuntimeError("No samples were evaluated; check the dataset and batch limit.")
    _write_metrics(args.output / "metrics.csv", metrics_rows)
    ordered_worst = [entry[2] for entry in sorted(worst_samples, reverse=True)]
    _save_visualisation(args.output / "worst_reconstructions.png", ordered_worst)

    summary: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_name": model_name,
        "split": args.split,
        "evaluated_samples": len(metrics_rows),
        "limited_evaluation": args.max_batches is not None,
        "device": str(device),
        **_mean_metrics(metrics_rows),
        **_codebook_statistics(code_counts),
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved evaluation artifacts to {args.output}")


if __name__ == "__main__":
    main()
