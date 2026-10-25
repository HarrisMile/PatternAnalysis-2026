"""Compare unconditional ConvVAE and VQ-VAE cohorts under a shared protocol."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_generation import (
    distribution_rows,
    downsample,
    image_features,
    load_training_data,
    prepare_output,
    read_json,
    sha256_file,
    squared_distance_matrix,
    validate_images,
    write_csv,
    write_json,
)
from metrics import reconstruction_metrics
from train import positive_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-samples", type=Path, required=True)
    parser.add_argument("--vae-summary", type=Path, required=True)
    parser.add_argument("--vq-samples", type=Path, required=True)
    parser.add_argument("--vq-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("outputs/formal-prior-data/train_samples.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report_assets/generative_baseline_comparison"),
    )
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--downsample-factor", type=positive_int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_samples(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = {"images", "sample_id", "sample_seed", "sample_index_within_seed"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        images = validate_images(archive["images"], f"images in {path}")
        sample_ids = np.asarray(archive["sample_id"], dtype=np.int64)
        sample_seeds = np.asarray(archive["sample_seed"], dtype=np.int64)
        within_seed = np.asarray(archive["sample_index_within_seed"], dtype=np.int64)
    if not (len(images) == len(sample_ids) == len(sample_seeds) == len(within_seed)):
        raise ValueError(f"Sample arrays in {path} have different lengths.")
    if not np.array_equal(sample_ids, np.arange(len(sample_ids))):
        raise ValueError(f"Sample IDs in {path} are not contiguous.")
    return {
        "images": images,
        "sample_id": sample_ids,
        "sample_seed": sample_seeds,
        "sample_index_within_seed": within_seed,
    }


def spatial_gradient_energy(images: np.ndarray) -> np.ndarray:
    vertical = np.abs(np.diff(images, axis=1)).mean(axis=(1, 2))
    horizontal = np.abs(np.diff(images, axis=2)).mean(axis=(1, 2))
    return 0.5 * (vertical + horizontal)


def image_diversity(images: np.ndarray) -> dict[str, float | int]:
    distances = squared_distance_matrix(images, images)
    upper = distances[np.triu_indices(len(images), k=1)]
    unique = len(np.unique(images.reshape(len(images), -1), axis=0))
    return {
        "exact_duplicate_images": int(len(images) - unique),
        "pairwise_mse_min": float(upper.min()),
        "pairwise_mse_median": float(np.median(upper)),
        "pairwise_mse_mean": float(upper.mean()),
    }


def evaluate_model(
    model_name: str,
    samples: dict[str, np.ndarray],
    sampling_summary: dict[str, Any],
    dataset: Any,
    training_features: np.ndarray,
    training_downsampled: np.ndarray,
    downsample_factor: int,
) -> dict[str, Any]:
    images = samples["images"]
    generated_features = image_features(images)
    feature_rows = distribution_rows(training_features, generated_features)
    generated_downsampled = downsample(images, downsample_factor)
    distances = squared_distance_matrix(training_downsampled, generated_downsampled)
    nearest_indices = distances.argmin(axis=0)
    nearest_images = np.stack(
        [dataset[int(index)]["image"][0].numpy() for index in nearest_indices]
    ).astype(np.float32, copy=False)
    metrics = reconstruction_metrics(
        torch.from_numpy(nearest_images[:, None]),
        torch.from_numpy(images[:, None]),
    )
    nearest_ssim = metrics["ssim"].numpy()
    nearest_mse = metrics["mse"].numpy()

    training_edge = spatial_gradient_energy(training_downsampled)
    generated_edge = spatial_gradient_energy(generated_downsampled)
    training_edge_std = float(training_edge.std())
    profile = sampling_summary.get("generation_profile")
    if not isinstance(profile, dict):
        raise ValueError(f"{model_name} summary has no generation profile.")
    feature_abs_z = [
        abs(float(row["standardised_mean_difference"])) for row in feature_rows
    ]
    feature_coverage = [
        float(row["generated_fraction_inside_train_p05_p95"]) for row in feature_rows
    ]
    feature_wasserstein = [
        float(row["quantile_wasserstein_distance"]) for row in feature_rows
    ]
    seeds = list(dict.fromkeys(int(seed) for seed in samples["sample_seed"]))
    return {
        "model": model_name,
        "samples": int(len(images)),
        "seeds": seeds,
        "exact_training_pixel_matches": int(np.count_nonzero(nearest_mse <= 1e-12)),
        "nearest_training_mse_median": float(np.median(nearest_mse)),
        "nearest_training_ssim_min": float(nearest_ssim.min()),
        "nearest_training_ssim_median": float(np.median(nearest_ssim)),
        "nearest_training_ssim_max": float(nearest_ssim.max()),
        "mean_absolute_feature_shift_z": float(np.mean(feature_abs_z)),
        "mean_feature_fraction_inside_train_p05_p95": float(np.mean(feature_coverage)),
        "mean_feature_quantile_wasserstein": float(np.mean(feature_wasserstein)),
        "gradient_energy_median": float(np.median(generated_edge)),
        "gradient_energy_shift_z": (
            float((generated_edge.mean() - training_edge.mean()) / training_edge_std)
            if training_edge_std > 0
            else float("nan")
        ),
        **image_diversity(images),
        "generation_seconds": float(profile["seconds"]),
        "milliseconds_per_image": float(profile["milliseconds_per_image"]),
        "images_per_second": float(profile["images_per_second"]),
        "peak_accelerator_memory_mb": profile.get("peak_accelerator_memory_mb"),
        "device_name": str(sampling_summary.get("device", {}).get("name", "unknown")),
        "feature_metrics": feature_rows,
    }


def save_comparison_panel(
    path: Path,
    cohorts: list[tuple[str, dict[str, np.ndarray]]],
) -> None:
    row_specs: list[tuple[str, int, np.ndarray]] = []
    for model_name, samples in cohorts:
        for seed in dict.fromkeys(int(value) for value in samples["sample_seed"]):
            indices = np.flatnonzero(samples["sample_seed"] == seed)[:4]
            row_specs.append((model_name, seed, indices))
    figure, axes = plt.subplots(
        len(row_specs),
        4,
        figsize=(10.0, 2.7 * len(row_specs)),
        squeeze=False,
    )
    cohort_lookup = {name: samples for name, samples in cohorts}
    for row_index, (model_name, seed, indices) in enumerate(row_specs):
        samples = cohort_lookup[model_name]
        for column_index, axis in enumerate(axes[row_index]):
            axis.axis("off")
            if column_index < len(indices):
                sample_index = int(indices[column_index])
                axis.imshow(
                    samples["images"][sample_index], cmap="gray", vmin=0.0, vmax=1.0
                )
                axis.set_title(
                    f"{model_name}, seed {seed}\n"
                    f"sample {int(samples['sample_index_within_seed'][sample_index]) + 1}",
                    fontsize=9,
                )
    figure.suptitle(
        "Uncurated ConvVAE and VQ-VAE samples: first four outputs per seed",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Generative Baseline Comparison",
        "",
        (
            "ConvVAE and VQ-VAE use the same seeds and sample count. Image-space metrics are "
            "computed against the complete training cohort; nearest-neighbour SSIM is a similarity "
            "and memorisation diagnostic, not the paired-reconstruction SSIM target."
        ),
        "",
        "| Model | Samples / seeds | Pairwise MSE median | Mean feature shift | Nearest-train SSIM | ms/image | Peak memory |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        peak = row["peak_accelerator_memory_mb"]
        peak_text = f"{float(peak):.1f} MiB" if peak is not None else "not available"
        lines.append(
            f"| {row['model']} | {row['samples']} / {len(row['seeds'])} | "
            f"{row['pairwise_mse_median']:.4f} | "
            f"{row['mean_absolute_feature_shift_z']:.3f} | "
            f"{row['nearest_training_ssim_median']:.3f} | "
            f"{row['milliseconds_per_image']:.2f} | {peak_text} |"
        )
    lines.extend(
        [
            "",
            (
                "Lower feature shift indicates closer agreement with the seven predefined training "
                "image statistics. Pairwise MSE measures numerical diversity but does not establish "
                "anatomical plausibility. The uncurated panel must be interpreted together with the "
                "VQ-VAE failure autopsy and nearest-neighbour audit."
            ),
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "vae_samples": args.vae_samples.resolve(),
        "vae_summary": args.vae_summary.resolve(),
        "vq_samples": args.vq_samples.resolve(),
        "vq_summary": args.vq_summary.resolve(),
        "manifest": args.manifest.resolve(),
        "train_metadata": args.train_metadata.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    output = args.output.resolve()
    prepare_output(output, args.overwrite)

    vae_samples = load_samples(paths["vae_samples"])
    vq_samples = load_samples(paths["vq_samples"])
    vae_summary = read_json(paths["vae_summary"])
    vq_summary = read_json(paths["vq_summary"])
    for name, samples, summary in (
        ("ConvVAE", vae_samples, vae_summary),
        ("VQ-VAE", vq_samples, vq_summary),
    ):
        if int(summary.get("num_samples", -1)) != len(samples["images"]):
            raise ValueError(
                f"{name} summary and archive have different sample counts."
            )
    if not (
        np.array_equal(vae_samples["sample_seed"], vq_samples["sample_seed"])
        and np.array_equal(
            vae_samples["sample_index_within_seed"],
            vq_samples["sample_index_within_seed"],
        )
    ):
        raise ValueError(
            "ConvVAE and VQ-VAE must use identical seeds and per-seed sample indices."
        )

    dataset, training_features, training_downsampled, _ = load_training_data(
        paths["manifest"],
        paths["train_metadata"],
        args.batch_size,
        args.num_workers,
        args.downsample_factor,
    )
    rows = [
        evaluate_model(
            "ConvVAE",
            vae_samples,
            vae_summary,
            dataset,
            training_features,
            training_downsampled,
            args.downsample_factor,
        ),
        evaluate_model(
            "VQ-VAE + PixelCNN",
            vq_samples,
            vq_summary,
            dataset,
            training_features,
            training_downsampled,
            args.downsample_factor,
        ),
    ]
    csv_rows = [
        {key: value for key, value in row.items() if key != "feature_metrics"}
        for row in rows
    ]
    comparison_csv = output / "generative_model_comparison.csv"
    comparison_panel = output / "uncurated_multiseed_comparison.png"
    report_path = output / "REPORT_RESULTS.md"
    write_csv(comparison_csv, csv_rows)
    save_comparison_panel(
        comparison_panel,
        [("ConvVAE", vae_samples), ("VQ-VAE + PixelCNN", vq_samples)],
    )
    write_report(report_path, rows)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "reference_split": "train",
            "training_samples": len(dataset),
            "sampling_seeds": list(
                dict.fromkeys(int(value) for value in vae_samples["sample_seed"])
            ),
            "samples_per_model": len(vae_samples["images"]),
            "same_seed_and_sample_indices_verified": True,
            "panel_selection_rule": (
                "First four samples per model and seed in generation order; no visual selection"
            ),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "models": rows,
        "artifacts": {
            "comparison_csv": comparison_csv.name,
            "comparison_panel": comparison_panel.name,
            "text_report": report_path.name,
        },
        "limitations": [
            "Image-feature distances are engineering diagnostics, not clinical realism scores.",
            "Nearest-neighbour SSIM is not the paired-reconstruction SSIM target.",
            "A multi-seed sample audit does not establish privacy or population coverage.",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    summary = compare(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
