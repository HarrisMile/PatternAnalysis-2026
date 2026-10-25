"""Audit unconditional VQ-VAE samples for diversity, coverage, and memorisation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset
from metrics import reconstruction_metrics
from modules import VQVAE
from train import describe_device, positive_int, select_device


FEATURE_NAMES = (
    "mean_intensity",
    "intensity_std",
    "near_zero_fraction",
    "tissue_fraction",
    "intensity_p95",
    "vertical_centre_of_mass",
    "horizontal_centre_of_mass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("predictions/formal-prior-samples/samples.npz"),
    )
    parser.add_argument(
        "--sample-summary",
        type=Path,
        help="Sampling summary JSON; defaults to summary.json beside --samples.",
    )
    parser.add_argument(
        "--train-codes",
        type=Path,
        default=Path("outputs/prior-data/train_code_indices.npz"),
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("outputs/prior-data/train_samples.csv"),
    )
    parser.add_argument(
        "--train-summary",
        type=Path,
        default=Path("outputs/prior-data/summary.json"),
    )
    parser.add_argument(
        "--prior-summary",
        type=Path,
        default=Path("outputs/formal-prior/summary.json"),
    )
    parser.add_argument(
        "--vqvae-checkpoint",
        type=Path,
        default=Path("outputs/formal-vqvae/best.pt"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report_assets/generative_evaluation"),
    )
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--downsample-factor", type=positive_int, default=4)
    parser.add_argument("--panel-samples", type=positive_int, default=8)
    parser.add_argument("--failure-cases", type=positive_int, default=5)
    parser.add_argument("--interpolation-pairs", type=positive_int, default=4)
    parser.add_argument("--interpolation-steps", type=positive_int, default=7)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Evaluation artifacts already exist in {path}. "
            "Use a new directory or pass --overwrite explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    memorisation = summary["memorisation_audit"]
    diversity = summary["diversity"]
    coverage = summary["distribution_coverage"]
    feature_metrics = {row["feature"]: row for row in coverage["image_feature_metrics"]}
    failure_lines = [
        "## Deterministically selected generation failures",
        "",
        (
            f"The {summary['failure_autopsy']['case_count']} cases below are the largest "
            "predefined feature-deviation scores, not visually cherry-picked examples."
        ),
        "",
    ]
    for case_number, row in enumerate(summary["failure_autopsy"]["rows"], 1):
        failure_lines.extend(
            [
                (
                    f"- Case {case_number}: seed {row['sample_seed']}, sample "
                    f"{row['sample_index_within_seed'] + 1}; dominant deviation "
                    f"`{row['dominant_feature']}` = {row['dominant_feature_z']:.2f} training "
                    f"standard deviations, nearest-training SSIM "
                    f"{row['full_resolution_ssim']:.3f}."
                ),
            ]
        )
    failure_lines.extend(
        [
            "",
            (
                "These cases require visual interpretation alongside the recorded quantitative "
                "trigger; the score is an audit selection rule, not a clinical abnormality score."
            ),
            "",
        ]
    )
    lines = [
        "# Unconditional Generation Audit",
        "",
        "## Scope",
        "",
        (
            f"This audit compares {summary['protocol']['generated_samples']} unconditional "
            f"samples with all {summary['protocol']['training_samples']:,} training slices. "
            "Only the training split is used as the memorisation reference."
        ),
        (
            f"Samples were drawn from seeds {summary['protocol']['sampling_seeds']} at "
            f"{summary['generation_profile']['milliseconds_per_image']:.2f} ms/image on "
            f"{summary['generation_profile']['device_name']}."
        ),
        "",
        "## Memorisation check",
        "",
        f"- Exact pixel matches: {memorisation['exact_pixel_matches']}",
        f"- Exact VQ code-grid matches: {memorisation['exact_code_grid_matches']}",
        (
            "- Nearest-training full-resolution SSIM: "
            f"median {memorisation['nearest_training_full_resolution_ssim_median']:.3f}, "
            f"range {memorisation['nearest_training_full_resolution_ssim_min']:.3f} to "
            f"{memorisation['nearest_training_full_resolution_ssim_max']:.3f}"
        ),
        (
            "- Nearest-training VQ mismatch fraction: "
            f"median {memorisation['nearest_training_code_mismatch_fraction_median']:.3f}, "
            f"range {memorisation['nearest_training_code_mismatch_fraction_min']:.3f} to "
            f"{memorisation['nearest_training_code_mismatch_fraction_max']:.3f}"
        ),
        "",
        (
            "No direct copy of a training image or training code grid was detected. This is "
            "evidence against exact memorisation, but it is not a formal privacy guarantee."
        ),
        "",
        "## Diversity",
        "",
        f"- Exact duplicate generated images: {diversity['exact_duplicate_images']}",
        f"- Exact duplicate generated code grids: {diversity['exact_duplicate_code_grids']}",
        (
            "- Pairwise generated-image MSE: "
            f"median {diversity['pairwise_pixel_mse_median']:.4f}, "
            f"minimum {diversity['pairwise_pixel_mse_min']:.4f}"
        ),
        (
            "- Pairwise generated-code mismatch fraction: "
            f"median {diversity['pairwise_code_mismatch_fraction_median']:.3f}, "
            f"minimum {diversity['pairwise_code_mismatch_fraction_min']:.3f}"
        ),
        "",
        "The sample set is not a collection of exact duplicates, although visual quality must "
        "be assessed separately from numerical diversity.",
        "",
        "## Distribution coverage",
        "",
        (
            f"The samples use {coverage['generated_active_codes']} of the "
            f"{coverage['train_active_codes']} codes active in the training export "
            f"({100.0 * coverage['generated_coverage_of_train_active_codes']:.1f}% coverage)."
        ),
        (
            f"Generated-code perplexity is {coverage['generated_code_perplexity']:.3f}, "
            f"compared with {coverage['train_code_perplexity']:.3f} for training codes; "
            f"code-frequency Jensen-Shannon divergence is "
            f"{coverage['code_jensen_shannon_divergence_nats']:.4f} nats."
        ),
        "",
        (
            "Image-space coverage is substantially weaker. The generated near-zero fraction "
            f"is {feature_metrics['near_zero_fraction']['standardised_mean_difference']:.2f} "
            "training standard deviations above the training mean, while tissue fraction is "
            f"{abs(feature_metrics['tissue_fraction']['standardised_mean_difference']):.2f} "
            "standard deviations below it. The vertical centre of mass is shifted by "
            f"{abs(feature_metrics['vertical_centre_of_mass']['standardised_mean_difference']):.2f} "
            "standard deviations. These shifts agree with the large black bands and fragmented "
            "anatomy visible in the sample panel."
        ),
        "",
        *failure_lines,
        "## Latent interpolation diagnostic",
        "",
        (
            f"{summary['latent_interpolation']['pair_count']} generated/training-neighbour "
            f"pairs were selected at fixed nearest-neighbour-distance ranks. Each pair was "
            f"decoded at {summary['latent_interpolation']['steps']} evenly spaced points between "
            "its training and generated codebook-embedding grids."
        ),
        (
            "Adjacent interpolation-frame SSIM has median row mean "
            f"{summary['latent_interpolation']['adjacent_ssim_mean_median']:.3f} and "
            f"worst observed value {summary['latent_interpolation']['adjacent_ssim_min']:.3f}."
        ),
        "",
        (
            "The interpolation is a decoder-continuity diagnostic, not a new sampling method: "
            "intermediate embedding tensors generally do not correspond to discrete codebook "
            "entries and may therefore be off the learned VQ support."
        ),
        "",
        "## Conclusion",
        "",
        (
            "The learned prior enables reproducible unconditional sampling and shows numerical "
            "diversity without detected exact training-set copies. However, code-frequency "
            "coverage does not translate into anatomically coherent images. The generated cohort "
            "should therefore be reported as a technically successful but low-quality negative "
            "result, not as clinically realistic synthetic MRI."
        ),
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def validate_images(images: np.ndarray, name: str) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError(f"{name} must have shape [N, H, W], received {images.shape}.")
    if len(images) < 2:
        raise ValueError(f"{name} must contain at least two images.")
    if not np.isfinite(images).all():
        raise ValueError(f"{name} contains non-finite values.")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise ValueError(f"{name} must be scaled to [0, 1].")
    return images


def image_features(images: np.ndarray) -> np.ndarray:
    images = validate_images(images, "images")
    height, width = images.shape[1:]
    flattened = images.reshape(len(images), -1)
    mass = flattened.sum(axis=1).clip(min=1e-12)
    vertical_coordinates = np.linspace(0.0, 1.0, height, dtype=np.float32)
    horizontal_coordinates = np.linspace(0.0, 1.0, width, dtype=np.float32)
    vertical_mass = images.sum(axis=2)
    horizontal_mass = images.sum(axis=1)
    return np.column_stack(
        (
            flattened.mean(axis=1),
            flattened.std(axis=1),
            (flattened < 0.01).mean(axis=1),
            (flattened > 0.05).mean(axis=1),
            np.quantile(flattened, 0.95, axis=1),
            (vertical_mass * vertical_coordinates[None, :]).sum(axis=1) / mass,
            (horizontal_mass * horizontal_coordinates[None, :]).sum(axis=1) / mass,
        )
    ).astype(np.float64, copy=False)


def downsample(images: np.ndarray, factor: int) -> np.ndarray:
    tensor = torch.from_numpy(images[:, None])
    if tensor.shape[-2] % factor or tensor.shape[-1] % factor:
        raise ValueError(
            f"Image shape {tuple(tensor.shape[-2:])} is not divisible by "
            f"--downsample-factor={factor}."
        )
    pooled = F.avg_pool2d(tensor, kernel_size=factor, stride=factor)
    return pooled[:, 0].numpy().astype(np.float32, copy=False)


def squared_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float32, copy=False)
    right = right.reshape(len(right), -1).astype(np.float32, copy=False)
    squared = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * (left @ right.T)
    )
    return np.maximum(squared, 0.0) / left.shape[1]


def code_statistics(indices: np.ndarray, num_embeddings: int) -> dict[str, Any]:
    counts = np.bincount(indices.reshape(-1).astype(np.int64), minlength=num_embeddings)
    probabilities = counts.astype(np.float64) / counts.sum()
    active = probabilities > 0
    perplexity = float(
        np.exp(-np.sum(probabilities[active] * np.log(probabilities[active])))
    )
    return {
        "counts": counts,
        "probabilities": probabilities,
        "active_codes": int(active.sum()),
        "perplexity": perplexity,
    }


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl_divergence(distribution: np.ndarray) -> float:
        nonzero = distribution > 0
        return float(
            np.sum(
                distribution[nonzero]
                * np.log(distribution[nonzero] / midpoint[nonzero])
            )
        )

    return 0.5 * (kl_divergence(left) + kl_divergence(right))


def approximate_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, 1001)
    return float(
        np.mean(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles)))
    )


def distribution_rows(
    train_features: np.ndarray, generated_features: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        training = train_features[:, feature_index]
        generated = generated_features[:, feature_index]
        train_standard_deviation = float(training.std())
        lower, upper = np.quantile(training, [0.05, 0.95])
        rows.append(
            {
                "feature": feature_name,
                "train_mean": float(training.mean()),
                "train_std": train_standard_deviation,
                "train_p05": float(lower),
                "train_p95": float(upper),
                "generated_mean": float(generated.mean()),
                "generated_std": float(generated.std()),
                "standardised_mean_difference": (
                    float(
                        (generated.mean() - training.mean()) / train_standard_deviation
                    )
                    if train_standard_deviation > 0
                    else float("nan")
                ),
                "generated_fraction_inside_train_p05_p95": float(
                    np.mean((generated >= lower) & (generated <= upper))
                ),
                "quantile_wasserstein_distance": approximate_wasserstein(
                    training, generated
                ),
            }
        )
    return rows


def load_training_data(
    manifest_path: Path,
    metadata_path: Path,
    batch_size: int,
    num_workers: int,
    downsample_factor: int,
) -> tuple[HipMRISliceDataset, np.ndarray, np.ndarray, list[dict[str, str]]]:
    if num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    dataset = HipMRISliceDataset(manifest_path, "train")
    metadata = read_csv(metadata_path)
    if len(metadata) != len(dataset):
        raise ValueError(
            "Training metadata and manifest contain different sample counts."
        )
    for index, (metadata_row, manifest_row) in enumerate(zip(metadata, dataset.rows)):
        expected = (
            str(index),
            manifest_row["subject_id"],
            str(int(manifest_row["week"])),
            str(int(manifest_row["slice_index"])),
        )
        observed = (
            metadata_row.get("sample_index"),
            metadata_row.get("subject_id"),
            metadata_row.get("week"),
            metadata_row.get("slice_index"),
        )
        if observed != expected:
            raise ValueError(
                f"Training metadata order differs from the manifest at index {index}: "
                f"observed={observed}, expected={expected}."
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    feature_batches: list[np.ndarray] = []
    downsampled_batches: list[np.ndarray] = []
    loaded = 0
    for batch in loader:
        images = batch["image"][:, 0].numpy().astype(np.float32, copy=False)
        feature_batches.append(image_features(images))
        downsampled_batches.append(downsample(images, downsample_factor))
        loaded += len(images)
        if loaded % (batch_size * 20) == 0 or loaded == len(dataset):
            print(f"loaded {loaded}/{len(dataset)} training slices", flush=True)
    return (
        dataset,
        np.concatenate(feature_batches, axis=0),
        np.concatenate(downsampled_batches, axis=0),
        metadata,
    )


def code_nearest_neighbours(
    generated_codes: np.ndarray,
    training_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest_indices = np.empty(len(generated_codes), dtype=np.int64)
    mismatch_fractions = np.empty(len(generated_codes), dtype=np.float64)
    tie_counts = np.empty(len(generated_codes), dtype=np.int64)
    for sample_index, generated in enumerate(generated_codes):
        mismatches = np.count_nonzero(
            training_codes != generated[None, ...], axis=(1, 2)
        )
        minimum = int(mismatches.min())
        nearest_indices[sample_index] = int(np.flatnonzero(mismatches == minimum)[0])
        mismatch_fractions[sample_index] = minimum / generated.size
        tie_counts[sample_index] = int(np.count_nonzero(mismatches == minimum))
    return nearest_indices, mismatch_fractions, tie_counts


def pairwise_summary(images: np.ndarray, code_indices: np.ndarray) -> dict[str, Any]:
    image_distances = squared_distance_matrix(images, images)
    np.fill_diagonal(image_distances, np.inf)
    nearest_image = image_distances.min(axis=1)
    upper_image = image_distances[np.triu_indices(len(images), k=1)]

    code_mismatches = np.empty((len(code_indices), len(code_indices)), dtype=np.float64)
    for index, code_grid in enumerate(code_indices):
        code_mismatches[index] = np.mean(
            code_indices != code_grid[None, ...], axis=(1, 2)
        )
    np.fill_diagonal(code_mismatches, np.inf)
    nearest_code = code_mismatches.min(axis=1)
    upper_code = code_mismatches[np.triu_indices(len(code_indices), k=1)]

    unique_images = len(np.unique(images.reshape(len(images), -1), axis=0))
    unique_codes = len(np.unique(code_indices.reshape(len(code_indices), -1), axis=0))
    return {
        "exact_duplicate_images": int(len(images) - unique_images),
        "exact_duplicate_code_grids": int(len(code_indices) - unique_codes),
        "pairwise_pixel_mse_min": float(upper_image.min()),
        "pairwise_pixel_mse_median": float(np.median(upper_image)),
        "pairwise_pixel_mse_mean": float(upper_image.mean()),
        "nearest_generated_pixel_mse_median": float(np.median(nearest_image)),
        "pairwise_code_mismatch_fraction_min": float(upper_code.min()),
        "pairwise_code_mismatch_fraction_median": float(np.median(upper_code)),
        "pairwise_code_mismatch_fraction_mean": float(upper_code.mean()),
        "nearest_generated_code_mismatch_fraction_median": float(
            np.median(nearest_code)
        ),
    }


def save_neighbour_panel(
    path: Path,
    generated_images: np.ndarray,
    nearest_training_images: np.ndarray,
    audit_rows: list[dict[str, Any]],
    panel_samples: int,
) -> None:
    shown_indices = np.argsort([row["full_resolution_mse"] for row in audit_rows])[
        : min(panel_samples, len(audit_rows))
    ]
    comparisons_per_row = 2
    panel_rows = math.ceil(len(shown_indices) / comparisons_per_row)
    figure, axes = plt.subplots(
        panel_rows,
        3 * comparisons_per_row,
        figsize=(18.0, 3.2 * panel_rows),
        squeeze=False,
    )
    for display_index, sample_index in enumerate(shown_indices):
        row_index = display_index // comparisons_per_row
        column_start = 3 * (display_index % comparisons_per_row)
        generated = generated_images[sample_index]
        neighbour = nearest_training_images[sample_index]
        difference = np.abs(generated - neighbour)
        row = audit_rows[sample_index]
        generated_axis = axes[row_index, column_start]
        neighbour_axis = axes[row_index, column_start + 1]
        difference_axis = axes[row_index, column_start + 2]
        generated_axis.imshow(generated, cmap="gray", vmin=0.0, vmax=1.0)
        generated_axis.set_title(
            f"Generated sample {int(row['sample_id']) + 1}", fontsize=9
        )
        neighbour_axis.imshow(neighbour, cmap="gray", vmin=0.0, vmax=1.0)
        neighbour_axis.set_title(
            "Nearest training slice\n"
            f"patient {row['pixel_nn_subject_id']}, week {row['pixel_nn_week']}, "
            f"slice {row['pixel_nn_slice_index']}",
            fontsize=9,
        )
        difference_axis.imshow(difference, cmap="magma", vmin=0.0, vmax=1.0)
        difference_axis.set_title(
            f"Absolute error\nMSE={row['full_resolution_mse']:.4f}, "
            f"SSIM={row['full_resolution_ssim']:.3f}",
            fontsize=9,
        )
        for axis in (generated_axis, neighbour_axis, difference_axis):
            axis.axis("off")
    for empty_index in range(len(shown_indices), panel_rows * comparisons_per_row):
        empty_row = empty_index // comparisons_per_row
        empty_column = 3 * (empty_index % comparisons_per_row)
        for axis in axes[empty_row, empty_column : empty_column + 3]:
            axis.axis("off")
    figure.suptitle("Generated samples and nearest training slices", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_sample_panel(
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
    columns = 4
    rows = math.ceil(len(shown_indices) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.5, 3.1 * rows),
        squeeze=False,
    )
    for axis_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if axis_index < len(shown_indices):
            sample_index = int(shown_indices[axis_index])
            axis.imshow(images[sample_index], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(
                f"seed {int(sample_seeds[sample_index])}, "
                f"sample {int(sample_indices_within_seed[sample_index]) + 1}",
                fontsize=9,
            )
    figure.suptitle("Uncurated unconditional VQ-VAE samples across seeds", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_failure_panel(
    path: Path,
    generated_images: np.ndarray,
    nearest_training_images: np.ndarray,
    failure_indices: np.ndarray,
    failure_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        len(failure_indices),
        3,
        figsize=(10.5, 3.0 * len(failure_indices)),
        squeeze=False,
    )
    for row_index, (sample_index, row) in enumerate(zip(failure_indices, failure_rows)):
        generated = generated_images[int(sample_index)]
        neighbour = nearest_training_images[int(sample_index)]
        difference = np.abs(generated - neighbour)
        axes[row_index, 0].imshow(generated, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 0].set_title(
            f"Case {row_index + 1}: seed {row['sample_seed']}, "
            f"sample {row['sample_index_within_seed'] + 1}\n"
            f"{row['dominant_feature']} z={row['dominant_feature_z']:.2f}",
            fontsize=9,
        )
        axes[row_index, 1].imshow(neighbour, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].set_title(
            f"Nearest training slice\npatient {row['pixel_nn_subject_id']}, "
            f"week {row['pixel_nn_week']}, slice {row['pixel_nn_slice_index']}",
            fontsize=9,
        )
        axes[row_index, 2].imshow(difference, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_index, 2].set_title(
            f"Absolute error\nMSE={row['full_resolution_mse']:.4f}, "
            f"SSIM={row['full_resolution_ssim']:.3f}",
            fontsize=9,
        )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(
        f"{len(failure_indices)} deterministic VQ-VAE generation failures "
        "by feature-deviation score",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.992))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def fixed_rank_indices(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Select deterministic ranks spanning the complete sorted value range."""

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Rank selection requires a non-empty one-dimensional array.")
    if count > len(values):
        raise ValueError(
            "--interpolation-pairs cannot exceed the generated sample count."
        )
    rank_positions = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    selected = np.argsort(values, kind="stable")[rank_positions]
    quantiles = rank_positions.astype(np.float64) / max(len(values) - 1, 1)
    return selected, quantiles


def load_vqvae_decoder(
    checkpoint_path: Path,
    expected_sha256: str,
    device: torch.device,
) -> tuple[VQVAE, dict[str, Any], str]:
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError(
            "The supplied VQ-VAE checkpoint does not match the checkpoint used for "
            "the training-code export."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "vqvae":
        raise ValueError("The interpolation checkpoint is not a VQ-VAE checkpoint.")
    model = VQVAE()
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    if not bool(model.quantizer.codebook_initialized.item()):
        raise ValueError("The VQ-VAE codebook is not marked as initialised.")
    return model, checkpoint, checkpoint_sha256


def latent_interpolation(
    model: VQVAE,
    generated_codes: np.ndarray,
    training_codes: np.ndarray,
    generated_images: np.ndarray,
    nearest_training_images: np.ndarray,
    pixel_nearest_indices: np.ndarray,
    metadata: list[dict[str, str]],
    nearest_mse: np.ndarray,
    pair_count: int,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    if steps < 2:
        raise ValueError("--interpolation-steps must be at least two.")
    selected, selection_quantiles = fixed_rank_indices(nearest_mse, pair_count)
    generated_selected = torch.from_numpy(
        generated_codes[selected].astype(np.int64, copy=False)
    ).to(device)
    training_selected_indices = pixel_nearest_indices[selected].astype(
        np.int64, copy=False
    )
    training_selected = torch.from_numpy(
        training_codes[training_selected_indices].astype(np.int64, copy=False)
    ).to(device)
    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    with torch.inference_mode():
        training_embeddings = model.quantizer.embedding(training_selected)
        generated_embeddings = model.quantizer.embedding(generated_selected)
        interpolated_embeddings = (
            1.0 - alphas[None, :, None, None, None]
        ) * training_embeddings[:, None] + alphas[
            None, :, None, None, None
        ] * generated_embeddings[
            :, None
        ]
        flat_embeddings = interpolated_embeddings.reshape(
            pair_count * steps,
            *interpolated_embeddings.shape[2:],
        )
        flat_embeddings = flat_embeddings.permute(0, 3, 1, 2).contiguous()
        decoded = model.decoder(flat_embeddings)
    decoded_numpy = (
        decoded[:, 0]
        .reshape(pair_count, steps, decoded.shape[-2], decoded.shape[-1])
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if not np.isfinite(decoded_numpy).all():
        raise RuntimeError("Latent interpolation produced non-finite decoded images.")

    adjacent_left = torch.from_numpy(decoded_numpy[:, :-1, None]).reshape(
        pair_count * (steps - 1), 1, decoded.shape[-2], decoded.shape[-1]
    )
    adjacent_right = torch.from_numpy(decoded_numpy[:, 1:, None]).reshape_as(
        adjacent_left
    )
    adjacent_metrics = reconstruction_metrics(adjacent_left, adjacent_right)
    adjacent_mse = adjacent_metrics["mse"].reshape(pair_count, steps - 1).numpy()
    adjacent_ssim = adjacent_metrics["ssim"].reshape(pair_count, steps - 1).numpy()

    training_endpoint_metrics = reconstruction_metrics(
        torch.from_numpy(nearest_training_images[selected][:, None]),
        torch.from_numpy(decoded_numpy[:, 0, None]),
    )
    endpoint_metrics = reconstruction_metrics(
        torch.from_numpy(decoded_numpy[:, 0, None]),
        torch.from_numpy(decoded_numpy[:, -1, None]),
    )
    generated_endpoint_errors = np.abs(
        decoded_numpy[:, -1] - generated_images[selected]
    )
    rows: list[dict[str, Any]] = []
    for row_index, sample_index in enumerate(selected):
        train_index = int(training_selected_indices[row_index])
        metadata_row = metadata[train_index]
        rows.append(
            {
                "selection_quantile": float(selection_quantiles[row_index]),
                "generated_sample_id": int(sample_index),
                "training_index": train_index,
                "training_subject_id": metadata_row["subject_id"],
                "training_week": int(metadata_row["week"]),
                "training_slice_index": int(metadata_row["slice_index"]),
                "nearest_training_full_resolution_mse": float(
                    nearest_mse[sample_index]
                ),
                "endpoint_code_mismatch_fraction": float(
                    np.mean(
                        generated_codes[sample_index] != training_codes[train_index]
                    )
                ),
                "training_image_to_code_decode_ssim": float(
                    training_endpoint_metrics["ssim"][row_index]
                ),
                "interpolation_endpoint_ssim": float(
                    endpoint_metrics["ssim"][row_index]
                ),
                "generated_endpoint_max_abs_error": float(
                    generated_endpoint_errors[row_index].max()
                ),
                "adjacent_mse_mean": float(adjacent_mse[row_index].mean()),
                "adjacent_mse_max": float(adjacent_mse[row_index].max()),
                "adjacent_ssim_mean": float(adjacent_ssim[row_index].mean()),
                "adjacent_ssim_min": float(adjacent_ssim[row_index].min()),
            }
        )

    generated_endpoint_error = float(generated_endpoint_errors.max())
    if generated_endpoint_error > 1e-3:
        raise RuntimeError(
            "The generated interpolation endpoint does not reproduce the saved sample "
            "within the cross-device numerical tolerance."
        )
    return decoded_numpy, rows, selected


def save_interpolation_panel(
    path: Path,
    interpolated_images: np.ndarray,
    interpolation_rows: list[dict[str, Any]],
) -> None:
    pair_count, steps = interpolated_images.shape[:2]
    alphas = np.linspace(0.0, 1.0, steps)
    figure, axes = plt.subplots(
        pair_count,
        steps,
        figsize=(2.55 * steps, 3.15 * pair_count),
        squeeze=False,
    )
    for row_index, row in enumerate(interpolation_rows):
        for step_index, alpha in enumerate(alphas):
            axis = axes[row_index, step_index]
            axis.imshow(
                interpolated_images[row_index, step_index],
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            if row_index == 0:
                endpoint = "\ntraining-code decode" if step_index == 0 else ""
                if step_index == steps - 1:
                    endpoint = "\ngenerated sample"
                axis.set_title(f"alpha={alpha:.2f}{endpoint}", fontsize=9)
            axis.axis("off")
        axes[row_index, 0].text(
            -0.10,
            0.5,
            (
                f"rank q={row['selection_quantile']:.2f}\n"
                f"sample {int(row['generated_sample_id']) + 1}\n"
                f"train {row['training_subject_id']} / w{row['training_week']} / "
                f"s{row['training_slice_index']}"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
    figure.suptitle(
        "Frozen-decoder interpolation from nearest training-code grids to generated-code grids",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples_path = args.samples.resolve()
    sample_summary_path = (
        args.sample_summary.resolve()
        if args.sample_summary is not None
        else (samples_path.parent / "summary.json").resolve()
    )
    train_codes_path = args.train_codes.resolve()
    metadata_path = args.train_metadata.resolve()
    train_summary_path = args.train_summary.resolve()
    prior_summary_path = args.prior_summary.resolve()
    vqvae_checkpoint_path = args.vqvae_checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    for path in (
        samples_path,
        sample_summary_path,
        train_codes_path,
        metadata_path,
        train_summary_path,
        prior_summary_path,
        vqvae_checkpoint_path,
        manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    prepare_output(output, args.overwrite)

    device = select_device(args.device)

    sample_summary = read_json(sample_summary_path)
    train_summary = read_json(train_summary_path)
    prior_summary = read_json(prior_summary_path)
    train_codes_sha256 = sha256_file(train_codes_path)
    expected_train_codes_sha256 = prior_summary.get("provenance", {}).get("data_sha256")
    if train_summary.get("source_split") != "train":
        raise ValueError("The code export is not labelled as training-only.")
    if train_summary.get("limited_export") is not False:
        raise ValueError("A limited code export cannot be used for the formal audit.")
    if expected_train_codes_sha256 != train_codes_sha256:
        raise ValueError(
            "The training code archive does not match the archive used to fit the formal prior: "
            f"expected {expected_train_codes_sha256}, received {train_codes_sha256}."
        )
    if prior_summary.get("provenance", {}).get(
        "source_vqvae_sha256"
    ) != train_summary.get("checkpoint", {}).get("sha256"):
        raise ValueError(
            "The prior and code-export summaries refer to different VQ-VAE checkpoints."
        )
    expected_vqvae_sha256 = str(train_summary.get("checkpoint", {}).get("sha256", ""))
    if not expected_vqvae_sha256:
        raise ValueError(
            "The training-code summary has no source VQ-VAE checkpoint hash."
        )
    vqvae, vqvae_checkpoint, vqvae_sha256 = load_vqvae_decoder(
        vqvae_checkpoint_path,
        expected_vqvae_sha256,
        device,
    )

    with np.load(samples_path) as archive:
        generated_images = validate_images(archive["images"], "generated images")
        generated_codes = np.asarray(archive["code_indices"], dtype=np.uint16)
        sample_ids = np.asarray(archive["sample_id"], dtype=np.int64)
        sample_seeds = (
            np.asarray(archive["sample_seed"], dtype=np.int64)
            if "sample_seed" in archive.files
            else np.full(
                len(sample_ids), int(sample_summary.get("seed", -1)), dtype=np.int64
            )
        )
        sample_indices_within_seed = (
            np.asarray(archive["sample_index_within_seed"], dtype=np.int64)
            if "sample_index_within_seed" in archive.files
            else sample_ids.copy()
        )
    with np.load(train_codes_path) as archive:
        if "indices" not in archive.files:
            raise ValueError("Training code archive has no 'indices' array.")
        training_codes = np.asarray(archive["indices"], dtype=np.uint16)
        archive_subject_ids = np.asarray(archive["subject_id"]).astype(str)
        archive_weeks = np.asarray(archive["week"], dtype=np.int64)
        archive_slice_indices = np.asarray(archive["slice_index"], dtype=np.int64)
    if generated_codes.ndim != 3 or training_codes.ndim != 3:
        raise ValueError("Code-index arrays must have shape [N, H, W].")
    if generated_codes.shape[1:] != training_codes.shape[1:]:
        raise ValueError("Generated and training code grids have different shapes.")
    if not (
        len(generated_images)
        == len(generated_codes)
        == len(sample_ids)
        == len(sample_seeds)
        == len(sample_indices_within_seed)
    ):
        raise ValueError("Generated images, code grids, and sample IDs are misaligned.")
    if not np.array_equal(sample_ids, np.arange(len(sample_ids))):
        raise ValueError("Generated sample IDs are not contiguous and deterministic.")
    if int(sample_summary.get("num_samples", -1)) != len(generated_images):
        raise ValueError(
            "The sampling summary and sample archive have different sample counts."
        )
    if (
        sample_summary.get("vqvae_checkpoint", {}).get("sha256")
        and sample_summary["vqvae_checkpoint"]["sha256"] != vqvae_sha256
    ):
        raise ValueError(
            "The sampling summary refers to a different VQ-VAE checkpoint."
        )
    generation_profile = sample_summary.get("generation_profile")
    if not isinstance(generation_profile, dict):
        raise ValueError("The sampling summary has no measured generation profile.")

    dataset, training_features, training_downsampled, metadata = load_training_data(
        manifest_path,
        metadata_path,
        args.batch_size,
        args.num_workers,
        args.downsample_factor,
    )
    if len(training_codes) != len(dataset):
        raise ValueError(
            "Training code archive and manifest contain different sample counts."
        )
    if int(train_summary.get("sample_count", -1)) != len(dataset):
        raise ValueError("Training code summary has an unexpected sample count.")
    metadata_subject_ids = np.asarray([row["subject_id"] for row in metadata])
    metadata_weeks = np.asarray([int(row["week"]) for row in metadata], dtype=np.int64)
    metadata_slice_indices = np.asarray(
        [int(row["slice_index"]) for row in metadata], dtype=np.int64
    )
    if not (
        np.array_equal(archive_subject_ids, metadata_subject_ids)
        and np.array_equal(archive_weeks, metadata_weeks)
        and np.array_equal(archive_slice_indices, metadata_slice_indices)
    ):
        raise ValueError(
            "Training code archive metadata does not match train_samples.csv."
        )

    generated_features = image_features(generated_images)
    generated_downsampled = downsample(generated_images, args.downsample_factor)
    pixel_distances = squared_distance_matrix(
        training_downsampled, generated_downsampled
    )
    pixel_nearest_indices = pixel_distances.argmin(axis=0)
    pixel_nearest_mse = pixel_distances[
        pixel_nearest_indices, np.arange(len(generated_images))
    ]
    code_indices, code_mismatch, code_ties = code_nearest_neighbours(
        generated_codes,
        training_codes,
    )

    nearest_training_images = np.stack(
        [dataset[int(index)]["image"][0].numpy() for index in pixel_nearest_indices]
    ).astype(np.float32, copy=False)
    full_metrics = reconstruction_metrics(
        torch.from_numpy(nearest_training_images[:, None]),
        torch.from_numpy(generated_images[:, None]),
    )
    full_metrics_numpy = {name: value.numpy() for name, value in full_metrics.items()}

    audit_rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        pixel_index = int(pixel_nearest_indices[sample_index])
        code_index = int(code_indices[sample_index])
        pixel_row = metadata[pixel_index]
        code_row = metadata[code_index]
        audit_rows.append(
            {
                "sample_id": int(sample_id),
                "sample_seed": int(sample_seeds[sample_index]),
                "sample_index_within_seed": int(
                    sample_indices_within_seed[sample_index]
                ),
                "pixel_nn_train_index": pixel_index,
                "pixel_nn_subject_id": pixel_row["subject_id"],
                "pixel_nn_week": int(pixel_row["week"]),
                "pixel_nn_slice_index": int(pixel_row["slice_index"]),
                "downsampled_pixel_mse": float(pixel_nearest_mse[sample_index]),
                "full_resolution_mse": float(full_metrics_numpy["mse"][sample_index]),
                "full_resolution_mae": float(full_metrics_numpy["mae"][sample_index]),
                "full_resolution_psnr": float(full_metrics_numpy["psnr"][sample_index]),
                "full_resolution_ssim": float(full_metrics_numpy["ssim"][sample_index]),
                "code_nn_train_index": code_index,
                "code_nn_subject_id": code_row["subject_id"],
                "code_nn_week": int(code_row["week"]),
                "code_nn_slice_index": int(code_row["slice_index"]),
                "code_nn_mismatch_fraction": float(code_mismatch[sample_index]),
                "code_nn_tie_count": int(code_ties[sample_index]),
                "exact_pixel_match": bool(
                    full_metrics_numpy["mse"][sample_index] <= 1e-12
                ),
                "exact_code_grid_match": bool(code_mismatch[sample_index] == 0.0),
            }
        )

    distribution = distribution_rows(training_features, generated_features)
    training_feature_mean = training_features.mean(axis=0)
    training_feature_std = training_features.std(axis=0)
    safe_feature_std = np.where(training_feature_std > 0, training_feature_std, 1.0)
    feature_z_scores = (generated_features - training_feature_mean) / safe_feature_std
    failure_scores = np.max(np.abs(feature_z_scores), axis=1)
    failure_count = min(args.failure_cases, len(generated_images))
    failure_indices = np.argsort(-failure_scores, kind="stable")[:failure_count]
    failure_rows: list[dict[str, Any]] = []
    for sample_index in failure_indices:
        dominant_index = int(np.argmax(np.abs(feature_z_scores[sample_index])))
        row = {
            **audit_rows[int(sample_index)],
            "failure_score": float(failure_scores[sample_index]),
            "dominant_feature": FEATURE_NAMES[dominant_index],
            "dominant_feature_z": float(feature_z_scores[sample_index, dominant_index]),
        }
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            row[f"{feature_name}_z"] = float(
                feature_z_scores[sample_index, feature_index]
            )
        failure_rows.append(row)
    train_code_statistics = code_statistics(training_codes, 512)
    generated_code_statistics = code_statistics(generated_codes, 512)
    train_active = train_code_statistics["counts"] > 0
    generated_active = generated_code_statistics["counts"] > 0

    neighbour_csv = output / "nearest_training_neighbours.csv"
    distribution_csv = output / "distribution_metrics.csv"
    neighbour_panel = output / "nearest_training_neighbours.png"
    sample_panel = output / "unconditional_samples.png"
    interpolation_csv = output / "latent_interpolation_metrics.csv"
    interpolation_panel = output / "latent_interpolation.png"
    failure_csv = output / "generation_failure_cases.csv"
    failure_panel = output / "generation_failure_cases.png"
    report_path = output / "REPORT_RESULTS.md"
    write_csv(neighbour_csv, audit_rows)
    write_csv(distribution_csv, distribution)
    write_csv(failure_csv, failure_rows)
    save_neighbour_panel(
        neighbour_panel,
        generated_images,
        nearest_training_images,
        audit_rows,
        args.panel_samples,
    )
    save_sample_panel(
        sample_panel,
        generated_images,
        sample_seeds,
        sample_indices_within_seed,
    )
    save_failure_panel(
        failure_panel,
        generated_images,
        nearest_training_images,
        failure_indices,
        failure_rows,
    )

    nearest_mse = np.asarray([row["full_resolution_mse"] for row in audit_rows])
    nearest_ssim = np.asarray([row["full_resolution_ssim"] for row in audit_rows])
    interpolated_images, interpolation_rows, interpolation_sample_indices = (
        latent_interpolation(
            vqvae,
            generated_codes,
            training_codes,
            generated_images,
            nearest_training_images,
            pixel_nearest_indices,
            metadata,
            nearest_mse,
            args.interpolation_pairs,
            args.interpolation_steps,
            device,
        )
    )
    write_csv(interpolation_csv, interpolation_rows)
    save_interpolation_panel(
        interpolation_panel,
        interpolated_images,
        interpolation_rows,
    )
    adjacent_ssim_means = np.asarray(
        [row["adjacent_ssim_mean"] for row in interpolation_rows], dtype=np.float64
    )
    adjacent_ssim_minima = np.asarray(
        [row["adjacent_ssim_min"] for row in interpolation_rows], dtype=np.float64
    )
    summary = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "reference_split": "train",
            "generated_samples": int(len(generated_images)),
            "sampling_seeds": list(dict.fromkeys(int(seed) for seed in sample_seeds)),
            "samples_per_seed": {
                str(seed): int(np.count_nonzero(sample_seeds == seed))
                for seed in dict.fromkeys(int(value) for value in sample_seeds)
            },
            "training_samples": int(len(dataset)),
            "patient_disjoint_manifest_retained": True,
            "formal_prior_data_hash_verified": True,
            "pixel_search": (
                f"MSE on deterministic {args.downsample_factor}x average-pooled "
                "normalised slices"
            ),
            "pixel_confirmation": "Full-resolution MSE, MAE, PSNR, and SSIM",
            "code_search": "Hamming mismatch fraction on 32x18 VQ index grids",
            "interpolation": (
                "Linear interpolation between frozen VQ codebook-embedding grids; "
                "intermediate points are not re-quantised"
            ),
        },
        "inputs": {
            "samples": {"path": str(samples_path), "sha256": sha256_file(samples_path)},
            "sample_summary": {
                "path": str(sample_summary_path),
                "sha256": sha256_file(sample_summary_path),
            },
            "train_codes": {
                "path": str(train_codes_path),
                "sha256": train_codes_sha256,
            },
            "train_metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "train_summary": {
                "path": str(train_summary_path),
                "sha256": sha256_file(train_summary_path),
            },
            "prior_summary": {
                "path": str(prior_summary_path),
                "sha256": sha256_file(prior_summary_path),
            },
            "vqvae_checkpoint": {
                "path": str(vqvae_checkpoint_path),
                "sha256": vqvae_sha256,
                "epoch": int(vqvae_checkpoint["epoch"]),
                "best_epoch": int(vqvae_checkpoint["best_epoch"]),
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
        "generation_profile": {
            "seconds": float(generation_profile["seconds"]),
            "milliseconds_per_image": float(
                generation_profile["milliseconds_per_image"]
            ),
            "images_per_second": float(generation_profile["images_per_second"]),
            "peak_accelerator_memory_mb": generation_profile.get(
                "peak_accelerator_memory_mb"
            ),
            "device_name": str(sample_summary.get("device", {}).get("name", "unknown")),
        },
        "memorisation_audit": {
            "exact_pixel_matches": int(
                sum(row["exact_pixel_match"] for row in audit_rows)
            ),
            "exact_code_grid_matches": int(
                sum(row["exact_code_grid_match"] for row in audit_rows)
            ),
            "nearest_training_full_resolution_mse_min": float(nearest_mse.min()),
            "nearest_training_full_resolution_mse_median": float(
                np.median(nearest_mse)
            ),
            "nearest_training_full_resolution_mse_max": float(nearest_mse.max()),
            "nearest_training_full_resolution_ssim_min": float(nearest_ssim.min()),
            "nearest_training_full_resolution_ssim_median": float(
                np.median(nearest_ssim)
            ),
            "nearest_training_full_resolution_ssim_max": float(nearest_ssim.max()),
            "nearest_training_code_mismatch_fraction_min": float(code_mismatch.min()),
            "nearest_training_code_mismatch_fraction_median": float(
                np.median(code_mismatch)
            ),
            "nearest_training_code_mismatch_fraction_max": float(code_mismatch.max()),
        },
        "diversity": pairwise_summary(generated_images, generated_codes),
        "distribution_coverage": {
            "train_active_codes": int(train_active.sum()),
            "generated_active_codes": int(generated_active.sum()),
            "generated_coverage_of_train_active_codes": float(
                np.sum(train_active & generated_active) / train_active.sum()
            ),
            "train_code_perplexity": float(train_code_statistics["perplexity"]),
            "generated_code_perplexity": float(generated_code_statistics["perplexity"]),
            "code_jensen_shannon_divergence_nats": jensen_shannon_divergence(
                train_code_statistics["probabilities"],
                generated_code_statistics["probabilities"],
            ),
            "image_feature_metrics": distribution,
        },
        "failure_autopsy": {
            "case_count": len(failure_rows),
            "selection_rule": (
                "Largest maximum absolute standardised deviation across seven predefined "
                "image features; stable sample-ID tie break; no visual selection"
            ),
            "rows": failure_rows,
        },
        "latent_interpolation": {
            "pair_count": int(len(interpolation_rows)),
            "steps": int(args.interpolation_steps),
            "alphas": np.linspace(0.0, 1.0, args.interpolation_steps).tolist(),
            "device": {
                "requested": args.device,
                "resolved": str(device),
                "name": describe_device(device),
            },
            "selection_rule": (
                "Evenly spaced ranks across generated-sample nearest-training full-resolution MSE"
            ),
            "selected_generated_sample_ids": interpolation_sample_indices.astype(
                int
            ).tolist(),
            "intermediate_embeddings_requantised": False,
            "generated_endpoint_max_abs_error": float(
                max(
                    row["generated_endpoint_max_abs_error"]
                    for row in interpolation_rows
                )
            ),
            "adjacent_ssim_mean_median": float(np.median(adjacent_ssim_means)),
            "adjacent_ssim_min": float(adjacent_ssim_minima.min()),
            "rows": interpolation_rows,
        },
        "limitations": [
            "Pixel-space nearest-neighbour search is sensitive to padding and intensity alignment.",
            "No exact or close neighbour is evidence against direct copying, not a privacy guarantee.",
            (
                f"{len(generated_images)} samples across {len(set(sample_seeds.tolist()))} "
                "seeds support an engineering audit but not clinical validation."
            ),
            (
                "Linear embedding interpolation probes decoder continuity but intermediate "
                "latents can lie outside the discrete codebook support."
            ),
        ],
        "artifacts": {
            "nearest_training_neighbours": neighbour_csv.name,
            "distribution_metrics": distribution_csv.name,
            "nearest_training_neighbour_panel": neighbour_panel.name,
            "unconditional_sample_panel": sample_panel.name,
            "latent_interpolation_metrics": interpolation_csv.name,
            "latent_interpolation_panel": interpolation_panel.name,
            "generation_failure_cases": failure_csv.name,
            "generation_failure_panel": failure_panel.name,
            "text_report": report_path.name,
        },
    }
    write_report(report_path, summary)
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
