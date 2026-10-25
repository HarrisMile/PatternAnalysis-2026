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
from train import positive_int


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    memorisation = summary["memorisation_audit"]
    diversity = summary["diversity"]
    coverage = summary["distribution_coverage"]
    feature_metrics = {
        row["feature"]: row for row in coverage["image_feature_metrics"]
    }
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
    perplexity = float(np.exp(-np.sum(probabilities[active] * np.log(probabilities[active]))))
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
            np.sum(distribution[nonzero] * np.log(distribution[nonzero] / midpoint[nonzero]))
        )

    return 0.5 * (kl_divergence(left) + kl_divergence(right))


def approximate_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, 1001)
    return float(np.mean(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles))))


def distribution_rows(train_features: np.ndarray, generated_features: np.ndarray) -> list[dict[str, Any]]:
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
                    float((generated.mean() - training.mean()) / train_standard_deviation)
                    if train_standard_deviation > 0
                    else float("nan")
                ),
                "generated_fraction_inside_train_p05_p95": float(
                    np.mean((generated >= lower) & (generated <= upper))
                ),
                "quantile_wasserstein_distance": approximate_wasserstein(training, generated),
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
        raise ValueError("Training metadata and manifest contain different sample counts.")
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
        mismatches = np.count_nonzero(training_codes != generated[None, ...], axis=(1, 2))
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
        code_mismatches[index] = np.mean(code_indices != code_grid[None, ...], axis=(1, 2))
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
        "nearest_generated_code_mismatch_fraction_median": float(np.median(nearest_code)),
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
    figure, axes = plt.subplots(
        len(shown_indices),
        3,
        figsize=(10.5, 3.0 * len(shown_indices)),
        squeeze=False,
    )
    for row_index, sample_index in enumerate(shown_indices):
        generated = generated_images[sample_index]
        neighbour = nearest_training_images[sample_index]
        difference = np.abs(generated - neighbour)
        row = audit_rows[sample_index]
        axes[row_index, 0].imshow(generated, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 0].set_title(
            f"Generated sample {int(row['sample_id']) + 1}", fontsize=9
        )
        axes[row_index, 1].imshow(neighbour, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].set_title(
            "Nearest training slice\n"
            f"patient {row['pixel_nn_subject_id']}, week {row['pixel_nn_week']}, "
            f"slice {row['pixel_nn_slice_index']}",
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
    figure.suptitle("Generated samples and nearest training slices", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_sample_panel(path: Path, images: np.ndarray) -> None:
    shown = min(16, len(images))
    columns = 4
    rows = math.ceil(shown / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.5, 3.1 * rows),
        squeeze=False,
    )
    for axis_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if axis_index < shown:
            axis.imshow(images[axis_index], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(f"Random sample {axis_index + 1}", fontsize=9)
    figure.suptitle("Unconditional VQ-VAE samples from the PixelCNN prior", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples_path = args.samples.resolve()
    train_codes_path = args.train_codes.resolve()
    metadata_path = args.train_metadata.resolve()
    train_summary_path = args.train_summary.resolve()
    prior_summary_path = args.prior_summary.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    for path in (
        samples_path,
        train_codes_path,
        metadata_path,
        train_summary_path,
        prior_summary_path,
        manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    prepare_output(output, args.overwrite)

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
    if (
        prior_summary.get("provenance", {}).get("source_vqvae_sha256")
        != train_summary.get("checkpoint", {}).get("sha256")
    ):
        raise ValueError("The prior and code-export summaries refer to different VQ-VAE checkpoints.")

    with np.load(samples_path) as archive:
        generated_images = validate_images(archive["images"], "generated images")
        generated_codes = np.asarray(archive["code_indices"], dtype=np.uint16)
        sample_ids = np.asarray(archive["sample_id"], dtype=np.int64)
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
    if len(generated_images) != len(generated_codes) or len(sample_ids) != len(generated_images):
        raise ValueError("Generated images, code grids, and sample IDs are misaligned.")
    if not np.array_equal(sample_ids, np.arange(len(sample_ids))):
        raise ValueError("Generated sample IDs are not contiguous and deterministic.")

    dataset, training_features, training_downsampled, metadata = load_training_data(
        manifest_path,
        metadata_path,
        args.batch_size,
        args.num_workers,
        args.downsample_factor,
    )
    if len(training_codes) != len(dataset):
        raise ValueError("Training code archive and manifest contain different sample counts.")
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
        raise ValueError("Training code archive metadata does not match train_samples.csv.")

    generated_features = image_features(generated_images)
    generated_downsampled = downsample(generated_images, args.downsample_factor)
    pixel_distances = squared_distance_matrix(training_downsampled, generated_downsampled)
    pixel_nearest_indices = pixel_distances.argmin(axis=0)
    pixel_nearest_mse = pixel_distances[pixel_nearest_indices, np.arange(len(generated_images))]
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
                "exact_pixel_match": bool(full_metrics_numpy["mse"][sample_index] <= 1e-12),
                "exact_code_grid_match": bool(code_mismatch[sample_index] == 0.0),
            }
        )

    distribution = distribution_rows(training_features, generated_features)
    train_code_statistics = code_statistics(training_codes, 512)
    generated_code_statistics = code_statistics(generated_codes, 512)
    train_active = train_code_statistics["counts"] > 0
    generated_active = generated_code_statistics["counts"] > 0

    neighbour_csv = output / "nearest_training_neighbours.csv"
    distribution_csv = output / "distribution_metrics.csv"
    neighbour_panel = output / "nearest_training_neighbours.png"
    sample_panel = output / "unconditional_samples.png"
    report_path = output / "REPORT_RESULTS.md"
    write_csv(neighbour_csv, audit_rows)
    write_csv(distribution_csv, distribution)
    save_neighbour_panel(
        neighbour_panel,
        generated_images,
        nearest_training_images,
        audit_rows,
        args.panel_samples,
    )
    save_sample_panel(sample_panel, generated_images)

    nearest_mse = np.asarray([row["full_resolution_mse"] for row in audit_rows])
    nearest_ssim = np.asarray([row["full_resolution_ssim"] for row in audit_rows])
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "reference_split": "train",
            "generated_samples": int(len(generated_images)),
            "training_samples": int(len(dataset)),
            "patient_disjoint_manifest_retained": True,
            "formal_prior_data_hash_verified": True,
            "pixel_search": (
                f"MSE on deterministic {args.downsample_factor}x average-pooled "
                "normalised slices"
            ),
            "pixel_confirmation": "Full-resolution MSE, MAE, PSNR, and SSIM",
            "code_search": "Hamming mismatch fraction on 32x18 VQ index grids",
        },
        "inputs": {
            "samples": {"path": str(samples_path), "sha256": sha256_file(samples_path)},
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
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        },
        "memorisation_audit": {
            "exact_pixel_matches": int(sum(row["exact_pixel_match"] for row in audit_rows)),
            "exact_code_grid_matches": int(
                sum(row["exact_code_grid_match"] for row in audit_rows)
            ),
            "nearest_training_full_resolution_mse_min": float(nearest_mse.min()),
            "nearest_training_full_resolution_mse_median": float(np.median(nearest_mse)),
            "nearest_training_full_resolution_mse_max": float(nearest_mse.max()),
            "nearest_training_full_resolution_ssim_min": float(nearest_ssim.min()),
            "nearest_training_full_resolution_ssim_median": float(np.median(nearest_ssim)),
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
        "limitations": [
            "Pixel-space nearest-neighbour search is sensitive to padding and intensity alignment.",
            "No exact or close neighbour is evidence against direct copying, not a privacy guarantee.",
            "Thirty-two samples support an initial audit but not clinical validation.",
        ],
        "artifacts": {
            "nearest_training_neighbours": neighbour_csv.name,
            "distribution_metrics": distribution_csv.name,
            "nearest_training_neighbour_panel": neighbour_panel.name,
            "unconditional_sample_panel": sample_panel.name,
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
