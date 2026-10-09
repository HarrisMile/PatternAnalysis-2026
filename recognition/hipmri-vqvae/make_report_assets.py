"""Build reproducible report tables and figures from the frozen experiment artifacts.

This script never trains a model or recomputes the final aggregate metrics. It reads
the saved histories, summaries, and per-slice metric CSVs. The frozen checkpoints are
loaded only to render deterministic, matched qualitative examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import HipMRISliceDataset
from train import build_model, select_device


MODEL_LABELS = {"vae": "ConvVAE", "vqvae": "VQ-VAE"}
MODEL_COLOURS = {"vae": "#0072B2", "vqvae": "#D55E00"}
FROZEN_CHECKPOINT_HASHES = {
    "vae": "1da683f0d4afcf7704f236827380dde4f1999acfbff7f3256ddef75224ea63e1",
    "vqvae": "5f1aa0c74259f4a7d3614ef9fba302c34d1cfcb96ac67052c5deec52325cf8a9",
}
METRIC_COLUMNS = (
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
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--vae-run", type=Path, default=Path("outputs/formal-vae"))
    parser.add_argument("--vqvae-run", type=Path, default=Path("outputs/formal-vqvae"))
    parser.add_argument(
        "--vae-validation", type=Path, default=Path("predictions/formal-vae-validation")
    )
    parser.add_argument(
        "--vqvae-validation",
        type=Path,
        default=Path("predictions/formal-vqvae-validation"),
    )
    parser.add_argument("--vae-test", type=Path, default=Path("predictions/formal-vae-test"))
    parser.add_argument(
        "--vqvae-test", type=Path, default=Path("predictions/formal-vqvae-test")
    )
    parser.add_argument("--output", type=Path, default=Path("report_assets"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_checkpoints(paths: dict[str, Path]) -> dict[str, str]:
    observed = {model: sha256(path) for model, path in paths.items()}
    mismatches = {
        model: {"expected": FROZEN_CHECKPOINT_HASHES[model], "observed": digest}
        for model, digest in observed.items()
        if digest != FROZEN_CHECKPOINT_HASHES[model]
    }
    if mismatches:
        raise RuntimeError(f"Frozen checkpoint hash mismatch: {mismatches}")
    return observed


def float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_training_curves(
    histories: dict[str, list[dict[str, str]]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    panels = (
        ("train_reconstruction_loss", "Training reconstruction MSE", True),
        ("validation_mse", "Validation MSE", True),
        ("validation_psnr", "Validation PSNR (dB)", False),
        ("validation_ssim", "Validation SSIM", False),
    )
    for axis, (column, title, log_scale) in zip(axes.flat, panels):
        for model in ("vae", "vqvae"):
            rows = histories[model]
            axis.plot(
                float_column(rows, "epoch"),
                float_column(rows, column),
                label=MODEL_LABELS[model],
                color=MODEL_COLOURS[model],
                linewidth=2.2,
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Training and validation behaviour under the same 40-epoch budget")
    figure.tight_layout()
    save_figure(figure, output / "training_curves.png")


def plot_codebook_usage(history: list[dict[str, str]], output: Path) -> None:
    epochs = float_column(history, "epoch")
    active = float_column(history, "validation_active_codes")
    perplexity = float_column(history, "validation_codebook_perplexity")
    figure, (axis_active, axis_perplexity) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axis_active.plot(
        epochs,
        active,
        color="#009E73",
        linewidth=2.2,
        marker="o",
        markersize=3.3,
    )
    axis_active.set_xlabel("Epoch")
    axis_active.set_ylabel("Active entries")
    axis_active.set_ylim(0, max(20, float(active.max()) + 2))
    axis_active.set_title("Active validation codes")
    axis_active.grid(alpha=0.25)
    axis_active.annotate(
        f"Selected epoch 38: {int(active[37])}/512 active",
        (38, active[37]),
        xytext=(-145, 38),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "0.35"},
        fontsize=9,
    )
    axis_perplexity.plot(
        epochs,
        perplexity,
        color="#CC79A7",
        linewidth=2.2,
    )
    axis_perplexity.set_xlabel("Epoch")
    axis_perplexity.set_ylabel("Perplexity")
    axis_perplexity.set_ylim(bottom=0)
    axis_perplexity.set_title("Validation codebook perplexity")
    axis_perplexity.grid(alpha=0.25)
    figure.suptitle("VQ-VAE codebook utilisation remains sparse (512 entries available)")
    figure.tight_layout()
    save_figure(figure, output / "vq_codebook_usage.png")


def plot_test_metric_comparison(
    summaries: dict[str, dict[str, Any]], output: Path
) -> None:
    panels = (
        ("mse", "Whole-image MSE", True),
        ("psnr", "Whole-image PSNR (dB)", False),
        ("ssim", "Whole-image SSIM", False),
        ("foreground_mse", "Foreground MSE", True),
        ("foreground_psnr", "Foreground PSNR (dB)", False),
        ("foreground_ssim", "Foreground SSIM", False),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
    models = ("vae", "vqvae")
    for axis, (metric, title, log_scale) in zip(axes.flat, panels):
        values = [float(summaries[model][metric]) for model in models]
        bars = axis.bar(
            [MODEL_LABELS[model] for model in models],
            values,
            color=[MODEL_COLOURS[model] for model in models],
            width=0.62,
        )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            label = f"{value:.6f}" if "mse" in metric else f"{value:.3f}"
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    figure.suptitle("Frozen one-time test performance (540 slices, patients 040-042)")
    figure.tight_layout()
    save_figure(figure, output / "test_metric_comparison.png")


def aggregate_by_subject(
    rows_by_model: dict[str, list[dict[str, str]]]
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for model, rows in rows_by_model.items():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["subject_id"]].append(row)
        for subject, subject_rows in sorted(grouped.items()):
            output_rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "subject_id": subject,
                    "samples": len(subject_rows),
                    **{
                        metric: mean(float(row[metric]) for row in subject_rows)
                        for metric in METRIC_COLUMNS
                    },
                }
            )
    return output_rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_per_subject(rows: list[dict[str, Any]], output: Path) -> None:
    subjects = sorted({str(row["subject_id"]) for row in rows})
    row_index = {(row["model"], str(row["subject_id"])): row for row in rows}
    panels = (("mse", "MSE", True), ("psnr", "PSNR (dB)", False), ("ssim", "SSIM", False))
    x = np.arange(len(subjects))
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    for axis, (metric, title, log_scale) in zip(axes, panels):
        for offset, model in ((-width / 2, "vae"), (width / 2, "vqvae")):
            label = MODEL_LABELS[model]
            values = [float(row_index[(label, subject)][metric]) for subject in subjects]
            axis.bar(x + offset, values, width, label=label, color=MODEL_COLOURS[model])
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xticks(x, [f"Patient {subject}" for subject in subjects])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Patient-level test performance")
    figure.tight_layout()
    save_figure(figure, output / "per_subject_test_metrics.png")


def metric_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["subject_id"], int(row["week"]), int(row["slice_index"])


def choose_matched_examples(vq_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in vq_rows:
        grouped[row["subject_id"]].append(row)
    selected: list[dict[str, Any]] = []
    for subject, rows in sorted(grouped.items()):
        subject_median = median(float(row["mse"]) for row in rows)
        representative = min(
            rows,
            key=lambda row: (
                abs(float(row["mse"]) - subject_median),
                int(row["week"]),
                int(row["slice_index"]),
            ),
        )
        challenging = max(
            rows,
            key=lambda row: (float(row["mse"]), -int(row["week"]), -int(row["slice_index"])),
        )
        selected.extend(
            (
                {"selection": "Representative (subject median VQ-VAE MSE)", **representative},
                {"selection": "Challenging (subject maximum VQ-VAE MSE)", **challenging},
            )
        )
    return selected


def load_models(checkpoints: dict[str, Path], device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for model_name, path in checkpoints.items():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["model_name"] != model_name:
            raise RuntimeError(f"Unexpected model in {path}: {checkpoint['model_name']}")
        model = build_model(model_name).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[model_name] = model
    return models


def render_matched_reconstructions(
    selected: list[dict[str, Any]],
    manifest: Path,
    checkpoints: dict[str, Path],
    device: torch.device,
    output: Path,
) -> list[dict[str, Any]]:
    dataset = HipMRISliceDataset(manifest, "test")
    dataset_indices = {
        (row["subject_id"], int(row["week"]), int(row["slice_index"])): index
        for index, row in enumerate(dataset.rows)
    }
    models = load_models(checkpoints, device)
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for selected_row in selected:
            key = metric_key(selected_row)
            sample = dataset[dataset_indices[key]]
            image = sample["image"].unsqueeze(0).to(device)
            reconstructions = {
                model_name: model(image)["reconstruction"][0, 0].cpu().numpy()
                for model_name, model in models.items()
            }
            results.append(
                {
                    **selected_row,
                    "original": image[0, 0].cpu().numpy(),
                    "mask": sample["mask"].numpy(),
                    "vae_reconstruction": reconstructions["vae"],
                    "vqvae_reconstruction": reconstructions["vqvae"],
                }
            )

    figure, axes = plt.subplots(len(results), 6, figsize=(14.2, 2.7 * len(results)), squeeze=False)
    titles = ("Original", "ConvVAE", "VQ-VAE", "ConvVAE error", "VQ-VAE error", "Mask")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row_index, result in enumerate(results):
        original = result["original"]
        vae_reconstruction = result["vae_reconstruction"]
        vqvae_reconstruction = result["vqvae_reconstruction"]
        vae_error = np.abs(vae_reconstruction - original)
        vqvae_error = np.abs(vqvae_reconstruction - original)
        shared_error_limit = max(float(np.percentile(vqvae_error, 99.5)), 1e-6)
        panels = (
            (original, "gray", 0.0, 1.0),
            (vae_reconstruction, "gray", 0.0, 1.0),
            (vqvae_reconstruction, "gray", 0.0, 1.0),
            (vae_error, "magma", 0.0, shared_error_limit),
            (vqvae_error, "magma", 0.0, shared_error_limit),
            (result["mask"], "viridis", 0, 5),
        )
        for axis, (array, cmap, minimum, maximum) in zip(axes[row_index], panels):
            axis.imshow(array, cmap=cmap, vmin=minimum, vmax=maximum)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"{result['selection'].split()[0]}\npatient {result['subject_id']}, "
            f"week {result['week']}, slice {result['slice_index']}",
            fontsize=8.5,
        )
    figure.suptitle("Matched frozen-model reconstructions on identical test slices", y=1.002)
    figure.tight_layout()
    save_figure(figure, output / "matched_test_reconstructions.png")
    return results


def build_model_comparison(
    run_summaries: dict[str, dict[str, Any]],
    validation_summaries: dict[str, dict[str, Any]],
    test_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("vae", "vqvae"):
        run = run_summaries[model]
        for split, summary in (("validation", validation_summaries[model]), ("test", test_summaries[model])):
            row: dict[str, Any] = {
                "model": MODEL_LABELS[model],
                "split": split,
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "samples": summary["evaluated_samples"],
                "parameters": run["trainable_parameters"],
                "training_seconds": run["training_seconds"],
                "peak_accelerator_memory_mb": run["peak_accelerator_memory_mb"],
            }
            for metric in METRIC_COLUMNS:
                row[metric] = summary.get(metric)
            row["active_codes"] = summary.get("active_codes")
            row["codebook_perplexity"] = summary.get("codebook_perplexity")
            row["active_code_fraction"] = summary.get("active_code_fraction")
            rows.append(row)
    return rows


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_results_note(
    output: Path,
    test_summaries: dict[str, dict[str, Any]],
    run_summaries: dict[str, dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    vae = test_summaries["vae"]
    vq = test_summaries["vqvae"]
    mse_ratio = float(vq["mse"]) / float(vae["mse"])
    psnr_difference = float(vae["psnr"]) - float(vq["psnr"])
    ssim_difference = float(vae["ssim"]) - float(vq["ssim"])
    comparison = format_markdown_table(
        ["Model", "MSE", "MAE", "PSNR", "SSIM", "Foreground MSE", "Foreground PSNR", "Foreground SSIM"],
        [
            [
                MODEL_LABELS[model],
                f"{float(test_summaries[model]['mse']):.6f}",
                f"{float(test_summaries[model]['mae']):.6f}",
                f"{float(test_summaries[model]['psnr']):.3f}",
                f"{float(test_summaries[model]['ssim']):.4f}",
                f"{float(test_summaries[model]['foreground_mse']):.6f}",
                f"{float(test_summaries[model]['foreground_psnr']):.3f}",
                f"{float(test_summaries[model]['foreground_ssim']):.4f}",
            ]
            for model in ("vae", "vqvae")
        ],
    )
    resource_table = format_markdown_table(
        ["Model", "Parameters", "Training time (s)", "Peak GPU memory (MiB)", "Selected epoch"],
        [
            [
                MODEL_LABELS[model],
                f"{int(run_summaries[model]['trainable_parameters']):,}",
                f"{float(run_summaries[model]['training_seconds']):.1f}",
                f"{float(run_summaries[model]['peak_accelerator_memory_mb']):.1f}",
                str(int(test_summaries[model]["checkpoint_epoch"])),
            ]
            for model in ("vae", "vqvae")
        ],
    )
    patient_table = format_markdown_table(
        ["Model", "Patient", "Slices", "MSE", "PSNR", "SSIM", "Foreground MSE", "Foreground SSIM"],
        [
            [
                str(row["model"]),
                str(row["subject_id"]),
                str(row["samples"]),
                f"{float(row['mse']):.6f}",
                f"{float(row['psnr']):.3f}",
                f"{float(row['ssim']):.4f}",
                f"{float(row['foreground_mse']):.6f}",
                f"{float(row['foreground_ssim']):.4f}",
            ]
            for row in subject_rows
        ],
    )
    selection_table = format_markdown_table(
        ["Selection", "Patient", "Week", "Slice", "VQ-VAE MSE"],
        [
            [
                str(row["selection"]),
                str(row["subject_id"]),
                str(row["week"]),
                str(row["slice_index"]),
                f"{float(row['mse']):.6f}",
            ]
            for row in selected
        ],
    )
    note = f"""# Frozen experiment results for the final report

This note is generated from the saved formal-run artifacts. It does not contain new
training, checkpoint selection, or test-set tuning.

## Core test comparison

{comparison}

The ConvVAE achieved a {mse_ratio:.2f}-fold lower MSE than the VQ-VAE, an
{psnr_difference:.2f} dB higher PSNR, and a {ssim_difference:.3f} higher SSIM. The
foreground-only comparison reaches the same conclusion, so the advantage is not an
artefact of padded background pixels.

## Resource and selection evidence

{resource_table}

The two models used nearly identical training time and peak GPU memory. The VQ-VAE
therefore did not obtain a reconstruction-quality benefit in exchange for its slightly
higher parameter count. Its discrete representation is compact at the index level, but
only {int(vq['active_codes'])}/512 codebook entries were active on test
({100 * float(vq['active_code_fraction']):.2f}%), with perplexity
{float(vq['codebook_perplexity']):.2f}.

## Patient-level robustness

{patient_table}

The ConvVAE is better for every held-out patient, so the aggregate conclusion is not
driven by one patient. Patient 040 contributes more slices than patients 041 and 042;
the patient-level table is therefore an important robustness check alongside the
slice-weighted aggregate metrics.

## Qualitative selection protocol

{selection_table}

For each test patient, one representative slice is selected as the slice closest to that
patient's median VQ-VAE MSE, and one challenging slice is selected as that patient's
maximum VQ-VAE MSE. Both frozen models are shown on exactly the same slices, with a
shared error-map scale within each row. This avoids comparing different model-specific
worst cases.

## Interpretation and recommendation

The ConvVAE preserves anatomical boundaries and fine texture substantially better. The
VQ-VAE reconstructions are visibly blurred and their errors are concentrated around
tissue interfaces and fine structures. The VQ-VAE avoided total single-code collapse,
but it still learned a heavily under-utilised codebook. Under the controlled 40-epoch
budget, the ConvVAE is the recommended model for HipMRI reconstruction quality and
stability. The VQ-VAE result should be reported as a valid negative result and a clear
limitation, not tuned further after test evaluation.
"""
    (output / "REPORT_RESULTS.md").write_text(note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "vae": args.vae_run / "best.pt",
        "vqvae": args.vqvae_run / "best.pt",
    }
    checkpoint_hashes = verify_frozen_checkpoints(checkpoints)
    histories = {
        "vae": read_csv(args.vae_run / "history.csv"),
        "vqvae": read_csv(args.vqvae_run / "history.csv"),
    }
    run_summaries = {
        "vae": read_json(args.vae_run / "summary.json"),
        "vqvae": read_json(args.vqvae_run / "summary.json"),
    }
    validation_summaries = {
        "vae": read_json(args.vae_validation / "summary.json"),
        "vqvae": read_json(args.vqvae_validation / "summary.json"),
    }
    test_summaries = {
        "vae": read_json(args.vae_test / "summary.json"),
        "vqvae": read_json(args.vqvae_test / "summary.json"),
    }
    test_rows = {
        "vae": read_csv(args.vae_test / "metrics.csv"),
        "vqvae": read_csv(args.vqvae_test / "metrics.csv"),
    }
    if len(test_rows["vae"]) != 540 or len(test_rows["vqvae"]) != 540:
        raise RuntimeError("Expected 540 complete per-slice test rows for each model.")
    if {metric_key(row) for row in test_rows["vae"]} != {
        metric_key(row) for row in test_rows["vqvae"]
    }:
        raise RuntimeError("VAE and VQ-VAE test metrics do not contain identical samples.")

    plot_training_curves(histories, args.output)
    plot_codebook_usage(histories["vqvae"], args.output)
    plot_test_metric_comparison(test_summaries, args.output)

    subject_rows = aggregate_by_subject(test_rows)
    write_csv(
        args.output / "per_subject_test_metrics.csv",
        subject_rows,
        ["model", "subject_id", "samples", *METRIC_COLUMNS],
    )
    plot_per_subject(subject_rows, args.output)

    comparison_rows = build_model_comparison(
        run_summaries, validation_summaries, test_summaries
    )
    comparison_fields = list(comparison_rows[0].keys())
    write_csv(args.output / "model_comparison.csv", comparison_rows, comparison_fields)

    selected = choose_matched_examples(test_rows["vqvae"])
    device = select_device(args.device)
    rendered = render_matched_reconstructions(
        selected, args.manifest, checkpoints, device, args.output
    )
    write_csv(
        args.output / "matched_test_selection.csv",
        [
            {
                "selection": row["selection"],
                "subject_id": row["subject_id"],
                "week": row["week"],
                "slice_index": row["slice_index"],
                "vqvae_mse": row["mse"],
            }
            for row in rendered
        ],
        ["selection", "subject_id", "week", "slice_index", "vqvae_mse"],
    )
    write_results_note(
        args.output, test_summaries, run_summaries, subject_rows, selected
    )

    provenance = {
        "checkpoint_sha256": checkpoint_hashes,
        "device_used_for_qualitative_rendering": str(device),
        "final_test_job": "644134",
        "test_samples": len(test_rows["vae"]),
        "qualitative_samples": len(selected),
        "qualitative_selection_rule": (
            "For each test patient, choose the VQ-VAE median-MSE-nearest slice and "
            "maximum-MSE slice; render both frozen models on those identical inputs."
        ),
    }
    with (args.output / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Verified frozen checkpoint hashes and wrote report assets to {args.output}")


if __name__ == "__main__":
    main()
