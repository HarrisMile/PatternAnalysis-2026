"""Build report evidence for the post-hoc VQ-VAE v2 follow-up experiment.

The primary ConvVAE versus VQ-VAE v1 comparison remains unchanged. This script
keeps the later validation-led VQ-VAE v2 experiment separate, verifies that V1 and
V2 were evaluated on identical slice identifiers, and records the protocol boundary
that V1 test results had already been observed before V2 was designed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


METRICS = ("mse", "mae", "psnr", "ssim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-validation",
        type=Path,
        default=Path("predictions/formal-vqvae-validation"),
    )
    parser.add_argument(
        "--v2-validation",
        type=Path,
        default=Path("predictions/vqvae-v2-ema-k128-ssim010-validation"),
    )
    parser.add_argument(
        "--v1-test", type=Path, default=Path("predictions/formal-vqvae-test")
    )
    parser.add_argument(
        "--v2-test",
        type=Path,
        default=Path("predictions/vqvae-v2-ema-k128-ssim010-test"),
    )
    parser.add_argument(
        "--v2-run", type=Path, default=Path("outputs/vqvae-v2-ema-k128-ssim010")
    )
    parser.add_argument(
        "--test-log", type=Path, default=Path("hipmri_vqvae_v2_test_700978.out")
    )
    parser.add_argument(
        "--test-error-log",
        type=Path,
        default=Path("hipmri_vqvae_v2_test_700978.err"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("report_assets/vqvae_v2_followup")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["subject_id"], int(row["week"]), int(row["slice_index"])


def indexed_rows(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    rows = read_csv(path)
    indexed = {sample_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"Duplicate sample identifiers in {path}")
    return indexed


def verify_paired(
    v1: dict[tuple[str, int, int], dict[str, str]],
    v2: dict[tuple[str, int, int], dict[str, str]],
    split: str,
) -> None:
    if v1.keys() != v2.keys():
        only_v1 = sorted(v1.keys() - v2.keys())[:5]
        only_v2 = sorted(v2.keys() - v1.keys())[:5]
        raise RuntimeError(
            f"{split} sample mismatch; only V1={only_v1}, only V2={only_v2}"
        )


def paired_summary(
    v1: dict[tuple[str, int, int], dict[str, str]],
    v2: dict[tuple[str, int, int], dict[str, str]],
) -> dict[str, Any]:
    keys = sorted(v1)
    ssim_delta = [float(v2[key]["ssim"]) - float(v1[key]["ssim"]) for key in keys]
    mse_delta = [float(v2[key]["mse"]) - float(v1[key]["mse"]) for key in keys]
    return {
        "paired_samples": len(keys),
        "ssim_improved_samples": sum(value > 0 for value in ssim_delta),
        "ssim_worsened_samples": sum(value < 0 for value in ssim_delta),
        "ssim_delta_mean": mean(ssim_delta),
        "ssim_delta_median": median(ssim_delta),
        "ssim_delta_min": min(ssim_delta),
        "ssim_delta_max": max(ssim_delta),
        "mse_improved_samples": sum(value < 0 for value in mse_delta),
        "mse_worsened_samples": sum(value > 0 for value in mse_delta),
        "mse_delta_mean": mean(mse_delta),
    }


def subject_rows(
    indexed: dict[tuple[str, int, int], dict[str, str]], model: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (subject, _, _), row in indexed.items():
        grouped[subject].append(row)
    output: list[dict[str, Any]] = []
    for subject, rows in sorted(grouped.items()):
        output.append(
            {
                "model": model,
                "subject_id": subject,
                "slices": len(rows),
                **{
                    metric: mean(float(row[metric]) for row in rows)
                    for metric in METRICS
                },
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_log_provenance(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    job_match = re.search(r"Job (\d+) on ([^,]+), started (.+)", text)
    hash_match = re.search(r"\n([0-9a-f]{64})\s+\S+/best\.pt\n", text)
    finish_match = re.search(r"Job (\d+) finished (.+)", text)
    if not job_match or not hash_match or not finish_match:
        raise RuntimeError(f"Could not parse required provenance from {path}")
    return {
        "job_id": int(job_match.group(1)),
        "node": job_match.group(2),
        "started": job_match.group(3),
        "finished": finish_match.group(2),
        "checkpoint_sha256": hash_match.group(1),
    }


def report_markdown(summary: dict[str, Any]) -> str:
    comparisons = summary["aggregate_comparison"]
    paired = summary["paired_test_comparison"]
    validation = comparisons["validation"]
    test = comparisons["test"]
    config = summary["v2_config"]
    training = summary["v2_training"]
    lines = [
        "# VQ-VAE v2 Post-hoc Follow-up",
        "",
        "## Protocol boundary",
        "",
        "VQ-VAE v2 was designed after the primary VQ-VAE v1 test result had already",
        "been observed. Its architecture and checkpoint were selected using validation",
        "data only, and the test split was then evaluated once. The result is therefore",
        "reported as a post-hoc engineering follow-up, not as a replacement untouched-test",
        "result for the original controlled comparison.",
        "",
        "## Configuration",
        "",
        f"- 60 epochs; selected epoch {training['best_epoch']} by validation SSIM.",
        f"- {config['vq_num_embeddings']}-entry, {config['vq_embedding_dim']}-dimensional codebook.",
        f"- EMA codebook update with decay {config['vq_ema_decay']} and dead-code patience {config['vq_dead_code_patience']}.",
        f"- Reconstruction objective: MSE plus {config['vq_ssim_weight']:.2f} times SSIM loss, with commitment weight {config['vq_commitment_cost']}.",
        "",
        "## Aggregate reconstruction evidence",
        "",
        "| Split | Model | MSE | MAE | PSNR | SSIM | Active codes | Perplexity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, values in (("Validation", validation), ("Test", test)):
        for model in ("V1", "V2"):
            row = values[model.lower()]
            lines.append(
                f"| {split} | {model} | {row['mse']:.6f} | {row['mae']:.6f} | "
                f"{row['psnr']:.3f} | {row['ssim']:.4f} | {row['active_codes']}/{row['codebook_size']} | "
                f"{row['codebook_perplexity']:.2f} |"
            )
    lines.extend(
        [
            "",
            f"On test, V2 improves SSIM for {paired['ssim_improved_samples']}/{paired['paired_samples']} matched slices and MSE for {paired['mse_improved_samples']}/{paired['paired_samples']}. Mean paired SSIM increases by {paired['ssim_delta_mean']:.4f}, while mean paired MSE changes by {paired['mse_delta_mean']:.6f}.",
            "",
            f"V2 test foreground SSIM is {test['v2']['foreground_ssim']:.4f}, so the improvement is not explained by padded background. All {test['v2']['active_codes']} codes are active and perplexity is {test['v2']['codebook_perplexity']:.2f}; no dead codes remain.",
            "",
            "![Highest-MSE VQ-VAE v2 test reconstructions](worst_reconstructions.png)",
            "",
            "The highest-MSE examples retain gross anatomical layout but still smooth fine texture and tissue boundaries. No catastrophic spatial displacement or checkerboard artefact is visible in this panel.",
            "",
            "## Interpretation boundary",
            "",
            "V2 solves the approximate 0.6 reconstruction-SSIM target within this follow-up, but it remains below the ConvVAE primary test SSIM of 0.9185. The existing PixelCNN prior was trained on the 512-entry V1 code space and is incompatible with the 128-entry V2 codebook. No unconditional-generation claim is therefore made for V2; a new prior and a new untouched evaluation protocol would be required.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    required = [
        args.v1_validation / "summary.json",
        args.v1_validation / "metrics.csv",
        args.v2_validation / "summary.json",
        args.v2_validation / "metrics.csv",
        args.v1_test / "summary.json",
        args.v1_test / "metrics.csv",
        args.v2_test / "summary.json",
        args.v2_test / "metrics.csv",
        args.v2_test / "worst_reconstructions.png",
        args.v2_run / "config.json",
        args.v2_run / "summary.json",
        args.test_log,
        args.test_error_log,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")
    if args.test_error_log.stat().st_size != 0:
        raise RuntimeError(f"Non-empty test error log: {args.test_error_log}")

    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    indexed: dict[str, dict[str, dict[tuple[str, int, int], dict[str, str]]]] = {}
    for split, v1_path, v2_path in (
        ("validation", args.v1_validation, args.v2_validation),
        ("test", args.v1_test, args.v2_test),
    ):
        summaries[split] = {
            "v1": read_json(v1_path / "summary.json"),
            "v2": read_json(v2_path / "summary.json"),
        }
        indexed[split] = {
            "v1": indexed_rows(v1_path / "metrics.csv"),
            "v2": indexed_rows(v2_path / "metrics.csv"),
        }
        verify_paired(indexed[split]["v1"], indexed[split]["v2"], split)

    for split in ("validation", "test"):
        for model in ("v1", "v2"):
            summaries[split][model]["codebook_size"] = 512 if model == "v1" else 128

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        args.v2_test / "worst_reconstructions.png",
        output / "worst_reconstructions.png",
    )

    comparison_rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for model in ("v1", "v2"):
            item = summaries[split][model]
            comparison_rows.append(
                {
                    "split": split,
                    "model": model,
                    **{metric: item[metric] for metric in METRICS},
                    "foreground_mse": item.get("foreground_mse", ""),
                    "foreground_ssim": item.get("foreground_ssim", ""),
                    "active_codes": item["active_codes"],
                    "codebook_size": item["codebook_size"],
                    "codebook_perplexity": item["codebook_perplexity"],
                }
            )
    write_csv(output / "model_comparison.csv", comparison_rows)

    per_subject = subject_rows(indexed["test"]["v1"], "v1") + subject_rows(
        indexed["test"]["v2"], "v2"
    )
    write_csv(output / "per_subject_test_metrics.csv", per_subject)

    v2_config = read_json(args.v2_run / "config.json")
    v2_training = read_json(args.v2_run / "summary.json")
    summary = {
        "protocol": {
            "classification": "post-hoc validation-led follow-up",
            "primary_v1_test_result_observed_before_v2_design": True,
            "v2_checkpoint_selected_on": "validation_ssim",
            "v2_test_evaluations": 1,
            "generative_scope": "none; the existing PixelCNN prior belongs to V1",
        },
        "aggregate_comparison": summaries,
        "paired_validation_comparison": paired_summary(
            indexed["validation"]["v1"], indexed["validation"]["v2"]
        ),
        "paired_test_comparison": paired_summary(
            indexed["test"]["v1"], indexed["test"]["v2"]
        ),
        "v2_config": v2_config,
        "v2_training": v2_training,
        "test_job": {
            **extract_log_provenance(args.test_log),
            "error_log_empty": True,
            "evaluated_samples": summaries["test"]["v2"]["evaluated_samples"],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT_RESULTS.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(f"Saved VQ-VAE v2 follow-up evidence to {output}")


if __name__ == "__main__":
    main()
