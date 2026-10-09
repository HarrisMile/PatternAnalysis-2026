"""Benchmark frozen HipMRI models on the validation split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset
from train import build_model, count_parameters, positive_int, select_device


EXPECTED_CHECKPOINTS = {
    "vae": "1da683f0d4afcf7704f236827380dde4f1999acfbff7f3256ddef75224ea63e1",
    "vqvae": "5f1aa0c74259f4a7d3614ef9fba302c34d1cfcb96ac67052c5deec52325cf8a9",
}
MODEL_LABELS = {"vae": "ConvVAE", "vqvae": "VQ-VAE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report_assets/inference_latency"),
    )
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=20)
    parser.add_argument("--max-batches", type=positive_int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def device_details(device: torch.device) -> dict[str, Any]:
    details: dict[str, Any] = {
        "type": device.type,
        "display_name": platform.processor() or platform.machine() or device.type,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        details.update(
            {
                "display_name": properties.name,
                "index": device.index if device.index is not None else torch.cuda.current_device(),
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    elif device.type == "mps":
        details["display_name"] = "Apple Metal Performance Shaders"
    return details


def validation_identity(dataset: HipMRISliceDataset, sample_count: int) -> dict[str, Any]:
    rows = dataset.rows[:sample_count]
    identifiers = [
        f"{row['subject_id']}|{row['week']}|{row['slice_index']}|{row['image_path']}"
        for row in rows
    ]
    digest = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
    return {
        "sample_identity_sha256": digest,
        "first_sample": identifiers[0],
        "last_sample": identifiers[-1],
    }


def preload_validation_batches(
    manifest: Path,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    device: torch.device,
) -> tuple[HipMRISliceDataset, list[torch.Tensor]]:
    dataset = HipMRISliceDataset(manifest, "validation")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    batches: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batches.append(
            batch["image"].to(device, non_blocking=device.type == "cuda").contiguous()
        )
    if not batches:
        raise RuntimeError("No validation batches were loaded.")
    synchronise(device)
    return dataset, batches


def benchmark_model(
    model_name: str,
    checkpoint_path: Path,
    batches: list[torch.Tensor],
    warmup_batches: int,
    repetitions: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_hash = sha256_file(checkpoint_path)
    expected_hash = EXPECTED_CHECKPOINTS[model_name]
    if checkpoint_hash != expected_hash:
        raise ValueError(
            f"Frozen {model_name} checkpoint hash mismatch: "
            f"expected {expected_hash}, received {checkpoint_hash}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != model_name:
        raise ValueError(
            f"Expected a {model_name} checkpoint, received {checkpoint.get('model_name')!r}."
        )
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for index in range(warmup_batches):
            model(batches[index % len(batches)])
        synchronise(device)

        sample_count = sum(int(batch.shape[0]) for batch in batches)
        repetition_ms_per_slice: list[float] = []
        repetition_total_ms: list[float] = []
        for _ in range(repetitions):
            synchronise(device)
            started = time.perf_counter_ns()
            for images in batches:
                model(images)
            synchronise(device)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            repetition_total_ms.append(elapsed_ms)
            repetition_ms_per_slice.append(elapsed_ms / sample_count)

    values = np.asarray(repetition_ms_per_slice, dtype=np.float64)
    q1, q3 = np.percentile(values, [25.0, 75.0])
    throughput_values = 1000.0 / values
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    result = {
        "model": MODEL_LABELS[model_name],
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "parameter_count": count_parameters(model),
        "samples_per_repetition": sample_count,
        "batches_per_repetition": len(batches),
        "repetitions": repetitions,
        "warmup_batches": warmup_batches,
        "repetition_total_ms": repetition_total_ms,
        "repetition_ms_per_slice": repetition_ms_per_slice,
        "median_ms_per_slice": float(np.median(values)),
        "q1_ms_per_slice": float(q1),
        "q3_ms_per_slice": float(q3),
        "iqr_ms_per_slice": float(q3 - q1),
        "min_ms_per_slice": float(values.min()),
        "max_ms_per_slice": float(values.max()),
        "median_slices_per_second": float(np.median(throughput_values)),
        "peak_cuda_memory_bytes": peak_memory_bytes,
    }
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_outputs(output: Path, payload: dict[str, Any], overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "summary.json"
    csv_path = output / "summary.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Benchmark output already exists: {existing}. Use --overwrite to replace it."
        )

    json_tmp = output / "summary.json.tmp"
    csv_tmp = output / "summary.csv.tmp"
    with json_tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = [
        "model",
        "model_name",
        "split",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "samples_per_repetition",
        "batches_per_repetition",
        "batch_size",
        "repetitions",
        "warmup_batches",
        "parameter_count",
        "device",
        "median_ms_per_slice",
        "q1_ms_per_slice",
        "q3_ms_per_slice",
        "iqr_ms_per_slice",
        "min_ms_per_slice",
        "max_ms_per_slice",
        "median_slices_per_second",
        "peak_cuda_memory_bytes",
    ]
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in payload["models"]:
            writer.writerow(
                {
                    **{name: result.get(name) for name in fieldnames},
                    "split": payload["protocol"]["split"],
                    "batch_size": payload["protocol"]["batch_size"],
                    "device": payload["environment"]["device"]["display_name"],
                }
            )
    json_tmp.replace(json_path)
    csv_tmp.replace(csv_path)


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    for path in (args.vae_checkpoint, args.vqvae_checkpoint, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = select_device(args.device)
    dataset, batches = preload_validation_batches(
        args.manifest,
        args.batch_size,
        args.num_workers,
        args.max_batches,
        device,
    )
    sample_count = sum(int(batch.shape[0]) for batch in batches)
    full_validation = args.max_batches is None and sample_count == len(dataset)
    if args.max_batches is None and not full_validation:
        raise RuntimeError(
            f"Expected all {len(dataset)} validation samples, loaded {sample_count}."
        )

    results = [
        benchmark_model(
            "vae",
            args.vae_checkpoint,
            batches,
            args.warmup_batches,
            args.repetitions,
            device,
        ),
        benchmark_model(
            "vqvae",
            args.vqvae_checkpoint,
            batches,
            args.warmup_batches,
            args.repetitions,
            device,
        ),
    ]
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "protocol": {
            "split": "validation",
            "full_validation_split": full_validation,
            "batch_size": args.batch_size,
            "num_workers_for_preload": args.num_workers,
            "warmup_batches": args.warmup_batches,
            "repetitions": args.repetitions,
            "samples_per_repetition": sample_count,
            "batches_per_repetition": len(batches),
            "timed_scope": "model forward pass on tensors preloaded to the selected device",
            "excluded_from_timing": [
                "disk input/output",
                "NIfTI loading",
                "normalisation and padding",
                "host-to-device transfer",
            ],
            **validation_identity(dataset, sample_count),
        },
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "device": device_details(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "models": results,
    }
    write_outputs(args.output, payload, args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved latency evidence to {args.output.resolve()}")


if __name__ == "__main__":
    main()
