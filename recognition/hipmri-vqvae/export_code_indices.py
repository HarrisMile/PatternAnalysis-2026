"""Export train-only VQ-VAE code-index grids for learned-prior training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset, read_manifest
from modules import VQVAE
from train import describe_device, positive_int, select_device


SOURCE_SPLIT = "train"
NUM_EMBEDDINGS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/prior-data"))
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--max-batches",
        type=positive_int,
        help="Limit batches for a pipeline check; never use a limited export for prior training.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_worktree_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if not overwrite and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Prior-data artifacts already exist in {output}. "
            "Use a new directory or pass --overwrite explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def codebook_statistics(code_counts: np.ndarray) -> dict[str, Any]:
    total = int(code_counts.sum())
    if total <= 0:
        raise ValueError("The export contains no code assignments.")
    probabilities = code_counts.astype(np.float64) / total
    nonzero = probabilities > 0
    perplexity = float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))))
    active_codes = int(nonzero.sum())
    return {
        "total_assignments": total,
        "active_codes": active_codes,
        "dead_codes": int(code_counts.size - active_codes),
        "active_code_fraction": active_codes / int(code_counts.size),
        "perplexity": perplexity,
    }


def export_indices(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    prepare_output(output, args.overwrite)

    manifest_rows = read_manifest(manifest_path)
    rows_by_split = {
        split: [row for row in manifest_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    subjects_by_split = {
        split: sorted({row["subject_id"] for row in rows})
        for split, rows in rows_by_split.items()
    }
    train_subjects = set(subjects_by_split[SOURCE_SPLIT])
    held_out_subjects = set(subjects_by_split["validation"]) | set(subjects_by_split["test"])
    overlap = sorted(train_subjects & held_out_subjects)
    if overlap:
        raise ValueError(f"Train/held-out patient leakage detected: {overlap}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "vqvae":
        raise ValueError(
            f"Expected a VQ-VAE checkpoint, received {checkpoint.get('model_name')!r}."
        )
    model = VQVAE()
    model.load_state_dict(checkpoint["model_state"])
    if not bool(model.quantizer.codebook_initialized.item()):
        raise ValueError("The checkpoint codebook is not marked as initialised.")

    device = select_device(args.device)
    model = model.to(device).eval()
    dataset = HipMRISliceDataset(manifest_path, SOURCE_SPLIT)
    if len(dataset) != len(rows_by_split[SOURCE_SPLIT]):
        raise RuntimeError("Training dataset length does not match the validated manifest.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    index_batches: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []
    expected_grid_shape: tuple[int, int] | None = None
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            encoded = model.pre_quantizer(model.encoder(images))
            indices = model.quantizer(encoded)["encoding_indices"]
            grid_shape = tuple(int(value) for value in indices.shape[1:])
            if expected_grid_shape is None:
                expected_grid_shape = grid_shape
            elif grid_shape != expected_grid_shape:
                raise RuntimeError(
                    f"Inconsistent code-grid shapes: {expected_grid_shape} and {grid_shape}."
                )
            indices_numpy = indices.cpu().numpy()
            if indices_numpy.min() < 0 or indices_numpy.max() >= NUM_EMBEDDINGS:
                raise RuntimeError("An exported code index is outside the VQ codebook range.")
            index_batches.append(indices_numpy.astype(np.uint16, copy=False))

            batch_size = int(indices.shape[0])
            for item_index in range(batch_size):
                sample_rows.append(
                    {
                        "sample_index": len(sample_rows),
                        "subject_id": str(batch["subject_id"][item_index]),
                        "week": int(batch["week"][item_index]),
                        "slice_index": int(batch["slice_index"][item_index]),
                    }
                )
            if (batch_index + 1) % 20 == 0 or len(sample_rows) == len(dataset):
                print(
                    f"exported {len(sample_rows)}/{len(dataset)} training slices",
                    flush=True,
                )

    if not index_batches or expected_grid_shape is None:
        raise RuntimeError("No training code-index grids were exported.")
    code_indices = np.concatenate(index_batches, axis=0)
    if code_indices.shape[0] != len(sample_rows):
        raise RuntimeError("Code-index and sample-metadata counts do not match.")
    if args.max_batches is None and code_indices.shape[0] != len(dataset):
        raise RuntimeError("A full export did not include every training sample.")

    code_counts = np.bincount(
        code_indices.reshape(-1).astype(np.int64),
        minlength=NUM_EMBEDDINGS,
    )
    if code_counts.size != NUM_EMBEDDINGS:
        raise RuntimeError("Unexpected VQ codebook size in exported assignments.")
    statistics = codebook_statistics(code_counts)

    subjects = sorted({row["subject_id"] for row in sample_rows})
    if not set(subjects).issubset(train_subjects):
        raise RuntimeError("The export contains a non-training patient.")

    npz_path = output / "train_code_indices.npz"
    samples_path = output / "train_samples.csv"
    counts_path = output / "code_counts.csv"
    summary_path = output / "summary.json"
    write_npz(
        npz_path,
        indices=code_indices,
        subject_id=np.asarray([row["subject_id"] for row in sample_rows]),
        week=np.asarray([row["week"] for row in sample_rows], dtype=np.int16),
        slice_index=np.asarray([row["slice_index"] for row in sample_rows], dtype=np.int16),
    )
    write_csv(
        samples_path,
        ["sample_index", "subject_id", "week", "slice_index"],
        sample_rows,
    )
    write_csv(
        counts_path,
        ["code_index", "count", "fraction"],
        [
            {
                "code_index": code_index,
                "count": int(count),
                "fraction": float(count / statistics["total_assignments"]),
            }
            for code_index, count in enumerate(code_counts)
        ],
    )

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": SOURCE_SPLIT,
        "limited_export": args.max_batches is not None,
        "max_batches": args.max_batches,
        "sample_count": int(code_indices.shape[0]),
        "manifest_train_sample_count": len(dataset),
        "subject_count": len(subjects),
        "subject_ids": subjects,
        "held_out_subject_ids": {
            "validation": subjects_by_split["validation"],
            "test": subjects_by_split["test"],
        },
        "code_grid_shape": list(expected_grid_shape),
        "index_dtype": str(code_indices.dtype),
        "num_embeddings": NUM_EMBEDDINGS,
        "embedding_dim": int(model.quantizer.embedding_dim),
        "codebook_statistics": statistics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "seed": int(checkpoint["seed"]),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "name": describe_device(device),
        },
        "source_commit": git_commit(),
        "source_worktree_dirty": git_worktree_dirty(),
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": {
            "indices": npz_path.name,
            "samples": samples_path.name,
            "code_counts": counts_path.name,
        },
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = export_indices(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
