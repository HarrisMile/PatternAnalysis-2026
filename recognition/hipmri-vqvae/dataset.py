"""Leakage-aware dataset interface for preprocessed HipMRI 2D slices."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"path", "subject_id", "split"}
VALID_SPLITS = {"train", "validation", "test"}


def read_manifest(manifest_path: str | Path) -> List[Dict[str, str]]:
    """Read and validate a CSV manifest without modifying it."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Manifest contains no samples.")
    invalid_splits = {row["split"] for row in rows} - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Unknown manifest splits: {sorted(invalid_splits)}")
    assert_subject_disjoint(rows)
    return rows


def assert_subject_disjoint(rows: List[Dict[str, str]]) -> None:
    """Raise when a subject occurs in more than one data split."""
    subjects_by_split = {
        split: {row["subject_id"] for row in rows if row["split"] == split}
        for split in VALID_SPLITS
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {
        f"{left}/{right}": sorted(subjects_by_split[left] & subjects_by_split[right])
        for left, right in pairs
        if subjects_by_split[left] & subjects_by_split[right]
    }
    if overlaps:
        raise ValueError(f"Subject leakage detected: {overlaps}")


class HipMRISliceDataset(Dataset[Dict[str, Tensor | str]]):
    """Load normalised 2D `.npy` slices listed in a fixed split manifest.

    Raw-data preprocessing will be added only after the downloaded HipMRI file
    structure and intensity conventions have been inspected.
    """

    def __init__(self, manifest_path: str | Path, split: str) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        self.manifest_path = Path(manifest_path)
        self.rows = [row for row in read_manifest(self.manifest_path) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Manifest has no samples for split '{split}'.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        row = self.rows[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Stage-one loader expects a .npy slice, received: {path}")

        image = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if image.ndim == 2:
            image = image[None, ...]
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(f"Expected [H, W] or [1, H, W], received {image.shape} from {path}")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite image values found in {path}")

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "subject_id": row["subject_id"],
            "path": str(path),
        }
