"""Leakage-safe HipMRI discovery, manifest generation, and NIfTI loading."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


VALID_SPLITS = ("train", "validation", "test")
DIRECTORY_SUFFIX = {"train": "train", "validation": "validate", "test": "test"}
REQUIRED_COLUMNS = {
    "image_path",
    "mask_path",
    "subject_id",
    "week",
    "slice_index",
    "split",
}
FILENAME_PATTERN = re.compile(
    r"^(?:case|seg)_(?P<subject>\d+)_week_(?P<week>\d+)_slice_(?P<slice>\d+)\.nii\.gz$"
)
DEFAULT_TARGET_SHAPE = (256, 144)


@dataclass(frozen=True)
class SampleRecord:
    """One paired MRI slice and segmentation mask."""

    image_path: Path
    mask_path: Path
    subject_id: str
    week: int
    slice_index: int
    split: str


def _parse_filename(path: Path) -> tuple[str, int, int]:
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected HipMRI filename: {path.name}")
    return match["subject"], int(match["week"]), int(match["slice"])


def _indexed_paths(directory: Path) -> Dict[tuple[str, int, int], Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"HipMRI directory not found: {directory}")
    indexed: Dict[tuple[str, int, int], Path] = {}
    for path in sorted(directory.glob("*.nii.gz")):
        key = _parse_filename(path)
        if key in indexed:
            raise ValueError(f"Duplicate HipMRI key {key} in {directory}")
        indexed[key] = path
    if not indexed:
        raise ValueError(f"No .nii.gz files found in {directory}")
    return indexed


def discover_samples(data_root: str | Path) -> List[SampleRecord]:
    """Discover the official paired splits and verify their integrity."""
    data_root = Path(data_root).resolve()
    records: List[SampleRecord] = []
    for split in VALID_SPLITS:
        suffix = DIRECTORY_SUFFIX[split]
        images = _indexed_paths(data_root / f"keras_slices_{suffix}")
        masks = _indexed_paths(data_root / f"keras_slices_seg_{suffix}")
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks or missing_images:
            raise ValueError(
                f"Unpaired files in split '{split}': "
                f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
            )
        for subject_id, week, slice_index in sorted(images):
            key = (subject_id, week, slice_index)
            records.append(
                SampleRecord(
                    image_path=images[key],
                    mask_path=masks[key],
                    subject_id=subject_id,
                    week=week,
                    slice_index=slice_index,
                    split=split,
                )
            )
    assert_subject_disjoint(records)
    return records


def assert_subject_disjoint(rows: Sequence[SampleRecord | Dict[str, str]]) -> None:
    """Raise when a patient occurs in more than one split."""
    subjects_by_split = {
        split: {
            row.subject_id if isinstance(row, SampleRecord) else row["subject_id"]
            for row in rows
            if (row.split if isinstance(row, SampleRecord) else row["split"]) == split
        }
        for split in VALID_SPLITS
    }
    overlaps = {}
    for index, left in enumerate(VALID_SPLITS):
        for right in VALID_SPLITS[index + 1 :]:
            shared = sorted(subjects_by_split[left] & subjects_by_split[right])
            if shared:
                overlaps[f"{left}/{right}"] = shared
    if overlaps:
        raise ValueError(f"Subject leakage detected: {overlaps}")


def write_manifest(data_root: str | Path, output_path: str | Path) -> List[SampleRecord]:
    """Create a deterministic CSV manifest using paths relative to the CSV."""
    records = discover_samples(data_root)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": os.path.relpath(record.image_path, output_path.parent),
                    "mask_path": os.path.relpath(record.mask_path, output_path.parent),
                    "subject_id": record.subject_id,
                    "week": record.week,
                    "slice_index": record.slice_index,
                    "split": record.split,
                }
            )
    return records


def read_manifest(manifest_path: str | Path) -> List[Dict[str, str]]:
    """Read a fixed manifest and re-check patient-level split isolation."""
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
    invalid_splits = {row["split"] for row in rows} - set(VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unknown manifest splits: {sorted(invalid_splits)}")
    assert_subject_disjoint(rows)
    return rows


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_2d_nifti(path: Path) -> np.ndarray:
    image = np.asarray(nib.load(path).dataobj)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D NIfTI array, received {image.shape} from {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite values found in {path}")
    return image


def _pad_to_shape(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if array.shape[0] > target_shape[0] or array.shape[1] > target_shape[1]:
        raise ValueError(f"Cannot pad shape {array.shape} to smaller target {target_shape}")
    height_padding = target_shape[0] - array.shape[0]
    width_padding = target_shape[1] - array.shape[1]
    return np.pad(
        array,
        (
            (height_padding // 2, height_padding - height_padding // 2),
            (width_padding // 2, width_padding - width_padding // 2),
        ),
        mode="constant",
    )


def normalise_mri(image: np.ndarray, upper_percentile: float = 99.5) -> np.ndarray:
    """Robustly scale one non-negative MRI slice to [0, 1]."""
    image = image.astype(np.float32, copy=False)
    if image.min() < 0:
        raise ValueError("HipMRI normalisation expects non-negative intensities.")
    foreground = image[image > 0]
    if foreground.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    upper = float(np.percentile(foreground, upper_percentile))
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError(f"Invalid normalisation upper bound: {upper}")
    return np.clip(image, 0.0, upper) / upper


class HipMRISliceDataset(Dataset[Dict[str, Tensor | str | int]]):
    """Load paired HipMRI NIfTI slices from an immutable split manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        target_shape: tuple[int, int] = DEFAULT_TARGET_SHAPE,
        upper_percentile: float = 99.5,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {list(VALID_SPLITS)}")
        if not 0.0 < upper_percentile <= 100.0:
            raise ValueError("upper_percentile must be in (0, 100].")
        self.manifest_path = Path(manifest_path).resolve()
        self.rows = [row for row in read_manifest(self.manifest_path) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Manifest has no samples for split '{split}'.")
        self.target_shape = target_shape
        self.upper_percentile = upper_percentile

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        row = self.rows[index]
        image_path = _resolve_path(self.manifest_path, row["image_path"])
        mask_path = _resolve_path(self.manifest_path, row["mask_path"])
        image = _load_2d_nifti(image_path)
        mask = _load_2d_nifti(mask_path)
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {image_path} and {mask_path}")

        image = _pad_to_shape(normalise_mri(image, self.upper_percentile), self.target_shape)
        mask = _pad_to_shape(mask.astype(np.int64, copy=False), self.target_shape)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image[None, ...])),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "subject_id": row["subject_id"],
            "week": int(row["week"]),
            "slice_index": int(row["slice_index"]),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def _split_summary(records: Iterable[SampleRecord]) -> str:
    records = list(records)
    lines = []
    for split in VALID_SPLITS:
        selected = [record for record in records if record.split == split]
        subjects = sorted({record.subject_id for record in selected})
        lines.append(f"{split}: samples={len(selected)}, subjects={subjects}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = write_manifest(args.data_root, args.output)
    print(f"Wrote {len(records)} paired samples to {args.output}")
    print(_split_summary(records))


if __name__ == "__main__":
    main()
