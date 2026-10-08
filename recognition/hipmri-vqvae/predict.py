"""Reconstruct HipMRI test slices from a saved ConvVAE or VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset
from train import build_model, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_name = checkpoint["model_name"]
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = HipMRISliceDataset(args.manifest, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.output.mkdir(parents=True, exist_ok=True)
    sample_index = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            reconstructions = model(images)["reconstruction"].cpu().numpy()
            originals = images.cpu().numpy()
            for original, reconstruction in zip(originals, reconstructions):
                np.savez_compressed(
                    args.output / f"sample_{sample_index:05d}.npz",
                    original=original,
                    reconstruction=reconstruction,
                )
                sample_index += 1

    print(f"Saved {sample_index} reconstructions to {args.output}")


if __name__ == "__main__":
    main()
