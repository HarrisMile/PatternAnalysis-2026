"""Training entry point for the HipMRI ConvVAE and VQ-VAE models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset
from modules import ConvVAE, VQVAE, vae_loss, vqvae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vae", "vqvae"), default="vae")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_name: str) -> torch.nn.Module:
    return ConvVAE() if model_name == "vae" else VQVAE()


def compute_loss(model_name: str, outputs: Dict[str, Tensor], images: Tensor) -> Dict[str, Tensor]:
    return vae_loss(outputs, images) if model_name == "vae" else vqvae_loss(outputs, images)


def run_smoke_test() -> None:
    images = torch.rand(2, 1, 64, 64)
    for model_name in ("vae", "vqvae"):
        model = build_model(model_name)
        outputs = model(images)
        losses = compute_loss(model_name, outputs, images)
        losses["loss"].backward()
        print(
            f"{model_name}: input={tuple(images.shape)}, "
            f"output={tuple(outputs['reconstruction'].shape)}, "
            f"loss={losses['loss'].item():.6f}"
        )


def train(args: argparse.Namespace) -> None:
    if args.manifest is None:
        raise ValueError("--manifest is required unless --smoke-test is used.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()
    dataset = HipMRISliceDataset(args.manifest, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = compute_loss(args.model, outputs, images)
            losses["loss"].backward()
            optimizer.step()
            running_loss += losses["loss"].item() * images.shape[0]
        print(f"epoch={epoch:03d} train_loss={running_loss / len(dataset):.6f}")

    checkpoint = {
        "model_name": args.model,
        "model_state": model.state_dict(),
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output / f"{args.model}.pt")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
