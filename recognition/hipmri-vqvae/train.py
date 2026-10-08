"""Train and validate HipMRI ConvVAE or VQ-VAE reconstruction models."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset
from metrics import reconstruction_metrics
from modules import ConvVAE, VQVAE, vae_loss, vqvae_loss


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vae", "vqvae"), default="vae")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--vae-beta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-train-batches", type=positive_int)
    parser.add_argument("--max-validation-batches", type=positive_int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def select_device(requested: str = "auto") -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "CPU"


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "vae":
        return ConvVAE()
    if model_name == "vqvae":
        return VQVAE()
    raise ValueError(f"Unknown model: {model_name}")


def compute_loss(
    model_name: str,
    outputs: Dict[str, Tensor],
    images: Tensor,
    vae_beta: float = 1e-4,
) -> Dict[str, Tensor]:
    if model_name == "vae":
        return vae_loss(outputs, images, beta=vae_beta)
    return vqvae_loss(outputs, images)


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(
    dataset: HipMRISliceDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _codebook_statistics(code_counts: Tensor) -> Dict[str, float]:
    total = code_counts.sum().clamp_min(1.0)
    probabilities = code_counts / total
    nonzero = probabilities > 0
    perplexity = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    active_codes = int(nonzero.sum().item())
    num_codes = int(code_counts.numel())
    return {
        "codebook_perplexity": float(perplexity.item()),
        "active_codes": float(active_codes),
        "dead_codes": float(num_codes - active_codes),
        "active_code_fraction": active_codes / num_codes,
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    device: torch.device,
    vae_beta: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
    include_reconstruction_metrics: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "regularisation_loss": 0.0,
    }
    if include_reconstruction_metrics:
        sums.update({"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0})
    sample_count = 0
    batch_count = 0
    code_counts: Optional[Tensor] = None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            losses = compute_loss(model_name, outputs, images, vae_beta)
            if training:
                losses["loss"].backward()
                optimizer.step()

        batch_size = images.shape[0]
        sample_count += batch_size
        batch_count += 1
        for name in ("loss", "reconstruction_loss", "regularisation_loss"):
            sums[name] += float(losses[name].detach().item()) * batch_size

        if include_reconstruction_metrics:
            metrics = reconstruction_metrics(images, outputs["reconstruction"])
            for name, values in metrics.items():
                sums[name] += float(values.detach().sum().item())

        if model_name == "vqvae":
            indices = outputs["encoding_indices"].detach().reshape(-1)
            num_embeddings = int(model.quantizer.num_embeddings)
            counts = torch.bincount(indices, minlength=num_embeddings).to(torch.float64).cpu()
            code_counts = counts if code_counts is None else code_counts + counts

    if sample_count == 0:
        raise RuntimeError("No batches were processed; check the dataset and batch limits.")
    statistics = {name: total / sample_count for name, total in sums.items()}
    statistics["samples"] = float(sample_count)
    statistics["batches"] = float(batch_count)
    if code_counts is not None:
        statistics.update(_codebook_statistics(code_counts))
    return statistics


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_history(path: Path, rows: list[Dict[str, float]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_history(path: Path) -> list[Dict[str, float]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.vae_beta < 0:
        raise ValueError("--vae-beta cannot be negative.")
    if args.resume is not None and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    if args.resume is not None and args.output.resolve() != args.resume.parent.resolve():
        raise ValueError("--output must be the resume checkpoint's directory.")
    if (
        args.resume is None
        and args.output.exists()
        and any(args.output.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output directory is not empty: {args.output}. Use --overwrite to reuse it."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    configure_reproducibility(args.seed)
    device = select_device(args.device)
    pin_memory = device.type == "cuda"
    train_dataset = HipMRISliceDataset(args.manifest, "train")
    validation_dataset = HipMRISliceDataset(args.manifest, "validation")
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
        pin_memory,
    )
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = count_parameters(model)
    history: list[Dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["model_name"] != args.model:
            raise ValueError(
                f"Checkpoint model is {checkpoint['model_name']}, requested model is {args.model}."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = load_history(args.output / "history.csv")
        if history and int(history[-1]["epoch"]) != start_epoch - 1:
            raise ValueError("history.csv does not match the resume checkpoint epoch.")
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already completed epoch {start_epoch - 1}; "
                f"--epochs must be at least {start_epoch}."
            )

    config = vars(args).copy()
    config.update(
        {
            "device_resolved": str(device),
            "device_name": describe_device(device),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "trainable_parameters": parameter_count,
        }
    )
    save_json(args.output / "config.json", config)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loader = make_loader(
            train_dataset,
            args.batch_size,
            True,
            args.num_workers,
            args.seed + epoch,
            pin_memory,
        )
        train_statistics = run_epoch(
            model,
            train_loader,
            args.model,
            device,
            args.vae_beta,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        with torch.inference_mode():
            validation_statistics = run_epoch(
                model,
                validation_loader,
                args.model,
                device,
                args.vae_beta,
                max_batches=args.max_validation_batches,
                include_reconstruction_metrics=True,
            )
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "epoch_seconds": epoch_seconds,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        row.update({f"train_{key}": value for key, value in train_statistics.items()})
        row.update({f"validation_{key}": value for key, value in validation_statistics.items()})
        history.append(row)
        save_history(args.output / "history.csv", history)

        current_validation_loss = validation_statistics["loss"]
        is_best = current_validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = current_validation_loss
            best_epoch = epoch
        checkpoint = {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "seed": args.seed,
            "config": _json_safe(config),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if is_best:
            torch.save(checkpoint, args.output / "best.pt")

        codebook_text = ""
        if args.model == "vqvae":
            codebook_text = (
                f" train_perplexity={train_statistics['codebook_perplexity']:.2f}"
                f" train_active_codes={int(train_statistics['active_codes'])}"
                f" val_perplexity={validation_statistics['codebook_perplexity']:.2f}"
                f" val_active_codes={int(validation_statistics['active_codes'])}"
            )
        print(
            f"epoch={epoch:03d} train_loss={train_statistics['loss']:.6f} "
            f"val_loss={current_validation_loss:.6f} "
            f"val_psnr={validation_statistics['psnr']:.3f} "
            f"val_ssim={validation_statistics['ssim']:.4f} "
            f"seconds={epoch_seconds:.2f}{codebook_text}"
        )

    synchronize(device)
    segment_training_seconds = time.perf_counter() - training_started
    recorded_epoch_seconds = sum(row["epoch_seconds"] for row in history)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "completed_epochs": int(history[-1]["epoch"]),
        "training_seconds": recorded_epoch_seconds,
        "latest_segment_seconds": segment_training_seconds,
        "trainable_parameters": parameter_count,
        "peak_accelerator_memory_mb": peak_memory_mb(device),
        "device": str(device),
        "device_name": describe_device(device),
        "sanity_limited": args.max_train_batches is not None
        or args.max_validation_batches is not None,
    }
    save_json(args.output / "summary.json", summary)
    print(f"Saved run artifacts to {args.output}")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
