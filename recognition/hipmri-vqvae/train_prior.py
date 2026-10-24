"""Train a patient-disjoint PixelCNN prior over frozen VQ-VAE code grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from modules import PixelCNNPrior
from train import (
    _json_safe,
    configure_reproducibility,
    count_parameters,
    describe_device,
    peak_memory_mb,
    positive_int,
    save_history,
    save_json,
    select_device,
    synchronize,
)


@dataclass(frozen=True)
class PriorData:
    """Validated code grids and provenance from the train-only export."""

    indices: np.ndarray
    subject_ids: np.ndarray
    weeks: np.ndarray
    slice_indices: np.ndarray
    summary: dict[str, Any]
    data_sha256: str
    summary_sha256: str


class PriorGridDataset(Dataset[Tensor]):
    """In-memory compact-token grids used by the categorical prior."""

    def __init__(self, grids: np.ndarray) -> None:
        if grids.ndim != 3 or grids.shape[0] == 0:
            raise ValueError("Prior grids must have shape [samples, height, width].")
        self.grids = torch.from_numpy(np.ascontiguousarray(grids, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.grids.shape[0])

    def __getitem__(self, index: int) -> Tensor:
        return self.grids[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/formal-prior"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=positive_int, default=40)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--embedding-dim", type=positive_int, default=64)
    parser.add_argument("--hidden-channels", type=positive_int, default=64)
    parser.add_argument("--residual-blocks", type=positive_int, default=6)
    parser.add_argument("--validation-subjects", type=positive_int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-train-batches", type=positive_int)
    parser.add_argument("--max-validation-batches", type=positive_int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prior_data(data_path: Path, summary_path: Path) -> PriorData:
    data_path = data_path.resolve()
    summary_path = summary_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Prior data not found: {data_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Prior-data summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("source_split") != "train":
        raise ValueError("The prior data must come only from the training split.")
    if summary.get("limited_export") is not False:
        raise ValueError("A limited code export cannot be used to train the prior.")

    with np.load(data_path, allow_pickle=False) as archive:
        required = {"indices", "subject_id", "week", "slice_index"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Prior archive is missing arrays: {sorted(missing)}")
        indices = np.asarray(archive["indices"]).copy()
        subject_ids = np.asarray(archive["subject_id"]).astype(str, copy=True)
        weeks = np.asarray(archive["week"]).copy()
        slice_indices = np.asarray(archive["slice_index"]).copy()

    if indices.ndim != 3 or indices.shape[0] == 0:
        raise ValueError("indices must have shape [samples, height, width].")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("Exported VQ indices must be integers.")
    sample_count = int(indices.shape[0])
    for name, values in {
        "subject_id": subject_ids,
        "week": weeks,
        "slice_index": slice_indices,
    }.items():
        if values.ndim != 1 or len(values) != sample_count:
            raise ValueError(f"{name} metadata does not align with the code grids.")

    expected_count = int(summary.get("sample_count", -1))
    expected_shape = tuple(int(value) for value in summary.get("code_grid_shape", []))
    num_embeddings = int(summary.get("num_embeddings", 0))
    if sample_count != expected_count or tuple(indices.shape[1:]) != expected_shape:
        raise ValueError("Prior archive shape does not match its export summary.")
    if num_embeddings <= 1 or int(indices.min()) < 0 or int(indices.max()) >= num_embeddings:
        raise ValueError("An exported code index is outside the declared codebook.")
    if len(set(zip(subject_ids.tolist(), weeks.tolist(), slice_indices.tolist()))) != sample_count:
        raise ValueError("Duplicate subject/week/slice keys found in the prior archive.")

    exported_subjects = set(subject_ids.tolist())
    expected_subjects = set(str(value) for value in summary.get("subject_ids", []))
    held_out = summary.get("held_out_subject_ids", {})
    held_out_subjects = {
        str(value)
        for split in ("validation", "test")
        for value in held_out.get(split, [])
    }
    if exported_subjects != expected_subjects:
        raise ValueError("Prior archive patients do not match the export summary.")
    if exported_subjects & held_out_subjects:
        raise ValueError("Official validation/test patient leakage detected in prior data.")

    return PriorData(
        indices=indices,
        subject_ids=subject_ids,
        weeks=weeks,
        slice_indices=slice_indices,
        summary=summary,
        data_sha256=sha256_file(data_path),
        summary_sha256=sha256_file(summary_path),
    )


def patient_split(
    subject_ids: np.ndarray,
    validation_subject_count: int,
    seed: int,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    subjects = sorted(set(subject_ids.tolist()))
    if not 1 <= validation_subject_count < len(subjects):
        raise ValueError(
            "--validation-subjects must leave at least one patient in each prior split."
        )
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(subjects).tolist()
    validation_subjects = sorted(shuffled[:validation_subject_count])
    train_subjects = sorted(shuffled[validation_subject_count:])
    train_mask = np.isin(subject_ids, train_subjects)
    validation_mask = np.isin(subject_ids, validation_subjects)
    if np.any(train_mask & validation_mask) or not np.all(train_mask | validation_mask):
        raise RuntimeError("Internal patient split masks are invalid.")
    return train_subjects, validation_subjects, train_mask, validation_mask


def compact_vocabulary(
    original_indices: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    num_embeddings: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocabulary = np.unique(original_indices[train_mask]).astype(np.int64)
    if len(vocabulary) <= 1:
        raise ValueError("The prior-training split contains fewer than two active VQ codes.")
    mapping = np.full(num_embeddings, -1, dtype=np.int64)
    mapping[vocabulary] = np.arange(len(vocabulary), dtype=np.int64)
    train_grids = mapping[original_indices[train_mask].astype(np.int64)]
    validation_grids = mapping[original_indices[validation_mask].astype(np.int64)]
    unseen_validation = np.unique(original_indices[validation_mask][validation_grids < 0])
    if len(unseen_validation):
        raise ValueError(
            "Internal validation patients contain codes unseen in prior training: "
            f"{unseen_validation.tolist()}"
        )
    return vocabulary, train_grids, validation_grids


def make_loader(
    dataset: PriorGridDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Tensor]:
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


def run_epoch(
    model: PixelCNNPrior,
    loader: DataLoader[Tensor],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_negative_log_likelihood = 0.0
    total_correct = 0
    total_tokens = 0
    total_samples = 0
    batch_count = 0

    for batch_index, grids in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        grids = grids.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(grids)
            loss = F.cross_entropy(logits, grids)
            if training:
                loss.backward()
                optimizer.step()

        token_count = int(grids.numel())
        total_negative_log_likelihood += float(loss.detach().item()) * token_count
        total_correct += int((logits.detach().argmax(dim=1) == grids).sum().item())
        total_tokens += token_count
        total_samples += int(grids.shape[0])
        batch_count += 1

    if total_tokens == 0:
        raise RuntimeError("No prior batches were processed.")
    nll = total_negative_log_likelihood / total_tokens
    return {
        "negative_log_likelihood": nll,
        "bits_per_code": nll / math.log(2.0),
        "perplexity": math.exp(min(nll, 80.0)),
        "token_accuracy": total_correct / total_tokens,
        "samples": float(total_samples),
        "tokens": float(total_tokens),
        "batches": float(batch_count),
    }


def check_causality(model: PixelCNNPrior) -> None:
    model.eval()
    generator = torch.Generator().manual_seed(3710)
    base = torch.randint(model.num_tokens, (2, 4, 5), generator=generator)
    changed = base.clone()
    changed[:, 2, 3] = (changed[:, 2, 3] + 1) % model.num_tokens
    with torch.inference_mode():
        original_logits = model(base)
        changed_logits = model(changed)
    changed_position = 2 * 5 + 3
    original_prefix = original_logits.flatten(2)[:, :, : changed_position + 1]
    changed_prefix = changed_logits.flatten(2)[:, :, : changed_position + 1]
    if not torch.allclose(original_prefix, changed_prefix, atol=1e-7, rtol=0.0):
        raise RuntimeError("PixelCNN causality check failed: future/current input changed a prefix logit.")


def run_smoke_test() -> None:
    configure_reproducibility(3710)
    model = PixelCNNPrior(
        num_tokens=4,
        embedding_dim=8,
        hidden_channels=16,
        residual_blocks=2,
    )
    check_causality(model)
    rows = torch.arange(4).view(1, 4, 1)
    columns = torch.arange(5).view(1, 1, 5)
    grids = ((rows + columns) % model.num_tokens).repeat(8, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(grids), grids).item())
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(grids), grids)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(F.cross_entropy(model(grids), grids).item())
    if not final_loss < initial_loss:
        raise RuntimeError("PixelCNN smoke optimisation did not reduce the loss.")
    samples = model.sample(
        batch_size=2,
        grid_shape=(4, 5),
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3710),
    )
    if samples.shape != (2, 4, 5) or int(samples.min()) < 0 or int(samples.max()) >= 4:
        raise RuntimeError("PixelCNN smoke sampling returned invalid token grids.")
    print(
        json.dumps(
            {
                "causality": "pass",
                "gradient_update": "pass",
                "sampling": "pass",
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "sample_shape": list(samples.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )


def train_prior(args: argparse.Namespace) -> None:
    if args.data is None:
        raise ValueError("--data is required unless --smoke-test is used.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
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

    summary_path = args.data_summary or args.data.parent / "summary.json"
    data = load_prior_data(args.data, summary_path)
    train_subjects, validation_subjects, train_mask, validation_mask = patient_split(
        data.subject_ids,
        args.validation_subjects,
        args.seed,
    )
    num_embeddings = int(data.summary["num_embeddings"])
    vocabulary, train_grids, validation_grids = compact_vocabulary(
        data.indices,
        train_mask,
        validation_mask,
        num_embeddings,
    )
    train_dataset = PriorGridDataset(train_grids)
    validation_dataset = PriorGridDataset(validation_grids)

    configure_reproducibility(args.seed)
    device = select_device(args.device)
    pin_memory = device.type == "cuda"
    model_config = {
        "num_tokens": int(len(vocabulary)),
        "embedding_dim": args.embedding_dim,
        "hidden_channels": args.hidden_channels,
        "residual_blocks": args.residual_blocks,
    }
    model = PixelCNNPrior(**model_config).to(device)
    check_causality(model.cpu())
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = count_parameters(model)
    history: list[dict[str, float]] = []
    best_validation_nll = float("inf")
    best_epoch = 0
    start_epoch = 1

    provenance = {
        "data_path": str(args.data.resolve()),
        "data_sha256": data.data_sha256,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": data.summary_sha256,
        "source_vqvae_sha256": data.summary.get("checkpoint", {}).get("sha256"),
        "source_manifest_sha256": data.summary.get("manifest", {}).get("sha256"),
    }
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("model_name") != "pixelcnn_prior":
            raise ValueError("The resume checkpoint is not a PixelCNN prior.")
        if checkpoint.get("model_config") != model_config:
            raise ValueError("Resume checkpoint model configuration does not match the request.")
        if checkpoint.get("vocabulary") != vocabulary.tolist():
            raise ValueError("Resume checkpoint vocabulary does not match the exported data.")
        if checkpoint.get("train_subjects") != train_subjects:
            raise ValueError("Resume checkpoint patient split does not match the exported data.")
        if checkpoint.get("validation_subjects") != validation_subjects:
            raise ValueError("Resume checkpoint validation patients do not match the exported data.")
        if checkpoint.get("provenance", {}).get("data_sha256") != data.data_sha256:
            raise ValueError("Resume checkpoint was trained from a different code archive.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # ``map_location=device`` also moves the CPU RNG state onto CUDA.
        # PyTorch requires CPU ByteTensors when restoring generator states.
        torch.set_rng_state(checkpoint["torch_rng_state"].detach().cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in checkpoint["cuda_rng_states"]]
            )
        best_validation_nll = float(checkpoint["best_validation_nll"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = args.output / "history.csv"
        if history_path.is_file():
            import csv

            with history_path.open("r", newline="", encoding="utf-8") as handle:
                history = [
                    {key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(handle)
                ]
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
            "code_grid_shape": list(data.indices.shape[1:]),
            "original_codebook_size": num_embeddings,
            "compact_vocabulary": vocabulary.tolist(),
            "train_subjects": train_subjects,
            "validation_subjects": validation_subjects,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "trainable_parameters": parameter_count,
            "provenance": provenance,
        }
    )
    save_json(args.output / "config.json", config)

    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
        pin_memory,
    )
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
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        with torch.inference_mode():
            validation_statistics = run_epoch(
                model,
                validation_loader,
                device,
                max_batches=args.max_validation_batches,
            )
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        row: dict[str, float] = {
            "epoch": float(epoch),
            "epoch_seconds": epoch_seconds,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        row.update({f"train_{key}": value for key, value in train_statistics.items()})
        row.update({f"validation_{key}": value for key, value in validation_statistics.items()})
        history.append(row)
        save_history(args.output / "history.csv", history)

        validation_nll = validation_statistics["negative_log_likelihood"]
        is_best = validation_nll < best_validation_nll
        if is_best:
            best_validation_nll = validation_nll
            best_epoch = epoch
        checkpoint = {
            "model_name": "pixelcnn_prior",
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_nll": best_validation_nll,
            "seed": args.seed,
            "grid_shape": list(data.indices.shape[1:]),
            "vocabulary": vocabulary.tolist(),
            "train_subjects": train_subjects,
            "validation_subjects": validation_subjects,
            "provenance": provenance,
            "config": _json_safe(config),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if is_best:
            torch.save(checkpoint, args.output / "best.pt")

        print(
            f"epoch={epoch:03d} "
            f"train_nll={train_statistics['negative_log_likelihood']:.6f} "
            f"val_nll={validation_nll:.6f} "
            f"val_perplexity={validation_statistics['perplexity']:.3f} "
            f"val_accuracy={validation_statistics['token_accuracy']:.4f} "
            f"seconds={epoch_seconds:.2f}",
            flush=True,
        )

    synchronize(device)
    segment_training_seconds = time.perf_counter() - training_started
    recorded_epoch_seconds = sum(row["epoch_seconds"] for row in history)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "best_validation_perplexity": math.exp(min(best_validation_nll, 80.0)),
        "completed_epochs": int(history[-1]["epoch"]),
        "training_seconds": recorded_epoch_seconds,
        "latest_segment_seconds": segment_training_seconds,
        "trainable_parameters": parameter_count,
        "peak_accelerator_memory_mb": peak_memory_mb(device),
        "device": str(device),
        "device_name": describe_device(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "compact_vocabulary_size": int(len(vocabulary)),
        "compact_vocabulary": vocabulary.tolist(),
        "sanity_limited": args.max_train_batches is not None
        or args.max_validation_batches is not None,
        "provenance": provenance,
    }
    save_json(args.output / "summary.json", summary)
    print(f"Saved prior run artifacts to {args.output}")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        train_prior(args)


if __name__ == "__main__":
    main()
