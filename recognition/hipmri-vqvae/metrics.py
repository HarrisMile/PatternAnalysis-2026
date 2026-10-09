"""Reconstruction metrics shared by training and inference."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F


def _validate_images(targets: Tensor, predictions: Tensor) -> None:
    if targets.shape != predictions.shape:
        raise ValueError(
            f"Metric inputs must have the same shape, received {targets.shape} and {predictions.shape}."
        )
    if targets.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W] tensors, received {targets.shape}.")


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _structural_similarity_map(
    targets: Tensor,
    predictions: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Return the local-window SSIM map for inputs scaled to [0, 1]."""
    _validate_images(targets, predictions)
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be an odd integer of at least 3.")
    if min(targets.shape[-2:]) <= window_size // 2:
        raise ValueError(f"Images are too small for a {window_size}x{window_size} SSIM window.")

    channels = targets.shape[1]
    window = _gaussian_window(
        window_size,
        sigma,
        channels,
        targets.device,
        targets.dtype,
    )
    padding = window_size // 2
    target_padded = F.pad(targets, (padding, padding, padding, padding), mode="reflect")
    prediction_padded = F.pad(predictions, (padding, padding, padding, padding), mode="reflect")

    mu_target = F.conv2d(target_padded, window, groups=channels)
    mu_prediction = F.conv2d(prediction_padded, window, groups=channels)
    mu_target_squared = mu_target.square()
    mu_prediction_squared = mu_prediction.square()
    mu_product = mu_target * mu_prediction

    variance_target = (
        F.conv2d(target_padded.square(), window, groups=channels) - mu_target_squared
    ).clamp_min(0.0)
    variance_prediction = (
        F.conv2d(prediction_padded.square(), window, groups=channels) - mu_prediction_squared
    ).clamp_min(0.0)
    covariance = (
        F.conv2d(target_padded * prediction_padded, window, groups=channels) - mu_product
    )

    constant_1 = (0.01 * data_range) ** 2
    constant_2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_product + constant_1) * (2 * covariance + constant_2)
    denominator = (
        (mu_target_squared + mu_prediction_squared + constant_1)
        * (variance_target + variance_prediction + constant_2)
    )
    return (numerator / denominator.clamp_min(torch.finfo(targets.dtype).eps)).clamp(
        -1.0, 1.0
    )


def structural_similarity(
    targets: Tensor,
    predictions: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Return standard local-window SSIM for each image in a batch."""
    score_map = _structural_similarity_map(
        targets,
        predictions,
        data_range=data_range,
        window_size=window_size,
        sigma=sigma,
    )
    return score_map.mean(dim=(1, 2, 3))


def reconstruction_metrics(targets: Tensor, predictions: Tensor) -> Dict[str, Tensor]:
    """Return per-image MSE, MAE, PSNR, and SSIM for inputs scaled to [0, 1]."""
    _validate_images(targets, predictions)
    errors = predictions - targets
    mse = errors.square().mean(dim=(1, 2, 3))
    mae = errors.abs().mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    ssim = structural_similarity(targets, predictions)
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def region_reconstruction_metrics(
    targets: Tensor,
    predictions: Tensor,
    masks: Tensor,
) -> Dict[str, Tensor]:
    """Return foreground/background metrics using nonzero mask labels as foreground."""
    _validate_images(targets, predictions)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4 or masks.shape[0] != targets.shape[0]:
        raise ValueError(
            "Masks must have shape [N, H, W] or [N, 1, H, W] with the same batch size."
        )
    if masks.shape[1] != 1 or masks.shape[-2:] != targets.shape[-2:]:
        raise ValueError(f"Mask shape {masks.shape} is incompatible with images {targets.shape}.")

    errors = predictions - targets
    squared_errors = errors.square()
    absolute_errors = errors.abs()
    ssim_map = _structural_similarity_map(targets, predictions)
    results: Dict[str, Tensor] = {}

    foreground = masks > 0
    for region_name, region in (("foreground", foreground), ("background", ~foreground)):
        weights = region.expand(-1, targets.shape[1], -1, -1).to(targets.dtype)
        counts = weights.sum(dim=(1, 2, 3))
        valid = counts > 0
        safe_counts = counts.clamp_min(1.0)
        mse = (squared_errors * weights).sum(dim=(1, 2, 3)) / safe_counts
        mae = (absolute_errors * weights).sum(dim=(1, 2, 3)) / safe_counts
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        ssim = (ssim_map * weights).sum(dim=(1, 2, 3)) / safe_counts
        invalid = torch.full_like(mse, float("nan"))
        results.update(
            {
                f"{region_name}_mse": torch.where(valid, mse, invalid),
                f"{region_name}_mae": torch.where(valid, mae, invalid),
                f"{region_name}_psnr": torch.where(valid, psnr, invalid),
                f"{region_name}_ssim": torch.where(valid, ssim, invalid),
            }
        )
    return results
