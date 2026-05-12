"""Image quality metrics: PSNR, SSIM, LPIPS, DISTS.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is shaped (B, 3, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected (3,H,W) or (B,3,H,W), got shape {tuple(x.shape)}.")
    return x


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR between two images in [0, 1]. Shape (3,H,W) or (B,3,H,W)."""
    pred = _as_batch(pred)
    target = _as_batch(target)
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((max_val**2) / mse)


def _gaussian_window(
    window_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """SSIM between two images in [0, 1]. Shape (3,H,W) or (B,3,H,W)."""
    pred = _as_batch(pred)
    target = _as_batch(target)

    channels = pred.shape[1]
    window_1d = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window_2d = window_1d[:, None] * window_1d[None, :]
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_y = F.conv2d(target, window, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean().item()


@functools.lru_cache(maxsize=4)
def _get_lpips_model(net: str, device: str):
    import lpips  # noqa: PLC0415

    model = lpips.LPIPS(net=net, verbose=False).to(device)
    model.eval()
    return model


@torch.no_grad()
def lpips_score(pred: torch.Tensor, target: torch.Tensor, net: str = "vgg") -> float:
    """LPIPS between two images in [0, 1]. Shape (3,H,W) or (B,3,H,W)."""
    pred = _as_batch(pred)
    target = _as_batch(target)
    model = _get_lpips_model(net, str(pred.device))
    # lpips expects inputs in [-1, 1]
    return model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().item()


@functools.lru_cache(maxsize=2)
def _get_dists_model(device: str):
    from DISTS_pytorch import DISTS  # noqa: PLC0415

    model = DISTS().to(device)
    model.eval()
    return model


@torch.no_grad()
def dists_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """DISTS between two images in [0, 1]. Shape (3,H,W) or (B,3,H,W)."""
    pred = _as_batch(pred)
    target = _as_batch(target)
    model = _get_dists_model(str(pred.device))
    return model(pred, target).mean().item()


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute PSNR / SSIM / LPIPS / DISTS for two [0, 1] images on the same device."""
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "lpips": lpips_score(pred, target),
        "dists": dists_score(pred, target),
    }
