"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)
from sharp.utils.metrics import compute_all as compute_metrics

from .render import render_gaussians, render_input_view

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--metrics/--no-metrics",
    "with_metrics",
    is_flag=True,
    default=False,
    help="Render a novel view from the input pose and compute PSNR/SSIM/LPIPS/DISTS against the input image. Uses --device for rendering; gsplat's rasterizer is CUDA-only in most prebuilt wheels and may fail on MPS/CPU.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    with_metrics: bool,
    device: str,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if (with_rendering or with_metrics) and device != "cuda":
        LOGGER.warning(
            "Rendering on '%s'. gsplat's rasterizer is CUDA-only in most prebuilt wheels; "
            "render/metrics may fail at the rasterization step.",
            device,
        )

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    metrics_rows: list[dict[str, float | str]] = []

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        metadata = None
        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path, device=torch.device(device))

        if with_metrics:
            if metadata is None:
                metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")

            LOGGER.info("Rendering novel view at input pose and computing metrics.")
            rendered = render_input_view(
                gaussians, metadata, device=torch.device(device)
            )  # (3, H, W) in [0, 1]

            gt = (
                torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
            ).to(rendered.device)

            metrics = compute_metrics(rendered, gt)
            LOGGER.info(
                "%s metrics: psnr=%.3f ssim=%.4f lpips=%.4f dists=%.4f",
                image_path.name,
                metrics["psnr"],
                metrics["ssim"],
                metrics["lpips"],
                metrics["dists"],
            )

            rendered_np = (rendered.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(
                np.uint8
            )
            gt_np = (gt.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            io.save_image(rendered_np, output_path / f"{image_path.stem}_render.png")
            io.save_image(gt_np, output_path / f"{image_path.stem}_gt.png")
            io.save_image(
                np.concatenate([gt_np, rendered_np], axis=1),
                output_path / f"{image_path.stem}_compare.png",
            )

            metrics_rows.append({"image": image_path.name, **metrics})

    if with_metrics and metrics_rows:
        csv_path = output_path / "metrics.csv"
        fieldnames = ["image", "psnr", "ssim", "lpips", "dists"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(row)
            mean_row = {
                "image": "mean",
                **{
                    k: float(np.mean([r[k] for r in metrics_rows]))  # type: ignore[arg-type]
                    for k in fieldnames[1:]
                },
            }
            writer.writerow(mean_row)
        LOGGER.info(
            "Wrote metrics to %s (mean: psnr=%.3f ssim=%.4f lpips=%.4f dists=%.4f).",
            csv_path,
            mean_row["psnr"],
            mean_row["ssim"],
            mean_row["lpips"],
            mean_row["dists"],
        )


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians
