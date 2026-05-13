"""Contains `sharp eval-h3ds` CLI implementation.

Evaluates SHARP on the H3DS multi-view face dataset. Uses view 0 of each
scene's selected views-config as the input image, predicts gaussians, then
renders at every other view's camera pose and computes PSNR/SSIM/LPIPS/DISTS
against the corresponding ground-truth image.

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
from PIL import Image

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import SceneMetaData, save_ply
from sharp.utils.metrics import compute_all as compute_metrics

from .predict import DEFAULT_MODEL_URL, predict_image
from .render import render_at_pose, render_gaussians

LOGGER = logging.getLogger(__name__)


def _resize_with_intrinsics(
    image_np: np.ndarray, K: np.ndarray, max_side: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image so max(H, W) <= max_side; scale K to match."""
    if max_side is None:
        return image_np, K.astype(np.float32)
    h, w = image_np.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_np, K.astype(np.float32)
    scale = max_side / longest
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image_pil = Image.fromarray(image_np)
    resized = np.array(image_pil.resize((new_w, new_h), Image.BILINEAR))
    K_new = K.astype(np.float32).copy()
    K_new[0] *= new_w / w
    K_new[1] *= new_h / h
    return resized, K_new


def _resize_mask_to(mask_pil: Image.Image, width: int, height: int) -> np.ndarray:
    """Resize a PIL mask via nearest neighbor; return (H, W) bool."""
    resized = mask_pil.resize((width, height), Image.NEAREST)
    arr = np.array(resized)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 127


def _resolve_device(device: str) -> str:
    if device != "default":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.command()
@click.option(
    "--h3ds-path",
    "h3ds_path",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Local directory for the H3DS dataset (created if missing).",
)
@click.option(
    "--h3ds-token",
    "h3ds_token",
    default=None,
    help="H3DS_ACCESS_TOKEN. If provided, runs h3ds.download() to fetch missing scenes.",
)
@click.option(
    "--config-id",
    default="config_v2",
    help="H3DS config id ('config_v1' or 'config_v2'). v2.0 is recommended.",
)
@click.option(
    "--scene-ids",
    default=None,
    help="Comma-separated H3DS scene IDs. Defaults to first --num-scenes scenes.",
)
@click.option(
    "--num-scenes",
    default=2,
    type=int,
    help="If --scene-ids not given, how many scenes to evaluate. Kept small by default.",
)
@click.option(
    "--views-config",
    default=None,
    help=(
        "H3DS view-config id (e.g. '3', '4', '6', '32'). Selects a precomputed subset of views. "
        "Index 0 is the input view; the rest are candidate GT views. "
        "If omitted, all views are loaded and filtered by --max-angle."
    ),
)
@click.option(
    "--max-angle",
    default=30.0,
    type=float,
    help=(
        "Maximum rotation (degrees) between view 0 and a candidate GT view. "
        "Larger baselines look bad for a monocular predictor, so we filter them out."
    ),
)
@click.option(
    "--max-gt-views",
    default=5,
    type=int,
    help="Cap the number of GT views per scene (after --max-angle filtering).",
)
@click.option(
    "--max-side",
    default=1024,
    type=int,
    help="Resize images so max(H, W) <= this; intrinsics scaled accordingly.",
)
@click.option(
    "--apply-mask/--no-apply-mask",
    "apply_mask",
    default=True,
    help="Multiply both render and GT by H3DS foreground masks before computing metrics.",
)
@click.option(
    "--world-scale",
    default=1.0,
    type=float,
    help=(
        "Multiplier applied to H3DS pose translations. SHARP predicts gaussians in metric "
        "meters; if H3DS poses are in different units (e.g. millimeters), set this to convert "
        "(0.001 for mm -> m). Use the pose-diagnostics logged at INFO to check |t| magnitudes."
    ),
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Where to write per-scene renders + metrics.csv.",
)
@click.option(
    "-c",
    "--checkpoint-path",
    default=None,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to a .pt checkpoint. Auto-downloads the default model if not given.",
)
@click.option(
    "--save-ply/--no-save-ply",
    "save_gaussians",
    default=False,
    help="Also save the predicted gaussians for each scene as .ply.",
)
@click.option(
    "--render-video/--no-render-video",
    "render_video",
    default=False,
    help="Also write the SHARP trajectory mp4 (color + depth) per scene, like `sharp predict --render`.",
)
@click.option(
    "--device",
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']. gsplat needs CUDA.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def eval_h3ds_cli(
    h3ds_path: Path,
    h3ds_token: str | None,
    config_id: str,
    scene_ids: str | None,
    num_scenes: int,
    views_config: str | None,
    max_angle: float,
    max_gt_views: int,
    max_side: int,
    apply_mask: bool,
    world_scale: float,
    output_path: Path,
    checkpoint_path: Path | None,
    save_gaussians: bool,
    render_video: bool,
    device: str,
    verbose: bool,
):
    """Evaluate SHARP on H3DS: input=view 0, GT=other views, real novel-view metrics."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    try:
        from h3ds.dataset import H3DS  # noqa: PLC0415
    except ImportError as exc:
        raise click.ClickException(
            "h3ds package is required: pip install h3ds (see https://github.com/CrisalixSA/h3ds)."
        ) from exc

    h3ds_path.mkdir(parents=True, exist_ok=True)
    h3ds = H3DS(path=str(h3ds_path), config_id=config_id)

    if h3ds_token:
        LOGGER.info("Downloading / verifying H3DS dataset under %s ...", h3ds_path)
        h3ds.download(token=h3ds_token)

    if not h3ds.is_available():
        raise click.ClickException(
            "H3DS dataset not available locally; pass --h3ds-token to download it."
        )

    if scene_ids:
        scenes_to_eval = [s.strip() for s in scene_ids.split(",") if s.strip()]
    else:
        all_scenes = h3ds.scenes()
        scenes_to_eval = all_scenes[: max(1, num_scenes)]
    LOGGER.info("Evaluating %d scene(s): %s", len(scenes_to_eval), scenes_to_eval)

    device = _resolve_device(device)
    LOGGER.info("Using device %s", device)
    if device != "cuda":
        LOGGER.warning(
            "Rendering on '%s'. gsplat's rasterizer is CUDA-only in most prebuilt wheels; "
            "render calls will likely fail outside CUDA.",
            device,
        )
    device_pt = torch.device(device)

    # Load checkpoint + predictor.
    if checkpoint_path is None:
        LOGGER.info("Downloading default SHARP model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device_pt)

    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for scene_id in scenes_to_eval:
        LOGGER.info("==== scene %s ====", scene_id)

        if views_config is not None:
            available_configs = [str(c) for c in h3ds.default_views_configs(scene_id)]
            chosen_config: str | None = str(views_config)
            if chosen_config not in available_configs:
                LOGGER.warning(
                    "Scene %s has no views-config '%s'. Available: %s. Skipping.",
                    scene_id,
                    chosen_config,
                    available_configs,
                )
                continue
        else:
            chosen_config = None  # load all views

        try:
            _, images_pil, masks_pil, cameras = h3ds.load_scene(
                scene_id=scene_id, views_config_id=chosen_config
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load scene %s: %s", scene_id, exc)
            continue

        if len(images_pil) < 2:
            LOGGER.warning(
                "Scene %s only has %d view(s) in config '%s'; need >=2 (1 input + >=1 GT). Skip.",
                scene_id,
                len(images_pil),
                chosen_config,
            )
            continue

        scene_dir = output_path / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        # View 0 := input view.
        input_np = np.array(images_pil[0])
        K0, pose0 = cameras[0]
        K0 = np.asarray(K0, dtype=np.float32)
        pose0 = np.asarray(pose0, dtype=np.float32).copy()
        pose0[:3, 3] *= world_scale
        input_np, K0_resized = _resize_with_intrinsics(input_np, K0, max_side)
        h_in, w_in = input_np.shape[:2]
        f_px = float((K0_resized[0, 0] + K0_resized[1, 1]) / 2.0)
        io.save_image(input_np, scene_dir / "view_00_input.png")

        LOGGER.info(
            "Predicting gaussians from view 0 (HxW=%dx%d, f_px=%.1f).", h_in, w_in, f_px
        )
        gaussians = predict_image(predictor, input_np, f_px, device_pt)

        if save_gaussians:
            save_ply(gaussians, f_px, (h_in, w_in), scene_dir / f"{scene_id}.ply")

        # --- Self-render at view 0 (identity extrinsics). Sanity check + metrics
        # for input-view reconstruction. Mask via view-0 mask so we evaluate on
        # the face region only.
        K0_3x3 = torch.from_numpy(K0_resized[:3, :3].astype(np.float32))
        try:
            self_rendered = render_at_pose(
                gaussians=gaussians,
                extrinsics=torch.eye(4, dtype=torch.float32),
                intrinsics_3x3=K0_3x3,
                image_height=h_in,
                image_width=w_in,
                color_space="linearRGB",
                device=device_pt,
            )
            self_rendered_np = (
                (self_rendered.permute(1, 2, 0).cpu().numpy() * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            io.save_image(self_rendered_np, scene_dir / "view_00_self_render.png")
            io.save_image(
                np.concatenate([input_np, self_rendered_np], axis=1),
                scene_dir / "view_00_self_compare.png",
            )

            input_t = (
                torch.from_numpy(input_np.copy()).float().permute(2, 0, 1) / 255.0
            ).to(self_rendered.device)
            if apply_mask:
                mask_bool_self = _resize_mask_to(masks_pil[0], w_in, h_in)
                mask_t_self = (
                    torch.from_numpy(mask_bool_self).to(self_rendered.device)[None].float()
                )
                rendered_for_metrics_self = self_rendered * mask_t_self
                gt_for_metrics_self = input_t * mask_t_self
            else:
                rendered_for_metrics_self = self_rendered
                gt_for_metrics_self = input_t

            self_metrics = compute_metrics(rendered_for_metrics_self, gt_for_metrics_self)
            LOGGER.info(
                "scene=%s view=0 (self) psnr=%.2f ssim=%.4f lpips=%.4f dists=%.4f",
                scene_id,
                self_metrics["psnr"],
                self_metrics["ssim"],
                self_metrics["lpips"],
                self_metrics["dists"],
            )
            rows.append(
                {"scene": scene_id, "view": 0, "angle": 0.0, **self_metrics}
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Self-render at view 0 failed: %s", exc)

        # --- Optional SHARP trajectory mp4 (color + depth).
        if render_video:
            try:
                metadata = SceneMetaData(f_px, (w_in, h_in), "linearRGB")
                render_gaussians(
                    gaussians, metadata, scene_dir / "trajectory.mp4", device=device_pt
                )
                LOGGER.info("Wrote trajectory video to %s/trajectory.mp4", scene_dir)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Trajectory render failed for %s: %s", scene_id, exc)

        # --- Filter GT views: compute relative pose for each candidate,
        # keep those within --max-angle, sort by angle, cap at --max-gt-views.
        candidate_views: list[tuple[int, float, float]] = []  # (idx, angle_deg, |t|)
        for cand_idx in range(1, len(images_pil)):
            _, pose_cand = cameras[cand_idx]
            pose_cand = np.asarray(pose_cand, dtype=np.float32).copy()
            pose_cand[:3, 3] *= world_scale
            rel = np.linalg.inv(pose_cand) @ pose0
            t = rel[:3, 3]
            cos_a = (np.trace(rel[:3, :3]) - 1.0) / 2.0
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            candidate_views.append((cand_idx, angle_deg, float(np.linalg.norm(t))))

        # Log all candidates (full diagnostic), then filter.
        for cand_idx, angle_deg, t_norm in candidate_views:
            LOGGER.info(
                "  pose[0->%d]: rot=%.1f deg, |t|=%.3f",
                cand_idx,
                angle_deg,
                t_norm,
            )
        kept = sorted(
            (c for c in candidate_views if c[1] <= max_angle), key=lambda c: c[1]
        )[: max(0, max_gt_views)]
        if not kept:
            LOGGER.warning(
                "Scene %s: no GT views within --max-angle=%.1f deg "
                "(closest was %.1f deg). Try a larger threshold or wider --views-config.",
                scene_id,
                max_angle,
                min((c[1] for c in candidate_views), default=float("nan")),
            )
        else:
            LOGGER.info(
                "Scene %s: evaluating %d GT view(s) within %.1f deg: %s",
                scene_id,
                len(kept),
                max_angle,
                [(idx, round(ang, 1)) for idx, ang, _ in kept],
            )

        for gt_idx, gt_angle, _ in kept:
            gt_image_pil = images_pil[gt_idx]
            gt_mask_pil = masks_pil[gt_idx]
            K_gt, pose_gt = cameras[gt_idx]
            K_gt = np.asarray(K_gt, dtype=np.float32)
            pose_gt = np.asarray(pose_gt, dtype=np.float32).copy()
            pose_gt[:3, 3] *= world_scale

            gt_np = np.array(gt_image_pil)
            gt_np, K_gt_resized = _resize_with_intrinsics(gt_np, K_gt, max_side)
            h_gt, w_gt = gt_np.shape[:2]

            # H3DS poses are camera-to-world (OpenCV). Gaussians live in view 0's
            # camera frame, so world-to-camera for the GT view is:
            #   extrinsics = inv(pose_gt) @ pose0
            extrinsics_np = np.linalg.inv(pose_gt) @ pose0
            extrinsics_pt = torch.from_numpy(extrinsics_np.astype(np.float32))
            K_gt_pt = torch.from_numpy(K_gt_resized.astype(np.float32))

            try:
                rendered = render_at_pose(
                    gaussians=gaussians,
                    extrinsics=extrinsics_pt,
                    intrinsics_3x3=K_gt_pt,
                    image_height=h_gt,
                    image_width=w_gt,
                    color_space="linearRGB",
                    device=device_pt,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Render failed for scene=%s view=%d: %s", scene_id, gt_idx, exc)
                continue

            gt_t = (
                torch.from_numpy(gt_np.copy()).float().permute(2, 0, 1) / 255.0
            ).to(rendered.device)

            if apply_mask:
                mask_bool = _resize_mask_to(gt_mask_pil, w_gt, h_gt)
                mask_t = torch.from_numpy(mask_bool).to(rendered.device)[None].float()
                rendered_for_metrics = rendered * mask_t
                gt_for_metrics = gt_t * mask_t
            else:
                rendered_for_metrics = rendered
                gt_for_metrics = gt_t

            metrics = compute_metrics(rendered_for_metrics, gt_for_metrics)
            LOGGER.info(
                "scene=%s view=%d (%.1f deg) psnr=%.2f ssim=%.4f lpips=%.4f dists=%.4f",
                scene_id,
                gt_idx,
                gt_angle,
                metrics["psnr"],
                metrics["ssim"],
                metrics["lpips"],
                metrics["dists"],
            )

            rendered_np = (
                (rendered.permute(1, 2, 0).cpu().numpy() * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            io.save_image(rendered_np, scene_dir / f"view_{gt_idx:02d}_render.png")
            io.save_image(gt_np, scene_dir / f"view_{gt_idx:02d}_gt.png")
            io.save_image(
                np.concatenate([gt_np, rendered_np], axis=1),
                scene_dir / f"view_{gt_idx:02d}_compare.png",
            )

            rows.append(
                {"scene": scene_id, "view": gt_idx, "angle": gt_angle, **metrics}
            )

    if not rows:
        LOGGER.error("No metrics computed. Check scene IDs / views-config / dataset state.")
        return

    csv_path = output_path / "metrics.csv"
    fieldnames = ["scene", "view", "angle", "psnr", "ssim", "lpips", "dists"]
    metric_fields = ["psnr", "ssim", "lpips", "dists"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        mean_row = {
            "scene": "MEAN",
            "view": "-",
            "angle": "-",
            **{k: float(np.mean([r[k] for r in rows])) for k in metric_fields},
        }
        writer.writerow(mean_row)
    LOGGER.info(
        "Wrote metrics to %s (mean psnr=%.2f ssim=%.4f lpips=%.4f dists=%.4f, n=%d).",
        csv_path,
        mean_row["psnr"],
        mean_row["ssim"],
        mean_row["lpips"],
        mean_row["dists"],
        len(rows),
    )
