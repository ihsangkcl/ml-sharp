"""Contains `sharp h3ds-summary` CLI: turns an eval-h3ds output dir into slide assets.

Produces under `<eval-path>/summary/`:
  - summary.csv          aggregate stats per angle bucket and overall
  - summary.md           markdown table (copy-paste into slides)
  - metrics_vs_angle.png 2x2 chart: PSNR/SSIM/LPIPS/DISTS vs angle
  - slide_grid.png       grid figure of input | self | novel-view examples

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)


METRIC_KEYS = ("psnr", "ssim", "lpips", "dists")
METRIC_PRETTY = {
    "psnr": "PSNR ↑",
    "ssim": "SSIM ↑",
    "lpips": "LPIPS ↓",
    "dists": "DISTS ↓",
}


def _load_metrics(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scene") == "MEAN":
                continue
            try:
                cleaned = {
                    "scene": row["scene"],
                    "view": int(row["view"]),
                    "angle": float(row.get("angle", 0.0)),
                }
                for k in METRIC_KEYS:
                    cleaned[k] = float(row[k])
                rows.append(cleaned)
            except (KeyError, ValueError) as exc:
                LOGGER.debug("Skipping row %s (%s)", row, exc)
    return rows


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _aggregate(rows: list[dict], bucket_edges: list[float]) -> list[dict]:
    """Aggregate by angle buckets. The bucket [0,0] holds the self rows."""
    out: list[dict] = []

    self_rows = [r for r in rows if r["view"] == 0]
    if self_rows:
        entry = {"label": "self (view 0)", "n": len(self_rows)}
        for k in METRIC_KEYS:
            mean, std = _stats([r[k] for r in self_rows])
            entry[f"{k}_mean"] = mean
            entry[f"{k}_std"] = std
        out.append(entry)

    gt_rows = [r for r in rows if r["view"] != 0]
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        bucket = [r for r in gt_rows if lo <= r["angle"] < hi]
        if not bucket:
            continue
        entry = {"label": f"novel [{lo:.0f}, {hi:.0f}) deg", "n": len(bucket)}
        for k in METRIC_KEYS:
            mean, std = _stats([r[k] for r in bucket])
            entry[f"{k}_mean"] = mean
            entry[f"{k}_std"] = std
        out.append(entry)

    if gt_rows:
        entry = {"label": "novel (all)", "n": len(gt_rows)}
        for k in METRIC_KEYS:
            mean, std = _stats([r[k] for r in gt_rows])
            entry[f"{k}_mean"] = mean
            entry[f"{k}_std"] = std
        out.append(entry)

    if rows:
        entry = {"label": "overall", "n": len(rows)}
        for k in METRIC_KEYS:
            mean, std = _stats([r[k] for r in rows])
            entry[f"{k}_mean"] = mean
            entry[f"{k}_std"] = std
        out.append(entry)

    return out


def _write_summary_csv(stats: list[dict], output: Path) -> None:
    fieldnames = ["label", "n"] + [f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std")]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)


def _write_summary_markdown(stats: list[dict], output: Path) -> None:
    lines = ["| split | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in stats:
        lines.append(
            f"| {s['label']} | {s['n']} "
            f"| {s['psnr_mean']:.2f} ± {s['psnr_std']:.2f} "
            f"| {s['ssim_mean']:.3f} ± {s['ssim_std']:.3f} "
            f"| {s['lpips_mean']:.3f} ± {s['lpips_std']:.3f} "
            f"| {s['dists_mean']:.3f} ± {s['dists_std']:.3f} |"
        )
    output.write_text("\n".join(lines) + "\n")


def _plot_metrics_vs_angle(rows: list[dict], output: Path) -> None:
    gt = [r for r in rows if r["view"] != 0]
    if not gt:
        LOGGER.warning("No novel-view rows; skipping metrics_vs_angle plot.")
        return
    angles = np.array([r["angle"] for r in gt])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key in zip(axes.flat, METRIC_KEYS):
        vals = np.array([r[key] for r in gt])
        ax.scatter(angles, vals, alpha=0.5, s=22, edgecolors="none")
        if len(angles) >= 4:
            bins = np.linspace(0, max(angles.max(), 1.0), 6)
            centers, means, stds = [], [], []
            for i in range(len(bins) - 1):
                m = (angles >= bins[i]) & (angles < bins[i + 1])
                if m.sum() >= 1:
                    centers.append(0.5 * (bins[i] + bins[i + 1]))
                    means.append(vals[m].mean())
                    stds.append(vals[m].std())
            if centers:
                centers, means, stds = map(np.array, (centers, means, stds))
                ax.plot(centers, means, "r-", linewidth=2, label="bin mean")
                ax.fill_between(centers, means - stds, means + stds, color="r", alpha=0.15)
                ax.legend(loc="best", fontsize=9)
        ax.set_xlabel("rotation angle (deg)")
        ax.set_title(METRIC_PRETTY[key])
        ax.grid(True, alpha=0.3)
    fig.suptitle("SHARP on H3DS — novel-view metrics vs rotation angle", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_slide_grid(
    eval_path: Path,
    rows: list[dict],
    output: Path,
    num_scenes: int = 4,
) -> None:
    """Grid: rows = scenes (median-PSNR), cols = input | self | each novel view (GT|render)."""
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    if not by_scene:
        LOGGER.warning("Empty rows; skipping slide_grid.")
        return

    scenes_sorted = sorted(by_scene.items(), key=lambda kv: float(np.mean([r["psnr"] for r in kv[1]])))
    if len(scenes_sorted) > num_scenes:
        start = (len(scenes_sorted) - num_scenes) // 2
        scenes_sorted = scenes_sorted[start : start + num_scenes]
    n_scenes = len(scenes_sorted)

    max_novel = max(
        sum(1 for r in scene_rows if r["view"] != 0) for _, scene_rows in scenes_sorted
    )
    cols = 2 + max_novel  # input + self + each novel
    if cols < 3:
        cols = 3

    fig, axes = plt.subplots(n_scenes, cols, figsize=(2.6 * cols, 2.8 * n_scenes))
    if n_scenes == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes[:, None]

    col_titles_set = False
    for r_idx, (scene_id, scene_rows) in enumerate(scenes_sorted):
        scene_rows = sorted(scene_rows, key=lambda r: r["angle"])
        scene_dir = eval_path / scene_id

        # Column 0: input
        ax = axes[r_idx, 0]
        ax.axis("off")
        p = scene_dir / "view_00_input.png"
        if p.exists():
            ax.imshow(np.asarray(Image.open(p).convert("RGB")))
        ax.text(
            -0.04, 0.5, scene_id[:8], rotation=90, transform=ax.transAxes,
            va="center", ha="right", fontsize=8, color="gray",
        )

        # Column 1: self render
        ax = axes[r_idx, 1]
        ax.axis("off")
        p = scene_dir / "view_00_self_render.png"
        if p.exists():
            ax.imshow(np.asarray(Image.open(p).convert("RGB")))
        self_row = next((r for r in scene_rows if r["view"] == 0), None)
        if self_row:
            ax.set_xlabel(f"psnr={self_row['psnr']:.1f}", fontsize=8)

        # Cols 2..: novel views, side-by-side compare images
        col = 2
        for r in scene_rows:
            if r["view"] == 0:
                continue
            if col >= cols:
                break
            ax = axes[r_idx, col]
            ax.axis("off")
            cmp_path = scene_dir / f"view_{r['view']:02d}_compare.png"
            if cmp_path.exists():
                ax.imshow(np.asarray(Image.open(cmp_path).convert("RGB")))
            ax.set_xlabel(
                f"{r['angle']:.0f}°  psnr={r['psnr']:.1f}  lpips={r['lpips']:.2f}",
                fontsize=8,
            )
            col += 1
        # blank remaining
        while col < cols:
            axes[r_idx, col].axis("off")
            col += 1

        if not col_titles_set:
            axes[0, 0].set_title("input", fontsize=10)
            axes[0, 1].set_title("self-render", fontsize=10)
            for c in range(2, cols):
                axes[0, c].set_title(f"novel view (GT | render)", fontsize=10)
            col_titles_set = True

    fig.suptitle(
        "SHARP on H3DS — input view, self-render, and novel views (left = GT, right = render)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.01, 0, 1, 0.96))
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option(
    "-i",
    "--eval-path",
    "eval_path",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    required=True,
    help="Output directory from a previous `sharp eval-h3ds` run (must contain metrics.csv).",
)
@click.option(
    "-o",
    "--summary-path",
    "summary_path",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Where to write summary outputs. Defaults to <eval-path>/summary/.",
)
@click.option(
    "--bucket-edges",
    default="0,15,30,45,60,90",
    help="Comma-separated angle bucket edges (degrees). Buckets are [edge_i, edge_{i+1}).",
)
@click.option(
    "--grid-scenes",
    default=4,
    type=int,
    help="Number of scenes to include in the slide grid (medians by PSNR).",
)
@click.option("-v", "--verbose", is_flag=True)
def h3ds_summary_cli(
    eval_path: Path,
    summary_path: Path | None,
    bucket_edges: str,
    grid_scenes: int,
    verbose: bool,
):
    """Aggregate eval-h3ds output into slide assets: summary.csv, summary.md, plot, grid."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    csv_path = eval_path / "metrics.csv"
    if not csv_path.exists():
        raise click.ClickException(f"metrics.csv not found in {eval_path}.")

    rows = _load_metrics(csv_path)
    if not rows:
        raise click.ClickException(f"No usable metric rows in {csv_path}.")

    edges = [float(x) for x in bucket_edges.split(",") if x.strip()]
    if len(edges) < 2:
        raise click.ClickException("--bucket-edges needs at least two numbers.")

    out_dir = summary_path if summary_path else eval_path / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = _aggregate(rows, edges)
    _write_summary_csv(stats, out_dir / "summary.csv")
    _write_summary_markdown(stats, out_dir / "summary.md")
    LOGGER.info("Wrote %s and summary.md", out_dir / "summary.csv")

    _plot_metrics_vs_angle(rows, out_dir / "metrics_vs_angle.png")
    LOGGER.info("Wrote %s", out_dir / "metrics_vs_angle.png")

    _build_slide_grid(eval_path, rows, out_dir / "slide_grid.png", num_scenes=grid_scenes)
    LOGGER.info("Wrote %s", out_dir / "slide_grid.png")

    LOGGER.info("Done. Slide assets in %s", out_dir)
