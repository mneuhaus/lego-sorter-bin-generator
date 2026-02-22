"""Baseline and heatmap diff utilities for overlap verification SVGs."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import cairosvg
from PIL import Image, ImageChops, ImageOps, ImageStat


def _parse_viewbox_size(svg_path: str) -> tuple[float, float]:
    """Return (width_mm, height_mm) from an SVG viewBox."""
    root = ET.parse(svg_path).getroot()
    view_box = root.attrib.get("viewBox", "").strip()
    if not view_box:
        raise ValueError(f"SVG has no viewBox: {svg_path}")
    vals = [float(v) for v in view_box.split()]
    if len(vals) != 4:
        raise ValueError(f"Unexpected viewBox format in {svg_path!r}: {view_box!r}")
    return vals[2], vals[3]


def _render_svg_to_png(svg_path: str, png_path: str, width_px: int, height_px: int) -> None:
    """Render an SVG to PNG at an explicit pixel size."""
    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        output_width=max(1, width_px),
        output_height=max(1, height_px),
        background_color="#ffffff",
    )


def create_overlap_diff_heatmap(
    current_svg_path: str,
    baseline_svg_path: str,
    output_dir: str,
    output_prefix: str = "lasercut-verify-overlap",
    px_per_mm: float = 3.0,
    diff_threshold: int = 10,
) -> dict[str, str | float | int]:
    """Create a visual heatmap diff between overlap verification SVGs."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be > 0")
    if diff_threshold < 0 or diff_threshold > 255:
        raise ValueError("diff_threshold must be between 0 and 255")
    if not os.path.exists(current_svg_path):
        raise FileNotFoundError(f"Current overlap SVG not found: {current_svg_path}")
    if not os.path.exists(baseline_svg_path):
        raise FileNotFoundError(f"Baseline overlap SVG not found: {baseline_svg_path}")

    os.makedirs(output_dir, exist_ok=True)

    width_mm, height_mm = _parse_viewbox_size(current_svg_path)
    width_px = int(round(width_mm * px_per_mm))
    height_px = int(round(height_mm * px_per_mm))

    current_png = os.path.join(output_dir, f"{output_prefix}-current.png")
    baseline_png = os.path.join(output_dir, f"{output_prefix}-baseline.png")
    diff_gray_png = os.path.join(output_dir, f"{output_prefix}-diff-gray.png")
    diff_mask_png = os.path.join(output_dir, f"{output_prefix}-diff-mask.png")
    diff_heatmap_png = os.path.join(output_dir, f"{output_prefix}-diff-heatmap.png")
    diff_overlay_png = os.path.join(output_dir, f"{output_prefix}-diff-overlay.png")
    diff_summary_json = os.path.join(output_dir, f"{output_prefix}-diff-summary.json")

    _render_svg_to_png(current_svg_path, current_png, width_px, height_px)
    _render_svg_to_png(baseline_svg_path, baseline_png, width_px, height_px)

    current_rgba = Image.open(current_png).convert("RGBA")
    baseline_rgba = Image.open(baseline_png).convert("RGBA")

    current_gray = current_rgba.convert("L")
    baseline_gray = baseline_rgba.convert("L")
    diff_gray = ImageChops.difference(current_gray, baseline_gray)
    diff_gray.save(diff_gray_png)

    mask = diff_gray.point(lambda v: 255 if v >= diff_threshold else 0, mode="L")
    mask.save(diff_mask_png)

    # Heatmap: dark -> transparent, bright diff -> yellow/red
    heat_color = ImageOps.colorize(diff_gray, black="#001122", mid="#ffb000", white="#ff2200")
    alpha = mask.point(lambda v: int(v * 0.85 / 255.0), mode="L")
    heatmap_rgba = heat_color.convert("RGBA")
    heatmap_rgba.putalpha(alpha)
    heatmap_rgba.save(diff_heatmap_png)

    overlay = current_rgba.copy()
    overlay.alpha_composite(heatmap_rgba)
    overlay.save(diff_overlay_png)

    hist = mask.histogram()
    changed_pixels = hist[255] if len(hist) > 255 else 0
    total_pixels = width_px * height_px
    changed_ratio = (changed_pixels / total_pixels) if total_pixels > 0 else 0.0
    mean_diff = float(ImageStat.Stat(diff_gray).mean[0])
    max_diff = int(ImageStat.Stat(diff_gray).extrema[0][1])

    summary = {
        "current_svg": current_svg_path,
        "baseline_svg": baseline_svg_path,
        "render_px": [width_px, height_px],
        "px_per_mm": px_per_mm,
        "diff_threshold": diff_threshold,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_ratio": changed_ratio,
        "mean_abs_diff": mean_diff,
        "max_abs_diff": max_diff,
        "artifacts": {
            "current_png": current_png,
            "baseline_png": baseline_png,
            "diff_gray_png": diff_gray_png,
            "diff_mask_png": diff_mask_png,
            "diff_heatmap_png": diff_heatmap_png,
            "diff_overlay_png": diff_overlay_png,
        },
    }

    with open(diff_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "current_png": current_png,
        "baseline_png": baseline_png,
        "diff_gray_png": diff_gray_png,
        "diff_mask_png": diff_mask_png,
        "diff_heatmap_png": diff_heatmap_png,
        "diff_overlay_png": diff_overlay_png,
        "diff_summary_json": diff_summary_json,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_ratio": changed_ratio,
    }
