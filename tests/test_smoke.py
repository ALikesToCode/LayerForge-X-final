from __future__ import annotations

import numpy as np

from layerforge.alpha import estimate_alpha
from layerforge.compose import rgba_from_rgb_alpha
from layerforge.depth import geometric_luminance_depth
from layerforge.metrics import compute_run_metrics
from layerforge.segment import classical_segments
from layerforge.types import Layer


def test_basics() -> None:
    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    rgb[:35] = [180, 210, 240]
    rgb[35:] = [90, 140, 90]
    rgb[28:58, 30:60] = [220, 80, 70]
    depth = geometric_luminance_depth(rgb).depth
    segs = classical_segments(rgb, {"min_area_ratio": 0.002, "slic_segments": 24, "slic_compactness": 10.0, "nms_iou": 0.9})
    assert depth.shape == rgb.shape[:2]
    assert len(segs) > 0
    alpha = estimate_alpha(rgb, segs[0].mask, depth, {"alpha_band_px": 5, "preserve_depth_edges": True})
    rgba = rgba_from_rgb_alpha(rgb, alpha)
    assert rgba.shape == (*rgb.shape[:2], 4)


def test_compute_run_metrics_handles_tiny_images() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    alpha = np.ones((4, 4), dtype=np.float32)
    rgba = rgba_from_rgb_alpha(rgb, alpha)
    layer = Layer(
        id=0,
        name="000_object",
        label="object",
        group="object",
        rank=0,
        depth_median=0.0,
        depth_p10=0.0,
        depth_p90=0.0,
        area=16,
        bbox=(0, 0, 4, 4),
        alpha=alpha,
        rgba=rgba,
        albedo_rgba=rgba,
        shading_rgba=rgba,
        visible_mask=alpha > 0,
    )
    metrics = compute_run_metrics(rgb, [layer], {})
    assert metrics["recompose_ssim"] == 1.0
