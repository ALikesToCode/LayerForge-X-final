from __future__ import annotations

import numpy as np

from layerforge.compose import rgba_from_rgb_alpha
from layerforge.matting import make_trimap, refine_layer_alpha


def _antialiased_circle(size: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    center = (size - 1) / 2.0
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    alpha = np.clip((size * 0.33 - dist + 1.5) / 3.0, 0.0, 1.0).astype(np.float32)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = 230
    rgb[..., 1] = 80
    rgb[..., 2] = 40
    mask = alpha > 0.08
    depth = np.tile(np.linspace(0.0, 1.0, size, dtype=np.float32), (size, 1))
    return rgb, mask, depth


def test_native_heuristic_matting_produces_soft_alpha_for_antialiased_shape() -> None:
    rgb, mask, depth = _antialiased_circle()

    alpha, metadata = refine_layer_alpha(
        rgb,
        mask,
        depth,
        {"method": "heuristic", "alpha_band_px": 5, "preserve_depth_edges": False},
        device="cpu",
    )

    assert alpha.shape == mask.shape
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0
    assert np.any((alpha > 0.05) & (alpha < 0.95))
    assert metadata["backend_used"] is False
    assert 0.0 <= float(metadata["alpha_quality_score"]) <= 1.0
    assert set(np.unique(make_trimap(mask, band_px=5))).issubset({0, 128, 255})


def test_missing_matting_backend_falls_back_without_recomposition_regression() -> None:
    rgb, mask, depth = _antialiased_circle()
    heuristic, _ = refine_layer_alpha(
        rgb,
        mask,
        depth,
        {"method": "heuristic", "alpha_band_px": 5, "preserve_depth_edges": False},
        device="cpu",
    )
    fallback, metadata = refine_layer_alpha(
        rgb,
        mask,
        depth,
        {"method": "external", "alpha_band_px": 5, "preserve_depth_edges": False},
        device="cpu",
    )

    heuristic_rgba = rgba_from_rgb_alpha(rgb, heuristic)
    fallback_rgba = rgba_from_rgb_alpha(rgb, fallback)
    assert metadata["fallback_used"] is True
    assert metadata["backend_used"] is False
    assert float(np.mean(np.abs(fallback_rgba.astype(np.float32) - heuristic_rgba.astype(np.float32)))) == 0.0
