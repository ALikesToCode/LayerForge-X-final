from __future__ import annotations

import numpy as np

from layerforge.compose import rgba_from_rgb_alpha
from layerforge.types import Layer
from layerforge.visualize import layer_surface_rgba, save_depth_crop_contact_sheet, save_layer_surface_contact_sheet


def _layer() -> Layer:
    alpha = np.zeros((12, 12), dtype=np.float32)
    alpha[3:9, 3:9] = 1.0
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    rgb[3:9, 3:9] = [200, 80, 40]
    hidden = np.zeros((12, 12), dtype=bool)
    hidden[3:9, 9:11] = True
    completed = rgba_from_rgb_alpha(rgb + np.array([1, 0, 0], dtype=np.uint8), np.maximum(alpha, hidden.astype(np.float32)))
    return Layer(
        id=0,
        name="000_object",
        label="object",
        group="object",
        rank=0,
        depth_median=0.1,
        depth_p10=0.1,
        depth_p90=0.1,
        area=36,
        bbox=(3, 3, 11, 9),
        alpha=alpha,
        rgba=rgba_from_rgb_alpha(rgb, alpha),
        albedo_rgba=rgba_from_rgb_alpha(rgb, alpha),
        shading_rgba=rgba_from_rgb_alpha(rgb, alpha),
        visible_mask=alpha > 0,
        amodal_mask=(alpha > 0) | hidden,
        hidden_mask=hidden,
        completed_rgba=completed,
    )


def test_layer_surface_contact_sheets_cover_expected_surfaces(tmp_path) -> None:
    layer = _layer()
    for surface in ["alpha", "amodal", "hidden", "completed", "albedo", "shading"]:
        rgba = layer_surface_rgba(layer, surface)
        out = save_layer_surface_contact_sheet(tmp_path / f"{surface}.png", [layer], surface)
        assert rgba.shape == layer.rgba.shape
        assert out.exists()
    depth_out = save_depth_crop_contact_sheet(tmp_path / "depth.png", [layer], np.linspace(0, 1, 144, dtype=np.float32).reshape(12, 12))
    assert depth_out.exists()
