from __future__ import annotations

import numpy as np

from layerforge.inpaint import inpaint_background


def test_lama_mode_falls_back_to_opencv_when_dependency_missing(monkeypatch) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 120
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    def fail_loader():
        raise ModuleNotFoundError("simple_lama_inpainting")

    monkeypatch.setattr("layerforge.inpaint._load_simple_lama", fail_loader)

    out, used_mask, method = inpaint_background(rgb, mask, {"method": "lama", "radius": 3})

    assert out.shape == rgb.shape
    assert used_mask.shape == mask.shape
    assert method == "opencv_telea_fallback"
