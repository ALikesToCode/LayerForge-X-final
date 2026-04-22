from __future__ import annotations

import numpy as np
from PIL import Image

from layerforge.depth import estimate_depth


def test_estimate_depth_preserves_raw_depth_for_metric_and_normalized_output():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(8, dtype=np.uint8)[None, :] * 8
    pil = Image.fromarray(rgb)
    cfg = {
        "method": "geometric_luminance",
        "near_is_smaller": True,
        "flip": False,
        "edge_smooth": False,
    }
    pred = estimate_depth(pil, rgb, cfg, device="cpu")
    assert pred.raw_depth is not None
    assert pred.raw_depth.shape == pred.depth.shape
    assert np.all(pred.depth >= 0.0)
    assert np.all(pred.depth <= 1.0)
    assert np.allclose(pred.raw_depth, pred.depth, atol=5e-3)
