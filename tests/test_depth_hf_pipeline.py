from __future__ import annotations

import numpy as np
from PIL import Image

from layerforge import depth as depth_mod


def test_hf_depth_estimation_accepts_array_output_without_boolean_coercion(monkeypatch) -> None:
    def fake_pipe(_pil):
        return {"predicted_depth": np.ones((4, 4), dtype=np.float32)}

    monkeypatch.setattr(depth_mod, "_load_hf_depth_pipeline", lambda model_name, device_index: fake_pipe)
    pil = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
    pred = depth_mod.hf_depth_estimation(
        pil,
        {"model": {"depth_anything_v2": "fake/model"}},
        "depth_anything_v2",
        "depth_anything_v2",
        device="cpu",
    )
    assert pred.depth.shape == (4, 4)
    assert pred.raw_depth is not None
