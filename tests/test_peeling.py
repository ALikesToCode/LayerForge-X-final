from __future__ import annotations

import json

import numpy as np
import pytest

from layerforge.peeling import _refine_effect_alpha, extract_associated_effect_layer
from layerforge.pipeline import LayerForgePipeline


def test_extract_associated_effect_layer_detects_shadow_region() -> None:
    current = np.full((48, 48, 3), 220, dtype=np.uint8)
    inpainted = current.copy()
    current[10:24, 16:28] = np.array([230, 90, 70], dtype=np.uint8)
    current[24:34, 14:30] = np.array([150, 150, 150], dtype=np.uint8)
    core_mask = np.zeros((48, 48), dtype=bool)
    core_mask[10:24, 16:28] = True

    layer = extract_associated_effect_layer(
        current_rgb=current,
        inpainted_rgb=inpainted,
        core_mask=core_mask,
        label="person",
        rank=1,
        cfg={
            "enabled": True,
            "dilate_px": 14,
            "inner_dilate_px": 2,
            "delta_threshold": 0.04,
            "alpha_scale": 0.12,
            "min_area_px": 12,
            "prefer_downward": True,
        },
    )

    assert layer is not None
    assert layer.group == "effect"
    assert layer.label == "person effect"
    assert float(layer.alpha[24:34, 14:30].mean()) > 0.05


def test_refine_effect_alpha_can_use_backend_prediction(monkeypatch) -> None:
    residual_alpha = np.zeros((24, 24), dtype=np.float32)
    residual_alpha[10:16, 7:19] = 0.2
    core_mask = np.zeros((24, 24), dtype=bool)
    core_mask[4:10, 8:16] = True
    candidate_mask = np.zeros((24, 24), dtype=bool)
    candidate_mask[10:16, 7:19] = True
    current_rgb = np.full((24, 24, 3), 210, dtype=np.uint8)

    matte_alpha = np.zeros((24, 24), dtype=np.float32)
    matte_alpha[4:10, 8:16] = 1.0
    matte_alpha[10:16, 7:19] = 0.7

    monkeypatch.setattr(
        "layerforge.peeling.predict_alpha_matte",
        lambda *args, **kwargs: (matte_alpha, {"backend": "birefnet", "used": True}),
    )

    alpha, metadata = _refine_effect_alpha(
        current_rgb=current_rgb,
        residual_alpha=residual_alpha,
        core_mask=core_mask,
        candidate_mask=candidate_mask,
        cfg={
            "alpha_backend": "birefnet",
            "alpha_backend_weight": 1.0,
        },
        device="cpu",
    )

    assert float(alpha[10:16, 7:19].mean()) > 0.55
    assert float(alpha[4:10, 8:16].mean()) < 0.05
    assert metadata["backend"] == "birefnet"


@pytest.mark.slow
def test_recursive_peeling_writes_iteration_artifacts(tmp_path) -> None:
    pipe = LayerForgePipeline("configs/fast.yaml")
    outputs = pipe.peel(
        "examples/synth/scene_000/image.png",
        tmp_path / "peel",
        segmenter="classical",
        depth_method="geometric_luminance",
        max_layers=3,
    )

    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))

    assert metrics["mode"] == "recursive_peeling"
    assert (tmp_path / "peel" / "iterations" / "iteration_00" / "input.png").exists()
    assert (tmp_path / "peel" / "iterations" / "iteration_00" / "selected_layer.png").exists()
    assert (tmp_path / "peel" / "iterations" / "iteration_00" / "residual_inpainted.png").exists()
    assert (tmp_path / "peel" / "layers_effects_rgba").exists()
    assert (tmp_path / "peel" / "debug" / "peeling_strip.png").exists()
    assert len(outputs.ordered_layer_paths) >= 2
