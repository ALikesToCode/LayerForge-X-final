from __future__ import annotations

import numpy as np

from layerforge.transparent import _refine_transparent_alpha, recover_transparent_foreground


def test_recover_transparent_foreground_recovers_known_overlay() -> None:
    background = np.full((16, 16, 3), 120, dtype=np.uint8)
    foreground = np.full((16, 16, 3), [240, 180, 60], dtype=np.uint8)
    alpha = np.zeros((16, 16), dtype=np.float32)
    alpha[4:12, 4:12] = 0.5
    composed = np.clip(foreground.astype(np.float32) * alpha[..., None] + background.astype(np.float32) * (1.0 - alpha[..., None]), 0, 255).astype(np.uint8)

    rgba = recover_transparent_foreground(composed, background, alpha)
    recovered = rgba[..., :3]

    target_patch = recovered[4:12, 4:12].astype(np.float32)
    assert float(np.mean(np.abs(target_patch - foreground[4:12, 4:12].astype(np.float32)))) < 2.0
    assert float(rgba[..., 3].astype(np.float32)[4:12, 4:12].mean()) > 120.0


def test_refine_transparent_alpha_can_use_backend_prediction(monkeypatch) -> None:
    input_rgb = np.full((16, 16, 3), 140, dtype=np.uint8)
    background_rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    base_alpha = np.zeros((16, 16), dtype=np.float32)
    base_alpha[4:12, 4:12] = 0.15
    support_mask = np.zeros((16, 16), dtype=bool)
    support_mask[4:12, 4:12] = True

    backend_alpha = np.zeros((16, 16), dtype=np.float32)
    backend_alpha[4:12, 4:12] = 0.85

    monkeypatch.setattr(
        "layerforge.transparent.predict_alpha_matte",
        lambda *args, **kwargs: (backend_alpha, {"backend": "birefnet", "used": True}),
    )

    alpha, metadata = _refine_transparent_alpha(
        input_rgb,
        background_rgb,
        base_alpha,
        support_mask,
        cfg={
            "residual_alpha_scale": 0.18,
            "base_alpha_weight": 0.35,
            "alpha_blur_radius": 0,
            "backend_blend_weight": 1.0,
            "backend": "birefnet",
        },
        device="cpu",
    )

    assert float(alpha[4:12, 4:12].mean()) > 0.75
    assert metadata["backend"] == "birefnet"
