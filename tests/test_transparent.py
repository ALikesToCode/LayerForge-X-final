from __future__ import annotations

import numpy as np

from layerforge.transparent import recover_transparent_foreground


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
