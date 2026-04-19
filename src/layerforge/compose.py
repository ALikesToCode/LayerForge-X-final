from __future__ import annotations

import numpy as np

from .utils import image_to_float


def rgba_from_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return a standard straight-alpha RGBA image.

    PNG viewers and editing tools expect RGB to store layer color and A to store
    opacity. Compositing functions multiply RGB by alpha at render time.
    """
    a = np.clip(alpha.astype(np.float32), 0, 1)
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb.astype(np.float32), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
    return out


def composite_layers_near_to_far(layers) -> np.ndarray:
    if not layers:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    h, w = layers[0].rgba.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    a_out = np.zeros((h, w, 1), dtype=np.float32)
    # Draw far -> near. Layer ranks are near -> far.
    for layer in sorted(layers, key=lambda l: l.rank, reverse=True):
        rgba = image_to_float(layer.rgba)
        src_rgb = rgba[..., :3]
        src_a = rgba[..., 3:4]
        rgb = src_rgb * src_a + rgb * (1 - src_a)
        a_out = src_a + a_out * (1 - src_a)
    return np.dstack([np.clip(rgb * 255, 0, 255), np.clip(a_out[..., 0] * 255, 0, 255)]).astype(np.uint8)
