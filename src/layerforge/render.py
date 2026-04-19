from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .types import Layer


def shift_rgba(rgba: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(rgba)
    h, w = rgba.shape[:2]
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    out[ys0:ys1, xs0:xs1] = rgba[src_y0:src_y1, src_x0:src_x1]
    return out


def composite_rgba_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    h, w = arrays[0].shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    for rgba in reversed(arrays):
        src = rgba.astype(np.float32) / 255.0
        a = src[..., 3:4]
        rgb = src[..., :3] * a + rgb * (1 - a)
        alpha = a + alpha * (1 - a)
    return np.dstack([np.clip(rgb * 255, 0, 255), np.clip(alpha[..., 0] * 255, 0, 255)]).astype(np.uint8)


def save_parallax_gif(path: str | Path, layers: list[Layer], frames: int = 24, max_pixels: float = 28.0) -> Path:
    if not layers:
        return Path(path)
    ims = []
    ordered = sorted(layers, key=lambda l: l.rank)
    for f in range(frames):
        t = np.sin(2 * np.pi * f / frames)
        arrs = []
        for l in ordered:
            strength = (1.0 - min(1.0, max(0.0, l.depth_median))) * max_pixels
            arrs.append(shift_rgba(l.rgba, int(round(t * strength)), 0))
        ims.append(Image.fromarray(composite_rgba_arrays(arrs), mode="RGBA"))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ims[0].save(p, save_all=True, append_images=ims[1:], duration=70, loop=0)
    return p
