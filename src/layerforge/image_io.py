from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .utils import ensure_dir, normalize01


def load_rgb(path: str | Path, max_side: int | None = None) -> tuple[np.ndarray, Image.Image]:
    pil = Image.open(path).convert("RGB")
    if max_side and max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        new_size = (max(1, int(round(pil.width * scale))), max(1, int(round(pil.height * scale))))
        pil = pil.resize(new_size, Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8), pil


def save_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(p)
    return p


def save_rgba(path: str | Path, rgba: np.ndarray) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), mode="RGBA").save(p)
    return p


def save_gray(path: str | Path, gray: np.ndarray) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    arr = normalize01(gray, robust=False)
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(p)
    return p


def save_depth16(path: str | Path, depth: np.ndarray) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    arr = (normalize01(depth, robust=False) * 65535).astype(np.uint16)
    Image.fromarray(arr, mode="I;16").save(p)
    return p
