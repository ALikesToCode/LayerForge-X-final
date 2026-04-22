from __future__ import annotations

import importlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


def optional_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"Optional dependency '{name}' is required for this mode") from exc


def transformers_pipeline_device_index(device: str) -> int:
    d = str(device).lower()
    if d == "cpu":
        return -1
    if d == "auto":
        try:
            torch = importlib.import_module("torch")
            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    if d == "cuda":
        return 0
    if d.startswith("cuda:"):
        try:
            return int(d.split(":", 1)[1])
        except Exception:
            return 0
    return -1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return p


def image_to_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind == "f":
        return np.clip(arr.astype(np.float32), 0, 1)
    return arr.astype(np.float32) / 255.0


def float_to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 1).astype(np.float32).round(3) if False else np.clip(x * 255.0, 0, 255).astype(np.uint8)


def normalize01(x: np.ndarray, robust: bool = True) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    if robust:
        lo, hi = np.percentile(arr[finite], [1, 99])
    else:
        lo, hi = float(arr[finite].min()), float(arr[finite].max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float32).ravel()
    order = np.argsort(flat)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0, 1, flat.size, dtype=np.float32)
    return ranks.reshape(x.shape)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def touches_border(mask: np.ndarray, margin: int = 2) -> bool:
    if mask.size == 0:
        return False
    m = mask.astype(bool)
    return bool(m[:margin, :].any() or m[-margin:, :].any() or m[:, :margin].any() or m[:, -margin:].any())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    return float(inter / union) if union else 1.0


def safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(text).lower()).strip("_")
    return s or "layer"
