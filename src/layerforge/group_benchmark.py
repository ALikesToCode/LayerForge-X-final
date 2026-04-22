from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .image_io import load_rgb
from .segment import segment_image
from .semantic import label_to_group


BENCHMARK_GROUPS = [
    "person",
    "animal",
    "vehicle",
    "furniture",
    "plant",
    "sky",
    "road",
    "ground",
    "building",
    "water",
    "stuff",
    "object",
]


def group_for_label(label: str, extra_keywords: dict[str, tuple[str, ...]] | None = None) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    if extra_keywords:
        for group, keys in extra_keywords.items():
            if any(key in text for key in keys):
                return group
    return label_to_group(text)


def empty_group_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {group: np.zeros(shape, dtype=bool) for group in BENCHMARK_GROUPS}


def resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = Image.fromarray(mask.astype(np.uint8) * 255)
    arr = arr.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(arr, dtype=np.uint8) > 127


def predict_group_masks_for_image(
    image_path: str | Path,
    cfg: dict[str, Any] | str | Path,
    *,
    device: str = "auto",
    segmenter: str | None = None,
    prompts: list[str] | None = None,
    prompt_source: str | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    full_cfg = load_config(cfg) if isinstance(cfg, (str, Path)) else load_config(None, cfg)
    seg_cfg = json.loads(json.dumps(full_cfg["segmentation"]))
    seg_cfg["add_background_segment"] = False
    if segmenter:
        seg_cfg["method"] = segmenter
    if prompts:
        seg_cfg["prompts"] = prompts
    if prompt_source:
        seg_cfg["prompt_source"] = prompt_source
    rgb, pil = load_rgb(image_path, full_cfg.get("io", {}).get("max_side"))
    segments = segment_image(rgb, pil, seg_cfg, device)
    masks = empty_group_masks(rgb.shape[:2])
    for seg in segments:
        group = str(seg.group)
        if group not in masks:
            group = "object"
        masks[group] |= seg.mask.astype(bool)
    return masks, len(segments)
