from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "LayerForge-X", "seed": 7},
    "io": {"max_side": 1280, "save_contact_sheet": True, "save_parallax_gif": True},
    "segmentation": {"method": "classical", "prompts": [], "min_area_ratio": 0.0018, "nms_iou": 0.86, "slic_segments": 96, "slic_compactness": 12.0, "add_background_segment": True, "model": {}},
    "depth": {"method": "geometric_luminance", "near_is_smaller": True, "flip": False, "edge_smooth": True, "ensemble": [], "model": {}},
    "layering": {"min_layer_area_ratio": 0.0015, "max_layers": 64, "split_stuff_depth_bins": 3, "occlusion_boundary_width": 5, "occlusion_depth_threshold": 0.025, "min_shared_boundary_px": 12, "amodal_enabled": True, "amodal_expand_px": 16},
    "matting": {"alpha_band_px": 9, "preserve_depth_edges": True},
    "inpainting": {"method": "opencv_telea", "radius": 5},
    "intrinsics": {"method": "retinex", "sigma": 28.0, "external_command": ""},
    "qwen": {"min_alpha": 0.02},
    "render": {"parallax_frames": 24, "parallax_pixels": 28},
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, loaded)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg
