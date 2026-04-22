from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "LayerForge-X", "seed": 7},
    "io": {"max_side": 1280, "save_contact_sheet": True, "save_parallax_gif": True},
    "segmentation": {
        "method": "classical",
        "prompts": [],
        "prompt_source": "manual",
        "min_area_ratio": 0.0018,
        "nms_iou": 0.86,
        "slic_segments": 96,
        "slic_compactness": 12.0,
        "add_background_segment": True,
        "model": {
            "gemini": "gemini-3-flash-preview",
            "gemini_max_layers": 12,
            "gemini_max_side": 1024,
            "gemini_timeout_sec": 300,
        },
    },
    "depth": {"method": "geometric_luminance", "near_is_smaller": True, "flip": False, "edge_smooth": True, "ensemble": [], "model": {}},
    "layering": {
        "min_layer_area_ratio": 0.0015,
        "max_layers": 64,
        "split_stuff_depth_bins": 3,
        "occlusion_boundary_width": 5,
        "occlusion_depth_threshold": 0.025,
        "min_shared_boundary_px": 12,
        "amodal_enabled": True,
        "amodal_expand_px": 16,
        "ordering_method": "boundary",
        "ranker_model_path": "",
        "merge_enabled": True,
        "merge_depth_threshold": 0.04,
        "merge_color_threshold": 0.17,
        "merge_spatial_gap_px": 20,
    },
    "matting": {"alpha_band_px": 9, "preserve_depth_edges": True},
    "inpainting": {"method": "opencv_telea", "radius": 5},
    "intrinsics": {"method": "retinex", "sigma": 28.0, "external_command": ""},
    "qwen": {
        "min_alpha": 0.02,
        "preserve_external_order": False,
        "merge_external_layers": False,
    },
    "peeling": {
        "max_layers": 6,
        "min_remaining_foreground_ratio": 0.001,
    },
    "effects": {
        "enabled": True,
        "dilate_px": 22,
        "inner_dilate_px": 4,
        "support_dilate_px": 12,
        "support_inpaint_radius": 5,
        "use_provided_reference": False,
        "delta_threshold": 0.05,
        "alpha_scale": 0.18,
        "min_area_px": 80,
        "prefer_downward": True,
    },
    "self_eval": {
        "weights": {
            "recomposition_fidelity": 0.20,
            "edit_preservation": 0.25,
            "semantic_separation": 0.20,
            "alpha_quality": 0.10,
            "graph_confidence": 0.15,
            "runtime": 0.10,
        }
    },
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
