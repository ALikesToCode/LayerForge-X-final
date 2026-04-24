from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .compose import composite_layers_near_to_far
from .types import Layer


def _safe_structural_similarity(rgb: np.ndarray, comp: np.ndarray) -> float:
    min_side = min(int(rgb.shape[0]), int(rgb.shape[1]))
    if min_side < 3:
        return 1.0 if np.array_equal(rgb, comp) else 0.0
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    return float(structural_similarity(rgb, comp, channel_axis=2, data_range=255, win_size=win_size))


def compute_run_metrics(rgb: np.ndarray, layers: list[Layer], cfg: dict) -> dict[str, float]:
    comp = composite_layers_near_to_far(layers)[..., :3]
    return {
        "num_layers": float(len(layers)),
        "foreground_alpha_coverage": float(np.mean(np.maximum.reduce([l.alpha for l in layers]) > 0.05)) if layers else 0.0,
        "recompose_psnr": float(peak_signal_noise_ratio(rgb, comp, data_range=255)),
        "recompose_ssim": _safe_structural_similarity(rgb, comp),
        "mean_layer_area": float(np.mean([l.area for l in layers])) if layers else 0.0,
        "mean_amodal_extra_ratio": float(np.mean([(max(0, int(l.amodal_mask.sum()) - l.area) / max(1, l.area)) for l in layers if l.amodal_mask is not None])) if any(l.amodal_mask is not None for l in layers) else 0.0,
    }
