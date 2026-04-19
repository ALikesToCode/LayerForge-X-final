from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .compose import composite_layers_near_to_far
from .types import Layer


def compute_run_metrics(rgb: np.ndarray, layers: list[Layer], cfg: dict) -> dict[str, float]:
    comp = composite_layers_near_to_far(layers)[..., :3]
    return {
        "num_layers": float(len(layers)),
        "foreground_alpha_coverage": float(np.mean(np.maximum.reduce([l.alpha for l in layers]) > 0.05)) if layers else 0.0,
        "recompose_psnr": float(peak_signal_noise_ratio(rgb, comp, data_range=255)),
        "recompose_ssim": float(structural_similarity(rgb, comp, channel_axis=2, data_range=255)),
        "mean_layer_area": float(np.mean([l.area for l in layers])) if layers else 0.0,
        "mean_amodal_extra_ratio": float(np.mean([(max(0, int(l.amodal_mask.sum()) - l.area) / max(1, l.area)) for l in layers if l.amodal_mask is not None])) if any(l.amodal_mask is not None for l in layers) else 0.0,
    }
