from __future__ import annotations

import numpy as np

from .utils import optional_import, normalize01


def estimate_alpha(rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray | None, cfg: dict) -> np.ndarray:
    cv2 = optional_import("cv2")
    m = mask.astype(bool)
    if not m.any():
        return np.zeros(mask.shape, dtype=np.float32)
    band = max(1, int(cfg.get("alpha_band_px", 9)))
    inside = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    signed = inside - outside
    alpha = np.clip((signed + band) / (2 * band), 0, 1).astype(np.float32)
    alpha[inside > band] = 1.0
    alpha[outside > band] = 0.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.6, band / 4))
    if depth is not None and bool(cfg.get("preserve_depth_edges", True)):
        gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        edges = normalize01(np.sqrt(gx * gx + gy * gy), robust=True)
        boundary = (inside <= band) & (outside <= band)
        hard = m.astype(np.float32)
        mix = np.clip(edges * 1.35, 0, 1)
        alpha[boundary] = alpha[boundary] * (1 - mix[boundary]) + hard[boundary] * mix[boundary]
    return np.clip(alpha, 0, 1).astype(np.float32)
