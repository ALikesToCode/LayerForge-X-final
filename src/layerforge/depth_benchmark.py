from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


DEPTH_EPS = 1e-6


def squeeze_depth_map(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {arr.shape}")
    return arr


def resize_depth_map(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = squeeze_depth_map(depth)
    if arr.shape == shape:
        return arr.astype(np.float32, copy=False)
    resized = Image.fromarray(arr.astype(np.float32)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.shape == shape:
        return arr.astype(bool, copy=False)
    resized = Image.fromarray((arr.astype(np.uint8) * 255), mode="L").resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def build_valid_depth_mask(gt_depth: np.ndarray, validity_mask: np.ndarray | None = None, min_depth: float = DEPTH_EPS) -> np.ndarray:
    gt = squeeze_depth_map(gt_depth)
    valid = np.isfinite(gt) & (gt > float(min_depth))
    if validity_mask is not None:
        valid &= resize_mask(validity_mask, gt.shape)
    return valid


def align_depth_prediction(pred_depth: np.ndarray, gt_depth: np.ndarray, valid_mask: np.ndarray, mode: str = "none") -> tuple[np.ndarray, dict[str, float]]:
    pred = squeeze_depth_map(pred_depth).astype(np.float64, copy=False)
    gt = squeeze_depth_map(gt_depth).astype(np.float64, copy=False)
    mask = np.asarray(valid_mask, dtype=bool)
    if pred.shape != gt.shape or mask.shape != gt.shape:
        raise ValueError("Prediction, ground truth, and valid mask must have the same shape")
    if not mask.any():
        return np.clip(pred, DEPTH_EPS, None).astype(np.float32), {}

    mode = str(mode).lower()
    meta: dict[str, float] = {}
    if mode == "none":
        aligned = pred
    elif mode == "scale":
        x = pred[mask]
        y = gt[mask]
        denom = float(np.dot(x, x))
        scale = float(np.dot(x, y) / denom) if denom > DEPTH_EPS else 1.0
        aligned = pred * scale
        meta["scale"] = scale
    elif mode in {"scale_shift", "affine"}:
        x = pred[mask]
        y = gt[mask]
        design = np.stack([x, np.ones_like(x)], axis=1)
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        scale = float(solution[0])
        shift = float(solution[1])
        aligned = pred * scale + shift
        meta["scale"] = scale
        meta["shift"] = shift
    else:
        raise ValueError(f"Unknown depth alignment mode: {mode}")
    return np.clip(aligned, DEPTH_EPS, None).astype(np.float32), meta


def compute_depth_metrics(gt_depth: np.ndarray, pred_depth: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    gt = squeeze_depth_map(gt_depth).astype(np.float64, copy=False)
    pred = squeeze_depth_map(pred_depth).astype(np.float64, copy=False)
    mask = np.asarray(valid_mask, dtype=bool)
    if gt.shape != pred.shape or mask.shape != gt.shape:
        raise ValueError("Prediction, ground truth, and valid mask must have the same shape")
    if not mask.any():
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "abs_rel": 0.0,
            "sq_rel": 0.0,
            "log_mae": 0.0,
            "log_rmse": 0.0,
            "rmse_log": 0.0,
            "silog": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
        }

    gt_eval = gt[mask]
    pred_eval = np.clip(pred[mask], DEPTH_EPS, None)
    abs_diff = np.abs(pred_eval - gt_eval)
    sq_diff = np.square(pred_eval - gt_eval)
    log10_diff = np.log10(pred_eval) - np.log10(gt_eval)
    log_diff = np.log(pred_eval) - np.log(gt_eval)
    ratio = np.maximum(gt_eval / pred_eval, pred_eval / gt_eval)
    silog_sq = float(np.mean(np.square(log_diff)) - np.square(np.mean(log_diff)))

    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(sq_diff))),
        "abs_rel": float(np.mean(abs_diff / gt_eval)),
        "sq_rel": float(np.mean(sq_diff / gt_eval)),
        "log_mae": float(np.mean(np.abs(log10_diff))),
        "log_rmse": float(np.sqrt(np.mean(np.square(log10_diff)))),
        "rmse_log": float(np.sqrt(np.mean(np.square(log_diff)))),
        "silog": float(np.sqrt(max(silog_sq, 0.0)) * 100.0),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def summarize_depth_rows(rows: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in metric_keys}
    return {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in metric_keys
    }
