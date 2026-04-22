from __future__ import annotations

import numpy as np

from layerforge.depth_benchmark import align_depth_prediction, build_valid_depth_mask, compute_depth_metrics


def test_align_depth_prediction_recovers_scale_factor() -> None:
    gt = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    pred = gt / 2.0
    mask = np.ones_like(gt, dtype=bool)
    aligned, meta = align_depth_prediction(pred, gt, mask, mode="scale")
    assert np.allclose(aligned, gt, atol=1e-6)
    assert abs(meta["scale"] - 2.0) < 1e-6


def test_align_depth_prediction_recovers_scale_and_shift() -> None:
    gt = np.array([[3.0, 5.0], [7.0, 9.0]], dtype=np.float32)
    pred = (gt - 1.5) / 2.5
    mask = np.ones_like(gt, dtype=bool)
    aligned, meta = align_depth_prediction(pred, gt, mask, mode="scale_shift")
    assert np.allclose(aligned, gt, atol=1e-5)
    assert abs(meta["scale"] - 2.5) < 1e-5
    assert abs(meta["shift"] - 1.5) < 1e-5


def test_compute_depth_metrics_matches_perfect_prediction() -> None:
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = gt.copy()
    mask = np.ones_like(gt, dtype=bool)
    metrics = compute_depth_metrics(gt, pred, mask)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["abs_rel"] == 0.0
    assert metrics["delta1"] == 1.0
    assert metrics["delta2"] == 1.0
    assert metrics["delta3"] == 1.0


def test_build_valid_depth_mask_applies_positive_depth_and_external_mask() -> None:
    gt = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    validity = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    mask = build_valid_depth_mask(gt, validity)
    assert mask.tolist() == [[False, True], [False, False]]
