from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .semantic import BACKGROUND_GROUPS


FEATURE_NAMES = [
    "delta_depth_median",
    "delta_depth_p10",
    "delta_depth_p90",
    "signed_boundary_gap",
    "has_boundary_contact",
    "log_shared_boundary",
    "delta_log_area",
    "delta_centroid_y",
    "delta_centroid_x",
    "bbox_overlap_ratio",
    "delta_border_touch",
    "delta_background_group",
]


@dataclass(slots=True)
class OrderRanker:
    feature_names: list[str]
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    training_metrics: dict[str, float]

    def predict_logit(self, features: np.ndarray) -> float:
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        return float(x @ self.weights + self.bias)

    def predict_probability(self, features: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(self.predict_logit(features), -30.0, 30.0))))

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "weights": [float(x) for x in self.weights],
            "bias": float(self.bias),
            "mean": [float(x) for x in self.mean],
            "scale": [float(x) for x in self.scale],
            "training_metrics": {k: float(v) for k, v in self.training_metrics.items()},
        }


def save_ranker(model: OrderRanker, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(model.to_json(), f, indent=2, sort_keys=True)
    return p


def load_ranker(path: str | Path) -> OrderRanker:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return OrderRanker(
        feature_names=list(data["feature_names"]),
        weights=np.asarray(data["weights"], dtype=np.float32),
        bias=float(data["bias"]),
        mean=np.asarray(data["mean"], dtype=np.float32),
        scale=np.asarray(data["scale"], dtype=np.float32),
        training_metrics={str(k): float(v) for k, v in dict(data.get("training_metrics", {})).items()},
    )


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return 0.5, 0.5
    h, w = mask.shape
    return float(ys.mean() / max(1, h - 1)), float(xs.mean() / max(1, w - 1))


def _bbox_overlap_ratio(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int], a_area: int, b_area: int) -> float:
    ax0, ay0, ax1, ay1 = a_bbox
    bx0, by0, bx1, by1 = b_bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    denom = max(1, min(a_area, b_area))
    return float(inter / denom)


def pairwise_feature_vector(node_a: Any, node_b: Any) -> np.ndarray:
    a_seg = node_a.segment
    b_seg = node_b.segment
    a_cy, a_cx = _mask_centroid(a_seg.mask)
    b_cy, b_cx = _mask_centroid(b_seg.mask)

    edge = node_a.outgoing_edges.get(b_seg.id) or node_b.outgoing_edges.get(a_seg.id)
    signed_gap = 0.0
    boundary_len = 0
    if edge is not None:
        boundary_len = int(edge.shared_boundary_length)
        signed_gap = float(edge.local_depth_gap if edge.near_id == a_seg.id else -edge.local_depth_gap)

    return np.asarray(
        [
            float(node_a.depth_median - node_b.depth_median),
            float(node_a.depth_p10 - node_b.depth_p10),
            float(node_a.depth_p90 - node_b.depth_p90),
            float(signed_gap),
            float(boundary_len > 0),
            float(np.log1p(boundary_len)),
            float(np.log1p(max(1, a_seg.area)) - np.log1p(max(1, b_seg.area))),
            float(a_cy - b_cy),
            float(a_cx - b_cx),
            float(_bbox_overlap_ratio(a_seg.bbox, b_seg.bbox, a_seg.area, b_seg.area)),
            float(node_a.border_touch) - float(node_b.border_touch),
            float(a_seg.group in BACKGROUND_GROUPS) - float(b_seg.group in BACKGROUND_GROUPS),
        ],
        dtype=np.float32,
    )


def predict_pair_probability(model: OrderRanker, node_a: Any, node_b: Any) -> float:
    return model.predict_probability(pairwise_feature_vector(node_a, node_b))


def learned_order(nodes: dict[int, Any], model: OrderRanker) -> tuple[list[int], dict[int, float]]:
    ids = list(nodes)
    if len(ids) <= 1:
        return ids, {sid: 0.0 for sid in ids}
    scores = {sid: 0.0 for sid in ids}
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            p = predict_pair_probability(model, nodes[a_id], nodes[b_id])
            scores[a_id] += p
            scores[b_id] += 1.0 - p
    order = sorted(ids, key=lambda sid: (-scores[sid], nodes[sid].depth_median, nodes[sid].segment.area))
    return order, scores


def _resolve_gt_layers(scene_dir: Path) -> list[dict[str, Any]]:
    gt_path = scene_dir / "ground_truth.json"
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    layers = []
    for idx, layer in enumerate(data.get("layers_near_to_far", [])):
        rel_path = Path(str(layer["path"]))
        path = rel_path if rel_path.is_absolute() else scene_dir / rel_path
        layers.append({"path": path, "rank": int(layer.get("rank_near_to_far", idx))})
    return layers


def _load_alpha_mask(path: Path, shape: tuple[int, int], threshold: float = 0.05) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    alpha = np.asarray(im, dtype=np.uint8)[..., 3].astype(np.float32) / 255.0
    return alpha > threshold


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    return float(inter / union) if union else 0.0


def _match_pred_segments_to_gt(pred_segments: list[Any], gt_layers: list[dict[str, Any]], shape: tuple[int, int], min_iou: float) -> list[tuple[int, int, float]]:
    gt_masks = [_load_alpha_mask(Path(layer["path"]), shape) for layer in gt_layers]
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    for pred in pred_segments:
        best_gt = -1
        best_iou = -1.0
        for gi, gt_mask in enumerate(gt_masks):
            if gi in used_gt:
                continue
            iou = _mask_iou(pred.mask, gt_mask)
            if iou > best_iou:
                best_gt = gi
                best_iou = iou
        if best_gt >= 0 and best_iou >= min_iou:
            used_gt.add(best_gt)
            matches.append((pred.id, best_gt, float(best_iou)))
    return matches


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return ((x - mean) / scale).astype(np.float32), mean, scale


def _fit_logistic_regression(x: np.ndarray, y: np.ndarray, *, steps: int = 800, lr: float = 0.2, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    w = np.zeros(x.shape[1], dtype=np.float32)
    b = 0.0
    y = y.astype(np.float32)
    n = float(len(y))
    for _ in range(max(50, steps)):
        logits = x @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        err = probs - y
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w.astype(np.float32)
        b -= lr * grad_b
    return w.astype(np.float32), float(b)


def train_synthetic_order_ranker(
    dataset_dir: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = "configs/fast.yaml",
    segmenter: str = "classical",
    depth: str = "geometric_luminance",
    device: str = "auto",
    max_scenes: int | None = None,
    min_match_iou: float = 0.05,
) -> dict[str, float | int | str]:
    from .depth import estimate_depth
    from .graph import build_nodes
    from .image_io import load_rgb
    from .segment import resolve_disjoint_masks, segment_image

    cfg = load_config(config_path)
    cfg["segmentation"]["method"] = segmenter
    cfg["depth"]["method"] = depth

    scenes = sorted([p for p in Path(dataset_dir).iterdir() if p.is_dir() and (p / "image.png").exists()])
    if max_scenes is not None:
        scenes = scenes[: max(0, int(max_scenes))]

    features: list[np.ndarray] = []
    labels: list[int] = []
    used_scenes = 0
    for scene_dir in scenes:
        rgb, pil = load_rgb(scene_dir / "image.png", cfg.get("io", {}).get("max_side"))
        raw_segments = segment_image(rgb, pil, cfg["segmentation"], device)
        depth_pred = estimate_depth(pil, rgb, cfg["depth"], device)
        segments = resolve_disjoint_masks(raw_segments, depth_pred.depth.astype(np.float32))
        nodes = build_nodes(segments, depth_pred.depth.astype(np.float32), cfg["layering"])
        gt_layers = _resolve_gt_layers(scene_dir)
        matches = _match_pred_segments_to_gt(segments, gt_layers, depth_pred.depth.shape, min_match_iou)
        if len(matches) < 2:
            continue
        gt_rank_by_pred = {pred_id: gt_layers[gt_idx]["rank"] for pred_id, gt_idx, _ in matches}
        matched_ids = sorted(gt_rank_by_pred)
        for i, a_id in enumerate(matched_ids):
            for b_id in matched_ids[i + 1:]:
                label = int(gt_rank_by_pred[a_id] < gt_rank_by_pred[b_id])
                feat = pairwise_feature_vector(nodes[a_id], nodes[b_id])
                features.append(feat)
                labels.append(label)
                features.append(-feat)
                labels.append(1 - label)
        used_scenes += 1

    if not features:
        raise RuntimeError(f"No training pairs were produced from {dataset_dir}")

    x = np.stack(features, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    x_norm, mean, scale = _standardize(x)
    weights, bias = _fit_logistic_regression(x_norm, y)
    probs = 1.0 / (1.0 + np.exp(-np.clip(x_norm @ weights + bias, -30.0, 30.0)))
    preds = (probs >= 0.5).astype(np.float32)
    accuracy = float((preds == y).mean())

    model = OrderRanker(
        feature_names=list(FEATURE_NAMES),
        weights=weights,
        bias=bias,
        mean=mean,
        scale=scale,
        training_metrics={
            "pair_count": float(len(y)),
            "scene_count": float(used_scenes),
            "train_accuracy": accuracy,
            "positive_rate": float(y.mean()),
        },
    )
    save_ranker(model, output_path)
    return {
        "output_path": str(output_path),
        "scene_count": used_scenes,
        "pair_count": int(len(y)),
        "train_accuracy": accuracy,
        "positive_rate": float(y.mean()),
    }
