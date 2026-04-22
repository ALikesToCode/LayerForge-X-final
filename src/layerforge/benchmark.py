from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .pipeline import LayerForgePipeline
from .utils import mask_iou, write_json


def _load_alpha_mask(path: str | Path, shape: tuple[int, int] | None = None, threshold: float = 0.05) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    if shape is not None and im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    arr = np.asarray(im, dtype=np.uint8)
    return arr[..., 3].astype(np.float32) / 255.0 > threshold


def _resolve_gt_path(scene_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    candidate = scene_dir / p.name
    if candidate.exists():
        return candidate
    candidate = scene_dir / "gt_layers" / p.name
    if candidate.exists():
        return candidate
    return scene_dir / raw_path


def load_ground_truth(scene_dir: Path) -> list[dict[str, Any]]:
    gt_path = scene_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground_truth.json in {scene_dir}")
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    layers = list(data.get("layers_near_to_far", []))
    for i, layer in enumerate(layers):
        layer["rank_near_to_far"] = int(layer.get("rank_near_to_far", i))
        layer["path"] = str(_resolve_gt_path(scene_dir, str(layer["path"])))
    return layers


def load_predicted_layers(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    layers = list(manifest.get("ordered_layers_near_to_far", []))
    for i, layer in enumerate(layers):
        layer["rank"] = int(layer.get("rank", i))
    return layers


def match_layers(gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> tuple[list[tuple[int, int, float]], float]:
    if not gt or not pred:
        return [], 0.0
    base_shape = _load_alpha_mask(gt[0]["path"]).shape
    gt_masks = [_load_alpha_mask(x["path"], base_shape) for x in gt]
    pred_masks = [_load_alpha_mask(x["path"], base_shape) for x in pred]
    pairs: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    for gi, gm in enumerate(gt_masks):
        best_j, best_iou = -1, -1.0
        for pj, pm in enumerate(pred_masks):
            if pj in used_pred:
                continue
            val = mask_iou(gm, pm)
            if val > best_iou:
                best_j, best_iou = pj, val
        if best_j >= 0:
            used_pred.add(best_j)
            pairs.append((gi, best_j, float(best_iou)))
    return pairs, float(np.mean([x[2] for x in pairs])) if pairs else 0.0


def pairwise_layer_order_accuracy(gt: list[dict[str, Any]], pred: list[dict[str, Any]], pairs: list[tuple[int, int, float]], min_iou: float = 0.05) -> float:
    good_pairs = [(gi, pj) for gi, pj, iou in pairs if iou >= min_iou]
    if len(good_pairs) < 2:
        return 0.0
    correct = 0
    total = 0
    gt_rank = {i: int(gt[i].get("rank_near_to_far", i)) for i, _ in good_pairs}
    pred_rank = {j: int(pred[j].get("rank", j)) for _, j in good_pairs}
    for a in range(len(good_pairs)):
        for b in range(a + 1, len(good_pairs)):
            gi, pj = good_pairs[a]
            gk, pk = good_pairs[b]
            gt_rel = gt_rank[gi] < gt_rank[gk]
            pred_rel = pred_rank[pj] < pred_rank[pk]
            correct += int(gt_rel == pred_rel)
            total += 1
    return float(correct / total) if total else 0.0


def run_synthetic_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    config_path: str | Path = "configs/fast.yaml",
    segmenter: str = "classical",
    depth: str = "geometric_luminance",
    device: str = "auto",
    max_scenes: int | None = None,
    ordering_method: str | None = None,
    ranker_model_path: str | Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    pipe = LayerForgePipeline(cfg, device=device)
    scenes = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and (p / "image.png").exists()])
    if max_scenes is not None:
        scenes = scenes[: max(0, int(max_scenes))]

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        run_dir = output_dir / "runs" / scene.name
        out = pipe.run(
            scene / "image.png",
            run_dir,
            segmenter=segmenter,
            depth_method=depth,
            save_parallax=False,
            ordering_method=ordering_method,
            ranker_model_path=ranker_model_path,
        )
        gt = load_ground_truth(scene)
        pred = load_predicted_layers(out.manifest_path)
        pairs, miou = match_layers(gt, pred)
        ploa = pairwise_layer_order_accuracy(gt, pred, pairs)
        with out.metrics_path.open("r", encoding="utf-8") as f:
            run_metrics = json.load(f)
        rows.append({
            "scene": scene.name,
            "method_segmenter": segmenter,
            "method_depth": depth,
            "num_gt_layers": len(gt),
            "num_pred_layers": len(pred),
            "matched_layers": len(pairs),
            "mean_best_iou": miou,
            "pairwise_layer_order_accuracy": ploa,
            "recompose_psnr": run_metrics.get("recompose_psnr"),
            "recompose_ssim": run_metrics.get("recompose_ssim"),
            "ordering_method": run_metrics.get("ordering_method"),
            "manifest": str(out.manifest_path),
        })

    csv_path = output_dir / "synthetic_benchmark.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "dataset_dir": str(dataset_dir),
        "num_scenes": len(rows),
        "mean_best_iou": float(np.mean([r["mean_best_iou"] for r in rows])) if rows else 0.0,
        "pairwise_layer_order_accuracy": float(np.mean([r["pairwise_layer_order_accuracy"] for r in rows])) if rows else 0.0,
        "mean_recompose_psnr": float(np.mean([r["recompose_psnr"] for r in rows if r["recompose_psnr"] is not None])) if rows else 0.0,
        "rows": rows,
        "csv": str(csv_path),
    }
    return write_json(output_dir / "synthetic_benchmark_summary.json", summary)
