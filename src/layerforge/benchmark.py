from __future__ import annotations

import csv
import json
import time
import resource
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .pipeline import LayerForgePipeline
from .utils import mask_iou, write_json


CI_METRIC_GATES: dict[str, float] = {
    "semantic_segmentation_quality": 0.0,
    "alpha_matting_quality": 0.0,
    "depth_ordering_quality": 0.0,
    "amodal_mask_quality": 0.0,
    "inpainting_quality": 0.0,
    "intrinsic_decomposition_quality": 0.0,
    "recomposition_quality": 0.0,
}


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


def summarize_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str, default: float = 0.0) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else default

    recomposition_ssim = mean_of("recompose_ssim")
    intrinsic_residual = mean_of("mean_intrinsic_residual")
    summary = {
        "num_scenes": len(rows),
        "semantic_segmentation_quality": mean_of("mean_best_iou"),
        "alpha_matting_quality": mean_of("mean_alpha_quality_score", default=1.0),
        "depth_ordering_quality": mean_of("pairwise_layer_order_accuracy"),
        "amodal_mask_quality": 1.0 / (1.0 + max(0.0, mean_of("mean_hidden_area_ratio"))),
        "inpainting_quality": mean_of("mean_completion_consistency", default=1.0),
        "intrinsic_decomposition_quality": float(np.clip(1.0 - intrinsic_residual, 0.0, 1.0)),
        "recomposition_quality": recomposition_ssim,
        "mean_runtime_sec": mean_of("runtime_sec"),
        "peak_memory_mb": mean_of("peak_memory_mb"),
    }
    summary["ci_gates"] = {
        key: {"threshold": threshold, "value": float(summary.get(key, 0.0)), "passed": float(summary.get(key, 0.0)) >= threshold}
        for key, threshold in CI_METRIC_GATES.items()
    }
    summary["ci_passed"] = all(row["passed"] for row in summary["ci_gates"].values())
    return summary


def write_markdown_benchmark_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    path = output_dir / "synthetic_benchmark_summary.md"
    rows = [
        "# LayerForge Synthetic Benchmark",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "semantic_segmentation_quality",
        "alpha_matting_quality",
        "depth_ordering_quality",
        "amodal_mask_quality",
        "inpainting_quality",
        "intrinsic_decomposition_quality",
        "recomposition_quality",
        "mean_runtime_sec",
        "peak_memory_mb",
    ]:
        rows.append(f"| {key} | {float(summary.get(key, 0.0)):.6f} |")
    rows.extend(["", "## CI Gates", "", "| Gate | Value | Threshold | Passed |", "| --- | ---: | ---: | :---: |"])
    for key, gate in summary.get("ci_gates", {}).items():
        rows.append(f"| {key} | {float(gate['value']):.6f} | {float(gate['threshold']):.6f} | {bool(gate['passed'])} |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


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
        start = time.perf_counter()
        out = pipe.run(
            scene / "image.png",
            run_dir,
            segmenter=segmenter,
            depth_method=depth,
            save_parallax=False,
            ordering_method=ordering_method,
            ranker_model_path=ranker_model_path,
        )
        runtime_sec = time.perf_counter() - start
        peak_memory_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
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
            "mean_alpha_quality_score": run_metrics.get("mean_alpha_quality_score", 1.0),
            "mean_hidden_area_ratio": run_metrics.get("mean_hidden_area_ratio", 0.0),
            "mean_completion_consistency": run_metrics.get("mean_completion_consistency", 1.0),
            "mean_intrinsic_residual": run_metrics.get("mean_intrinsic_residual", 0.0),
            "runtime_sec": runtime_sec,
            "peak_memory_mb": peak_memory_mb,
            "ordering_method": run_metrics.get("ordering_method"),
            "manifest": str(out.manifest_path),
        })

    csv_path = output_dir / "synthetic_benchmark.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metric_summary = summarize_benchmark_rows(rows)
    summary = {
        "dataset_dir": str(dataset_dir),
        "num_scenes": metric_summary["num_scenes"],
        "mean_best_iou": float(np.mean([r["mean_best_iou"] for r in rows])) if rows else 0.0,
        "pairwise_layer_order_accuracy": float(np.mean([r["pairwise_layer_order_accuracy"] for r in rows])) if rows else 0.0,
        "mean_recompose_psnr": float(np.mean([r["recompose_psnr"] for r in rows if r["recompose_psnr"] is not None])) if rows else 0.0,
        **metric_summary,
        "rows": rows,
        "csv": str(csv_path),
    }
    summary["markdown"] = str(write_markdown_benchmark_report(output_dir, summary))
    return write_json(output_dir / "synthetic_benchmark_summary.json", summary)
