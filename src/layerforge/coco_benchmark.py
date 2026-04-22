from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .group_benchmark import BENCHMARK_GROUPS, empty_group_masks, group_for_label, predict_group_masks_for_image, resize_bool_mask
from .utils import write_json


COCO_BENCHMARK_GROUPS = BENCHMARK_GROUPS

_COCO_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "road": ("road", "sidewalk", "pavement", "railroad", "platform", "path", "runway"),
    "ground": ("ground", "floor", "field", "playingfield", "sand", "snow", "dirt", "mud", "gravel", "mountain", "hill"),
    "building": ("building", "house", "wall", "roof", "ceiling", "bridge", "window", "door", "fence", "tent", "stairs"),
    "water": ("water", "river", "sea", "ocean", "lake", "waterfall"),
    "plant": ("plant", "tree", "grass", "flower", "bush", "branch", "leaves", "moss"),
    "stuff": ("banner", "blanket", "curtain", "cloth", "napkin", "towel", "rug", "mat", "cardboard", "paper", "mirror", "net", "textile"),
}


def panoptic_rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB array")
    return arr[..., 0] + 256 * arr[..., 1] + 256 * 256 * arr[..., 2]


def coco_category_to_group(name: str, supercategory: str = "") -> str:
    return group_for_label(f"{name} {supercategory}", _COCO_GROUP_KEYWORDS)


def load_coco_panoptic_metadata(annotation_json: str | Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    with Path(annotation_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = list(data.get("annotations", []))
    images = {int(item["id"]): item for item in data.get("images", [])}
    categories = {int(item["id"]): item for item in data.get("categories", [])}
    return annotations, images, categories


def load_coco_ground_truth_group_masks(
    panoptic_png_path: str | Path,
    segments_info: list[dict[str, Any]],
    categories: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    ids = panoptic_rgb_to_id(np.asarray(Image.open(panoptic_png_path).convert("RGB"), dtype=np.uint8))
    masks = empty_group_masks(ids.shape[:2])
    for segment in segments_info:
        category = categories.get(int(segment["category_id"]), {})
        group = coco_category_to_group(str(category.get("name", "")), str(category.get("supercategory", "")))
        if group not in masks:
            continue
        masks[group] |= ids == int(segment["id"])
    return masks


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / union) if union else 0.0


def run_coco_panoptic_group_benchmark(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "configs/fast.yaml",
    segmenter: str = "mask2former",
    prompts: list[str] | None = None,
    prompt_source: str | None = None,
    device: str = "auto",
    max_images: int | None = None,
    seed: int = 7,
) -> Path:
    dataset_root = Path(dataset_dir)
    images_dir = dataset_root / "val2017"
    annotations_json = dataset_root / "annotations" / "panoptic_val2017.json"
    masks_dir = dataset_root / "annotations" / "panoptic_val2017"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing COCO val images directory: {images_dir}")
    if not annotations_json.exists():
        raise FileNotFoundError(f"Missing COCO panoptic annotation JSON: {annotations_json}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing COCO panoptic masks directory: {masks_dir}")

    annotations, images, categories = load_coco_panoptic_metadata(annotations_json)
    rng = random.Random(int(seed))
    selected = list(annotations)
    rng.shuffle(selected)
    if max_images is not None:
        selected = selected[: max(0, int(max_images))]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    rows: list[dict[str, Any]] = []
    agg = {group: {"intersection": 0, "union": 0, "gt_pixels": 0, "pred_pixels": 0} for group in COCO_BENCHMARK_GROUPS}

    for idx, ann in enumerate(selected, start=1):
        image_info = images[int(ann["image_id"])]
        image_path = images_dir / str(image_info["file_name"])
        panoptic_png = masks_dir / str(ann["file_name"])
        gt_masks = load_coco_ground_truth_group_masks(panoptic_png, list(ann.get("segments_info", [])), categories)
        pred_masks, num_segments = predict_group_masks_for_image(
            image_path,
            cfg,
            device=device,
            segmenter=segmenter,
            prompts=prompts,
            prompt_source=prompt_source,
        )
        row: dict[str, Any] = {
            "image_id": int(ann["image_id"]),
            "file_name": str(image_info["file_name"]),
            "num_pred_segments": num_segments,
        }
        present_ious: list[float] = []
        for group in COCO_BENCHMARK_GROUPS:
            gt = gt_masks[group]
            pred = pred_masks[group]
            if pred.shape != gt.shape:
                pred = resize_bool_mask(pred, gt.shape)
            inter = int(np.logical_and(gt, pred).sum())
            union = int(np.logical_or(gt, pred).sum())
            agg[group]["intersection"] += inter
            agg[group]["union"] += union
            agg[group]["gt_pixels"] += int(gt.sum())
            agg[group]["pred_pixels"] += int(pred.sum())
            iou = float(inter / union) if union else 0.0
            row[f"iou_{group}"] = iou
            if union:
                present_ious.append(iou)
        row["miou_present_groups"] = float(np.mean(present_ious)) if present_ious else 0.0
        rows.append(row)
        if idx % 10 == 0 or idx == len(selected):
            print(f"[coco-panoptic] {idx}/{len(selected)} images", flush=True)

    csv_path = output_root / "coco_panoptic_group_benchmark.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    per_group_iou: dict[str, float] = {}
    valid_groups: list[str] = []
    for group, stats in agg.items():
        union = int(stats["union"])
        if union > 0:
            per_group_iou[group] = float(stats["intersection"] / union)
            valid_groups.append(group)
    thing_groups = [g for g in ["person", "animal", "vehicle", "furniture", "plant", "object"] if g in per_group_iou]
    stuff_groups = [g for g in ["sky", "road", "ground", "building", "water", "stuff"] if g in per_group_iou]
    summary = {
        "dataset": "COCO Panoptic val2017",
        "dataset_dir": str(dataset_root),
        "images_evaluated": len(rows),
        "segmenter": segmenter,
        "prompt_source": prompt_source,
        "prompts": prompts or [],
        "selected_by": "random_sample" if max_images is not None else "full_split",
        "seed": int(seed),
        "group_iou": per_group_iou,
        "miou_supported_groups": float(np.mean([per_group_iou[g] for g in valid_groups])) if valid_groups else 0.0,
        "thing_miou": float(np.mean([per_group_iou[g] for g in thing_groups])) if thing_groups else 0.0,
        "stuff_miou": float(np.mean([per_group_iou[g] for g in stuff_groups])) if stuff_groups else 0.0,
        "mean_image_miou": float(np.mean([row["miou_present_groups"] for row in rows])) if rows else 0.0,
        "median_image_miou": float(np.median([row["miou_present_groups"] for row in rows])) if rows else 0.0,
        "mean_pred_segments": float(np.mean([row["num_pred_segments"] for row in rows])) if rows else 0.0,
        "csv": str(csv_path),
    }
    return write_json(output_root / "coco_panoptic_group_benchmark_summary.json", summary)
