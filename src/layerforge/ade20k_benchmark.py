from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .group_benchmark import BENCHMARK_GROUPS, empty_group_masks, group_for_label, predict_group_masks_for_image, resize_bool_mask
from .utils import write_json


ADE20K_BENCHMARK_GROUPS = BENCHMARK_GROUPS

_ADE_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "road": ("sidewalk", "pavement", "runway"),
    "ground": ("floor", "flooring", "earth", "ground", "field", "sand", "snow", "mountain"),
    "building": ("wall", "ceiling", "edifice", "windowpane", "skyscraper"),
    "water": ("water", "river", "sea", "ocean", "lake", "waterfall"),
    "plant": ("plant", "tree", "grass", "flower", "palm"),
    "stuff": ("curtain", "rug", "blanket", "screen", "mirror"),
}


def resolve_ade20k_root(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir)
    if (root / "images" / "validation").exists():
        return root
    nested = root / "ADEChallengeData2016"
    if (nested / "images" / "validation").exists():
        return nested
    raise FileNotFoundError(f"Could not locate ADEChallengeData2016 under {root}")


def load_ade20k_category_names(object_info_path: str | Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    with Path(object_info_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(str(row.get("Idx", "")).strip())
            name = str(row.get("Name", "")).split(",")[0].strip().lower()
            if idx > 0 and name:
                categories[idx] = name
    if not categories:
        raise ValueError(f"No ADE20K categories parsed from {object_info_path}")
    return categories


def ade20k_category_to_group(name: str) -> str:
    return group_for_label(name, _ADE_GROUP_KEYWORDS)


def load_ade20k_ground_truth_group_masks(
    annotation_png_path: str | Path,
    categories: dict[int, str],
) -> dict[str, np.ndarray]:
    arr = np.asarray(Image.open(annotation_png_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int32)
    masks = empty_group_masks(arr.shape[:2])
    present_ids = np.unique(arr)
    for category_id in present_ids.tolist():
        if category_id <= 0:
            continue
        label = categories.get(int(category_id))
        if not label:
            continue
        group = ade20k_category_to_group(label)
        if group not in masks:
            continue
        masks[group] |= arr == int(category_id)
    return masks


def run_ade20k_group_benchmark(
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
    dataset_root = resolve_ade20k_root(dataset_dir)
    images_dir = dataset_root / "images" / "validation"
    annotations_dir = dataset_root / "annotations" / "validation"
    object_info_path = dataset_root / "objectInfo150.txt"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing ADE20K validation images directory: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing ADE20K validation annotations directory: {annotations_dir}")
    if not object_info_path.exists():
        raise FileNotFoundError(f"Missing ADE20K object info file: {object_info_path}")

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    rng = random.Random(int(seed))
    rng.shuffle(image_paths)
    if max_images is not None:
        image_paths = image_paths[: max(0, int(max_images))]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    categories = load_ade20k_category_names(object_info_path)
    rows: list[dict[str, Any]] = []
    agg = {group: {"intersection": 0, "union": 0, "gt_pixels": 0, "pred_pixels": 0} for group in ADE20K_BENCHMARK_GROUPS}

    for idx, image_path in enumerate(image_paths, start=1):
        annotation_path = annotations_dir / f"{image_path.stem}.png"
        if not annotation_path.exists():
            continue
        gt_masks = load_ade20k_ground_truth_group_masks(annotation_path, categories)
        pred_masks, num_segments = predict_group_masks_for_image(
            image_path,
            cfg,
            device=device,
            segmenter=segmenter,
            prompts=prompts,
            prompt_source=prompt_source,
        )
        row: dict[str, Any] = {"file_name": image_path.name, "num_pred_segments": num_segments}
        present_ious: list[float] = []
        for group in ADE20K_BENCHMARK_GROUPS:
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
        if idx % 50 == 0 or idx == len(image_paths):
            print(f"[ade20k] {idx}/{len(image_paths)} images", flush=True)

    csv_path = output_root / "ade20k_group_benchmark.csv"
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
        "dataset": "ADE20K SceneParse150 validation",
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
    return write_json(output_root / "ade20k_group_benchmark_summary.json", summary)
