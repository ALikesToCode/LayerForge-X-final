from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .depth import estimate_depth
from .depth_benchmark import align_depth_prediction, build_valid_depth_mask, compute_depth_metrics, resize_depth_map, summarize_depth_rows
from .image_io import load_rgb
from .utils import seed_everything, write_json


METRIC_KEYS = ["mae", "rmse", "abs_rel", "sq_rel", "log_mae", "log_rmse", "rmse_log", "silog", "delta1", "delta2", "delta3"]


def resolve_diode_split_root(dataset_dir: str | Path, split: str = "val") -> Path:
    root = Path(dataset_dir)
    direct = root / split
    if (direct / "indoors").exists() or (direct / "outdoor").exists():
        return direct
    if (root / "indoors").exists() or (root / "outdoor").exists():
        return root
    raise FileNotFoundError(f"Could not locate DIODE split '{split}' under {root}")


def scene_type_from_path(path: str | Path) -> str:
    parts = {part.lower() for part in Path(path).parts}
    if "indoors" in parts:
        return "indoors"
    if "outdoor" in parts:
        return "outdoor"
    return "unknown"


def enumerate_diode_depth_samples(split_root: str | Path) -> list[dict[str, Path | str]]:
    root = Path(split_root)
    samples: list[dict[str, Path | str]] = []
    for depth_path in sorted(root.rglob("*_depth.npy")):
        if depth_path.name.endswith("_depth_mask.npy"):
            continue
        base = depth_path.with_name(depth_path.name[: -len("_depth.npy")])
        image_path = base.with_suffix(".png")
        mask_path = base.with_name(base.name + "_depth_mask.npy")
        if not image_path.exists() or not mask_path.exists():
            continue
        samples.append({
            "image": image_path,
            "depth": depth_path,
            "mask": mask_path,
            "scene_type": scene_type_from_path(depth_path),
        })
    if not samples:
        raise FileNotFoundError(f"No DIODE depth samples found under {root}")
    return samples


def auto_alignment_mode(depth_method: str) -> str:
    return "none" if str(depth_method).lower() in {"depth_pro", "depthpro", "depth-pro"} else "scale"


def run_diode_depth_benchmark(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "configs/fast.yaml",
    depth_method: str = "depth_pro",
    device: str = "auto",
    max_images: int | None = None,
    seed: int = 7,
    alignment: str = "auto",
) -> Path:
    split_root = resolve_diode_split_root(dataset_dir, split="val")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    depth_cfg = dict(cfg.get("depth", {}))
    depth_cfg["method"] = depth_method
    seed_everything(int(seed))

    samples = enumerate_diode_depth_samples(split_root)
    rng = random.Random(int(seed))
    rng.shuffle(samples)
    if max_images is not None:
        samples = samples[: max(0, int(max_images))]

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        gt_depth = np.asarray(np.load(sample["depth"]), dtype=np.float32).squeeze()
        gt_mask = np.asarray(np.load(sample["mask"])).squeeze()
        valid_mask = build_valid_depth_mask(gt_depth, gt_mask)
        rgb, pil = load_rgb(sample["image"], max_side=cfg.get("io", {}).get("max_side"))
        pred = estimate_depth(pil, rgb, depth_cfg, device=device)
        raw_pred = pred.raw_depth if pred.raw_depth is not None else pred.depth
        pred_resized = resize_depth_map(raw_pred, gt_depth.shape)
        actual_alignment = auto_alignment_mode(depth_method) if alignment == "auto" else alignment
        pred_aligned, alignment_meta = align_depth_prediction(pred_resized, gt_depth, valid_mask, mode=actual_alignment)
        metrics = compute_depth_metrics(gt_depth, pred_aligned, valid_mask)
        row: dict[str, Any] = {
            "file_name": Path(sample["image"]).name,
            "scene_type": str(sample["scene_type"]),
            "valid_pixels": int(valid_mask.sum()),
            "alignment": actual_alignment,
        }
        row.update(metrics)
        row.update({f"align_{key}": value for key, value in alignment_meta.items()})
        rows.append(row)
        if idx % 50 == 0 or idx == len(samples):
            print(f"[diode] {idx}/{len(samples)} images", flush=True)

    csv_path = output_root / "diode_depth_benchmark.csv"
    if rows:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    scene_breakdown: dict[str, dict[str, float]] = {}
    for scene_type in sorted({str(row["scene_type"]) for row in rows}):
        scene_rows = [row for row in rows if str(row["scene_type"]) == scene_type]
        scene_breakdown[scene_type] = summarize_depth_rows(scene_rows, METRIC_KEYS)
        scene_breakdown[scene_type]["images"] = len(scene_rows)

    summary = {
        "dataset": "DIODE validation depth",
        "dataset_dir": str(split_root),
        "images_evaluated": len(rows),
        "depth_method": depth_method,
        "alignment": rows[0]["alignment"] if rows else (auto_alignment_mode(depth_method) if alignment == "auto" else alignment),
        "seed": int(seed),
        "selected_by": "random_sample" if max_images is not None else "full_split",
        "metrics": summarize_depth_rows(rows, METRIC_KEYS),
        "scene_breakdown": scene_breakdown,
        "csv": str(csv_path),
    }
    return write_json(output_root / "diode_depth_benchmark_summary.json", summary)
