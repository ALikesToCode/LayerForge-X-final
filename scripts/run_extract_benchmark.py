#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from layerforge.editability import export_target_assets
from layerforge.pipeline import LayerForgePipeline
from layerforge.utils import mask_iou

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LayerForge promptable extraction on synthetic LayerBench-style scenes.")
    p.add_argument("--dataset-dir", required=True, help="Directory containing scene_*/image.png and ground_truth.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default="configs/fast.yaml")
    p.add_argument("--segmenter", default="grounded_sam2")
    p.add_argument("--depth", default="ensemble")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-scenes", type=int, default=10)
    return p.parse_args()


def load_gt_layers(scene_dir: Path) -> list[dict]:
    payload = json.loads((scene_dir / "ground_truth.json").read_text(encoding="utf-8"))
    return list(payload.get("layers_near_to_far", []))


def load_gt_alpha(scene_dir: Path, rel_path: str) -> np.ndarray:
    rgba = np.asarray(Image.open(scene_dir / rel_path).convert("RGBA"), dtype=np.uint8)
    return rgba[..., 3].astype(np.float32) / 255.0


def choose_target(scene_dir: Path) -> dict:
    candidates = []
    for layer in load_gt_layers(scene_dir):
        name = str(layer.get("name", ""))
        if "background" in name or "shadow" in name:
            continue
        alpha = load_gt_alpha(scene_dir, str(layer["path"]))
        area = float(np.count_nonzero(alpha > 0.05))
        candidates.append((area, layer, alpha))
    if not candidates:
        raise ValueError(f"No eligible target layer in {scene_dir}")
    _, layer, alpha = max(candidates, key=lambda item: item[0])
    return {"layer": layer, "alpha": alpha}


def bbox_from_alpha(alpha: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(alpha > 0.05)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def center_from_alpha(alpha: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(alpha > 0.05)
    if xs.size == 0 or ys.size == 0:
        return (0, 0)
    return (int(np.median(xs)), int(np.median(ys)))


def evaluate_query(run_dir: Path, scene_dir: Path, query_name: str, kwargs: dict, gt_name: str, gt_alpha: np.ndarray, out_root: Path) -> dict:
    query_out = out_root / query_name
    metadata = export_target_assets(run_dir, output_dir=query_out, **kwargs)
    pred_alpha = np.asarray(Image.open(query_out / "target_alpha.png").convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    pred_mask = pred_alpha > 0.05
    gt_mask = gt_alpha > 0.05
    selected_name = str(metadata["selected_target"]["name"])
    hit = gt_name in selected_name or selected_name in gt_name
    alpha_mae = float(np.mean(np.abs(pred_alpha - gt_alpha)))
    row = {
        "scene": scene_dir.name,
        "query_type": query_name,
        "target_name": gt_name,
        "selected_name": selected_name,
        "target_hit": bool(hit),
        "target_iou": float(mask_iou(gt_mask, pred_mask)),
        "alpha_mae": alpha_mae,
        "prompt": kwargs.get("prompt"),
        "point": list(kwargs["point"]) if kwargs.get("point") is not None else None,
        "box": list(kwargs["box"]) if kwargs.get("box") is not None else None,
    }
    return row


def main() -> int:
    args = parse_args()
    dataset_dir = (ROOT / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = sorted([path for path in dataset_dir.iterdir() if path.is_dir() and (path / "image.png").exists()])[: args.max_scenes]

    pipe = LayerForgePipeline(args.config, device=args.device)
    rows: list[dict] = []
    for scene_dir in scenes:
        target = choose_target(scene_dir)
        gt_layer = target["layer"]
        gt_name = str(gt_layer["name"])
        gt_alpha = target["alpha"]
        bbox = bbox_from_alpha(gt_alpha)
        point = center_from_alpha(gt_alpha)
        scene_extract_root = output_dir / "extracts" / scene_dir.name
        prompt = gt_name.replace("_", " ")
        queries = {
            "text": {"prompt": prompt},
            "point": {"point": point},
            "box": {"box": bbox},
            "text_point": {"prompt": prompt, "point": point},
            "text_box": {"prompt": prompt, "box": bbox},
        }
        for query_name, kwargs in queries.items():
            query_run_dir = output_dir / "runs" / scene_dir.name / query_name
            outputs = pipe.run(
                scene_dir / "image.png",
                query_run_dir,
                segmenter=args.segmenter,
                depth_method=args.depth,
                prompts=[prompt] if kwargs.get("prompt") else None,
                prompt_source="manual" if kwargs.get("prompt") else None,
                save_parallax=False,
            )
            rows.append(
                evaluate_query(
                    outputs.output_dir,
                    scene_dir,
                    query_name,
                    kwargs,
                    gt_name,
                    gt_alpha,
                    scene_extract_root,
                )
            )

    summary_rows = []
    for query_type in sorted({row["query_type"] for row in rows}):
        items = [row for row in rows if row["query_type"] == query_type]
        summary_rows.append(
            {
                "query_type": query_type,
                "queries": len(items),
                "target_hit_rate": float(sum(1 for row in items if row["target_hit"]) / max(1, len(items))),
                "mean_target_iou": float(sum(row["target_iou"] for row in items) / max(1, len(items))),
                "mean_alpha_mae": float(sum(row["alpha_mae"] for row in items) / max(1, len(items))),
            }
        )

    payload = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "max_scenes": len(scenes),
        "rows": rows,
        "summary": summary_rows,
    }
    out_path = output_dir / "extract_benchmark_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
