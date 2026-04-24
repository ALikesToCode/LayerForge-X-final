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
from layerforge.editability import target_geometry_is_confident
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


def _safe_run_key(prompt: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
    return f"geometry_prompted_{normalized or 'target'}"


def evaluate_query(run_dir: Path, scene_dir: Path, query_name: str, kwargs: dict, gt_name: str, gt_alpha: np.ndarray, out_root: Path) -> dict:
    query_out = out_root / query_name
    metadata = export_target_assets(run_dir, output_dir=query_out, **kwargs)
    selected_target = metadata["selected_target"]
    pred_alpha = np.asarray(Image.open(query_out / "target_alpha.png").convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    pred_mask = pred_alpha > 0.05
    gt_mask = gt_alpha > 0.05
    selected_name = str(selected_target["name"])
    semantic_candidates = [
        selected_name,
        str(selected_target.get("label", "")),
        str(selected_target.get("semantic_name", "")),
        str(selected_target.get("semantic_label", "")),
        str(metadata.get("resolved_prompt", "")),
        str(metadata.get("prompt", "")),
    ]
    gt_key = gt_name.replace(" ", "_").lower()
    normalized = [item.replace(" ", "_").lower() for item in semantic_candidates if item]
    semantic_hit = any(gt_key in item or item in gt_key for item in normalized)
    alpha_mae = float(np.mean(np.abs(pred_alpha - gt_alpha)))
    target_iou = float(mask_iou(gt_mask, pred_mask))
    row = {
        "scene": scene_dir.name,
        "query_type": query_name,
        "target_name": gt_name,
        "selected_name": selected_name,
        "selected_semantic_name": selected_target.get("semantic_name"),
        "selected_semantic_label": selected_target.get("semantic_label"),
        "semantic_hit": bool(semantic_hit),
        "target_hit": bool(semantic_hit and target_iou > 0.05),
        "target_iou": target_iou,
        "alpha_mae": alpha_mae,
        "prompt": kwargs.get("prompt"),
        "resolved_prompt": metadata.get("resolved_prompt"),
        "geometry_match": metadata.get("geometry_match"),
        "point": list(kwargs["point"]) if kwargs.get("point") is not None else None,
        "box": list(kwargs["box"]) if kwargs.get("box") is not None else None,
    }
    return row


def _needs_geometry_prompted_fallback(query_name: str, row: dict) -> bool:
    return (
        query_name in {"point", "box"}
        and bool(row.get("resolved_prompt"))
        and not target_geometry_is_confident({"geometry_match": row.get("geometry_match")})
    )


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
        scene_run_root = output_dir / "runs" / scene_dir.name
        queries = {
            "text": {"base_run_key": "prompted", "prompt": prompt},
            "point": {"base_run_key": "unguided", "point": point},
            "box": {"base_run_key": "unguided", "box": bbox},
            "text_point": {"base_run_key": "prompted", "prompt": prompt, "point": point},
            "text_box": {"base_run_key": "prompted", "prompt": prompt, "box": bbox},
        }

        base_runs: dict[str, Path] = {}
        for run_key, prompts in {
            "prompted": [prompt],
            "unguided": None,
        }.items():
            query_run_dir = scene_run_root / run_key
            outputs = pipe.run(
                scene_dir / "image.png",
                query_run_dir,
                segmenter=args.segmenter,
                depth_method=args.depth,
                prompts=prompts,
                prompt_source="manual" if prompts else None,
                save_parallax=False,
            )
            base_runs[run_key] = outputs.output_dir

        for query_name, kwargs in queries.items():
            export_kwargs = {key: value for key, value in kwargs.items() if key != "base_run_key"}
            row = evaluate_query(
                base_runs[kwargs["base_run_key"]],
                scene_dir,
                query_name,
                export_kwargs,
                gt_name,
                gt_alpha,
                scene_extract_root,
            )
            if _needs_geometry_prompted_fallback(query_name, row):
                fallback_prompt = str(row["resolved_prompt"])
                run_key = _safe_run_key(fallback_prompt)
                if run_key not in base_runs:
                    outputs = pipe.run(
                        scene_dir / "image.png",
                        scene_run_root / run_key,
                        segmenter=args.segmenter,
                        depth_method=args.depth,
                        prompts=[fallback_prompt],
                        prompt_source="manual",
                        save_parallax=False,
                    )
                    base_runs[run_key] = outputs.output_dir
                fallback_kwargs = dict(export_kwargs)
                fallback_kwargs["prompt"] = fallback_prompt
                row = evaluate_query(
                    base_runs[run_key],
                    scene_dir,
                    query_name,
                    fallback_kwargs,
                    gt_name,
                    gt_alpha,
                    scene_extract_root,
                )
                row["used_geometry_prompt_fallback"] = True
                row["fallback_prompt"] = fallback_prompt
            else:
                row["used_geometry_prompt_fallback"] = False
            rows.append(row)

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
