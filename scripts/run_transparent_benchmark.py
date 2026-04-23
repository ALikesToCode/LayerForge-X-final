#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from layerforge.pipeline import LayerForgePipeline
from layerforge.transparent import export_transparent_assets

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LayerForge transparent-mode approximation on synthetic transparent scenes.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default="configs/fast.yaml")
    p.add_argument("--segmenter", default="classical")
    p.add_argument("--depth", default="geometric_luminance")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-scenes", type=int, default=None)
    return p.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8).astype(np.float32) / 255.0


def main() -> int:
    args = parse_args()
    dataset_dir = (ROOT / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = sorted([path for path in dataset_dir.iterdir() if path.is_dir() and (path / "input.png").exists()])
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    pipe = LayerForgePipeline(args.config, device=args.device)
    rows: list[dict] = []
    for scene_dir in scenes:
        metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
        label = str(metadata["label"]).replace("_", " ")
        scene_run_dir = output_dir / "runs" / scene_dir.name
        outputs = pipe.run(
            scene_dir / "input.png",
            scene_run_dir,
            segmenter=args.segmenter,
            depth_method=args.depth,
            prompts=[label],
            save_parallax=False,
        )
        result = export_transparent_assets(
            outputs.output_dir,
            output_dir=output_dir / "transparent_extract" / scene_dir.name,
            prompt=label,
        )
        gt_background = load_rgb(scene_dir / "background.png")
        gt_alpha = load_gray(scene_dir / "alpha_map.png")
        pred_background = load_rgb(output_dir / "transparent_extract" / scene_dir.name / "estimated_clean_background.png")
        pred_alpha = load_gray(output_dir / "transparent_extract" / scene_dir.name / "alpha_map.png")
        recomposition = load_rgb(output_dir / "transparent_extract" / scene_dir.name / "recomposition.png")
        input_rgb = load_rgb(scene_dir / "input.png")
        rows.append(
            {
                "scene": scene_dir.name,
                "label": metadata["label"],
                "run_dir": str(scene_run_dir.relative_to(ROOT)),
                "transparent_alpha_mae": float(np.mean(np.abs(pred_alpha - gt_alpha))),
                "background_psnr": float(peak_signal_noise_ratio(gt_background, pred_background, data_range=255)),
                "background_ssim": float(structural_similarity(gt_background, pred_background, channel_axis=2, data_range=255)),
                "recompose_psnr": float(peak_signal_noise_ratio(input_rgb, recomposition, data_range=255)),
                "recompose_ssim": float(structural_similarity(input_rgb, recomposition, channel_axis=2, data_range=255)),
                "alpha_nonzero_ratio": result["alpha_nonzero_ratio"],
            }
        )

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "rows": rows,
        "mean_transparent_alpha_mae": float(np.mean([row["transparent_alpha_mae"] for row in rows])) if rows else 0.0,
        "mean_background_psnr": float(np.mean([row["background_psnr"] for row in rows])) if rows else 0.0,
        "mean_background_ssim": float(np.mean([row["background_ssim"] for row in rows])) if rows else 0.0,
        "mean_recompose_psnr": float(np.mean([row["recompose_psnr"] for row in rows])) if rows else 0.0,
        "mean_recompose_ssim": float(np.mean([row["recompose_ssim"] for row in rows])) if rows else 0.0,
    }
    out_path = output_dir / "transparent_benchmark_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
