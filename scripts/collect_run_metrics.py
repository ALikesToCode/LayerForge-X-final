#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect LayerForge/Qwen run metrics into a compact markdown or JSON table.")
    p.add_argument("runs", nargs="+", help="Run directories containing metrics.json and optionally manifest.json")
    p.add_argument("--format", default="markdown", choices=["markdown", "json"], help="Output format")
    return p.parse_args()


def load_run(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    metrics_path = root / "metrics.json"
    manifest_path = root / "manifest.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {root}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    ordered = manifest.get("ordered_layers_near_to_far", [])
    return {
        "run": str(root),
        "input": manifest.get("input"),
        "segmentation_method": metrics.get("segmentation_method"),
        "depth_method": metrics.get("depth_method"),
        "ordering_method": metrics.get("ordering_method"),
        "inpaint_method": metrics.get("inpaint_method"),
        "num_layers": metrics.get("num_layers", len(ordered)),
        "recompose_psnr": metrics.get("recompose_psnr"),
        "recompose_ssim": metrics.get("recompose_ssim"),
        "mean_amodal_extra_ratio": metrics.get("mean_amodal_extra_ratio"),
        "mode": metrics.get("mode", "layerforge"),
    }


def to_markdown(rows: list[dict]) -> str:
    header = [
        "Run",
        "Mode",
        "Segmenter",
        "Depth",
        "Ordering",
        "Inpaint",
        "Layers",
        "PSNR",
        "SSIM",
        "Amodal+",
    ]
    out = [
        "| " + " | ".join(header) + " |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    Path(row["run"]).name,
                    str(row.get("mode", "")),
                    str(row.get("segmentation_method", "")),
                    str(row.get("depth_method", "")),
                    str(row.get("ordering_method", "")),
                    str(row.get("inpaint_method", "")),
                    f"{float(row.get('num_layers', 0)):.1f}" if row.get("num_layers") is not None else "",
                    f"{float(row.get('recompose_psnr', 0.0)):.4f}" if row.get("recompose_psnr") is not None else "",
                    f"{float(row.get('recompose_ssim', 0.0)):.4f}" if row.get("recompose_ssim") is not None else "",
                    f"{float(row.get('mean_amodal_extra_ratio', 0.0)):.4f}" if row.get("mean_amodal_extra_ratio") is not None else "",
                ]
            )
            + " |"
        )
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    rows = [load_run(p) for p in args.runs]
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        print(to_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
