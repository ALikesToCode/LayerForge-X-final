#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "report_artifacts"


SNAPSHOTS = {
    "coco_panoptic_group_benchmark_summary.json": ROOT / "results/coco_panoptic_mask2former_512/coco_panoptic_group_benchmark_summary.json",
    "ade20k_group_benchmark_summary.json": ROOT / "results/ade20k_mask2former_512/ade20k_group_benchmark_summary.json",
    "diode_depth_benchmark_summary.json": ROOT / "results/diode_depthpro_full/diode_depth_benchmark_summary.json",
    "diode_depth_scale_benchmark_summary.json": ROOT / "results/diode_depthpro_scale_full/diode_depth_benchmark_summary.json",
    "diode_geometric_benchmark_summary.json": ROOT / "results/diode_geometric_full/diode_depth_benchmark_summary.json",
    "synthetic_benchmark_summary.json": ROOT / "results/synthetic_fast/synthetic_benchmark_summary.json",
    "truck_best_metrics.json": ROOT / "runs/truck_state_of_art_search_v2/best/metrics.json",
    "truck_search_summary.json": ROOT / "runs/truck_state_of_art_search_v2/search_summary.json",
    "qwen_baseline_metrics.json": ROOT / "runs/qwen_truck_layers_raw_640_20/metrics.json",
    "qwen_enriched_metrics.json": ROOT / "runs/qwen_truck_enriched_640_20/metrics.json",
    "qwen_five_image_review_summary.json": ROOT / "runs/qwen_five_image_review/comparison_summary.json",
    "effects_demo_metrics.json": ROOT / "runs/effects_groundtruth_demo_cutting_edge/metrics.json",
}


FIGURE_SOURCES = {
    "truck_recomposition_comparison": [
        "runs/qwen_truck_layers_raw_640_20",
        "runs/qwen_truck_enriched_640_20",
        "runs/truck_state_of_art_search_v2/best",
    ],
    "truck_layer_stack_comparison": [
        "runs/qwen_truck_layers_raw_640_20",
        "runs/qwen_truck_enriched_640_20",
        "runs/truck_state_of_art_search_v2/best",
    ],
    "truck_metrics_comparison": [
        "runs/qwen_truck_layers_raw_640_20",
        "runs/qwen_truck_enriched_640_20",
        "runs/truck_best_score",
        "runs/truck_best_score_manual",
        "runs/truck_best_score_augment",
        "runs/truck_state_of_art_search_v2/best",
    ],
    "truck_prompt_ablation": [
        "runs/truck_best_score",
        "runs/truck_best_score_manual",
        "runs/truck_best_score_augment",
        "runs/truck_state_of_art_search_v2",
    ],
    "synthetic_ordering_ablation": [
        "results/synthetic_fast",
        "results/synth_boundary_test",
        "results/synth_learned_test",
    ],
    "public_benchmark_comparison": [
        "results/coco_panoptic_mask2former_512",
        "results/ade20k_mask2former_512",
    ],
    "public_depth_comparison": [
        "results/diode_depthpro_full",
        "results/diode_depthpro_scale_full",
        "results/diode_geometric_full",
    ],
    "qualitative_gallery": [
        "data/qualitative_pack/astronaut.png",
        "data/qualitative_pack/coffee.png",
        "data/qualitative_pack/chelsea_cat.png",
        "runs/qualitative_pack_cutting_edge/astronaut",
        "runs/qualitative_pack_cutting_edge/coffee",
        "runs/qualitative_pack_cutting_edge/chelsea_cat",
    ],
    "effects_layer_demo": [
        "runs/effects_demo_source/scene_000",
        "runs/effects_groundtruth_demo_cutting_edge",
    ],
}


COMMAND_LOG = """# Command Log

These are the exact command families used to produce the auditable summaries copied into `report_artifacts/metrics_snapshots/`.

## Validation

```bash
./.venv/bin/pytest -q
./.venv/bin/pytest -q tests/test_smoke.py tests/test_merge.py tests/test_segment_api.py
```

## Synthetic ordering benchmark

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 20
layerforge benchmark --dataset-dir data/synthetic_layerbench --output-dir results/synthetic_fast --config configs/fast.yaml --segmenter classical --depth geometric_luminance
```

## Qwen baseline and hybrid enrichment

```bash
python scripts/run_qwen_image_layered.py --input data/demo/truck.jpg --output-dir runs/qwen_truck_layers_raw_640_20 --layers 4 --resolution 640 --steps 20 --device cuda --dtype bfloat16 --offload sequential
layerforge enrich-qwen --input data/demo/truck.jpg --layers-dir runs/qwen_truck_layers_raw_640_20 --output runs/qwen_truck_enriched_640_20 --config configs/cutting_edge.yaml --depth depth_pro
```

## Five-image Qwen review

```bash
python scripts/run_curated_comparison.py --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png data/qualitative_pack/coffee.png data/qualitative_pack/chelsea_cat.png examples/synth/scene_000/image.png --output-root runs/qwen_five_image_review --qwen-layers 4 --qwen-steps 10 --qwen-resolution 640 --qwen-device cuda --qwen-dtype bfloat16 --qwen-offload sequential --skip-native --skip-existing
```

## Associated-effect demo

```bash
python scripts/make_synthetic_dataset.py --output runs/effects_demo_source --count 1 --seed 77 --width 640 --height 420 --output-format layerbench_pp --with-effects
python scripts/run_effects_demo.py --scene-dir runs/effects_demo_source/scene_000 --output runs/effects_groundtruth_demo_cutting_edge --config configs/cutting_edge.yaml
```

## Native search run

```bash
layerforge autotune --input data/demo/truck.jpg --output runs/truck_state_of_art_search_v2 --config configs/best_score.yaml --prompts "truck,road,sky,tree,building,window,wheel,car" --device cuda --no-parallax
```

## Public grouping and depth benchmarks

```bash
layerforge benchmark-coco-panoptic --dataset-dir data/coco_panoptic_val --output-dir results/coco_panoptic_mask2former_512 --config configs/fast.yaml --segmenter mask2former --device cuda --max-images 512 --seed 7
layerforge benchmark-ade20k --dataset-dir data/ade20k --output-dir results/ade20k_mask2former_512 --config configs/ade20k_mask2former.yaml --segmenter mask2former --device cuda --max-images 512 --seed 7
layerforge benchmark-diode --dataset-dir data/diode --output-dir results/diode_depthpro_full --config configs/diode_depthpro.yaml --depth depth_pro --device cuda --seed 7
layerforge benchmark-diode --dataset-dir data/diode --output-dir results/diode_depthpro_scale_full --config configs/diode_depthpro.yaml --depth depth_pro --alignment scale --device cuda --seed 7
layerforge benchmark-diode --dataset-dir data/diode --output-dir results/diode_geometric_full --config configs/diode_depthpro.yaml --depth geometric_luminance --alignment scale --device cpu --seed 7
```
"""


def copy_snapshots() -> None:
    dst = TARGET / "metrics_snapshots"
    dst.mkdir(parents=True, exist_ok=True)
    for name, src in SNAPSHOTS.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source artifact: {src}")
        shutil.copy2(src, dst / name)


def write_figure_sources() -> None:
    dst = TARGET / "figure_sources"
    dst.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_figures": {name: f"docs/figures/{name}.png" for name in FIGURE_SOURCES},
        "source_dependencies": FIGURE_SOURCES,
    }
    (dst / "figure_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_readme() -> None:
    text = """# Report Artifacts

This folder is the submission-safe evidence pack for the claims made in the README and `PROJECT_MANIFEST.json`.

- `metrics_snapshots/` contains compact JSON summaries copied from the measured `runs/` and `results/` directories.
- `figure_sources/figure_manifest.json` records which raw runs and datasets were used to build the report figures.
- `command_log.md` lists the command families used to generate the copied artifacts.

The goal is to keep the archive auditable even when heavyweight directories such as `data/`, `runs/`, and `results/` are excluded from a ZIP submission.
"""
    (TARGET / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    copy_snapshots()
    write_figure_sources()
    (TARGET / "command_log.md").write_text(COMMAND_LOG, encoding="utf-8")
    write_readme()
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
