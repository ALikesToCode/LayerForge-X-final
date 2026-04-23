#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "report_artifacts"
ARCHIVE_TAG = "LayerForge-X-final-submission-2026-04-23"

sys.path.insert(0, str(ROOT / "src"))

from layerforge.site_data import write_project_site_payload  # noqa: E402


SNAPSHOTS = {
    "coco_panoptic_group_benchmark_summary.json": ROOT / "results/coco_panoptic_mask2former_512/coco_panoptic_group_benchmark_summary.json",
    "ade20k_group_benchmark_summary.json": ROOT / "results/ade20k_mask2former_512/ade20k_group_benchmark_summary.json",
    "diode_depth_benchmark_summary.json": ROOT / "results/diode_depthpro_full/diode_depth_benchmark_summary.json",
    "diode_depth_scale_benchmark_summary.json": ROOT / "results/diode_depthpro_scale_full/diode_depth_benchmark_summary.json",
    "diode_geometric_benchmark_summary.json": ROOT / "results/diode_geometric_full/diode_depth_benchmark_summary.json",
    "synthetic_benchmark_summary.json": ROOT / "results/synthetic_fast/synthetic_benchmark_summary.json",
    "truck_best_metrics.json": ROOT / "runs/truck_candidate_search_v2/best/metrics.json",
    "truck_search_summary.json": ROOT / "runs/truck_candidate_search_v2/search_summary.json",
    "qwen_baseline_metrics.json": ROOT / "runs/qwen_truck_layers_raw_640_20/metrics.json",
    "qwen_enriched_metrics.json": ROOT / "runs/qwen_truck_enriched_640_20/metrics.json",
    "qwen_five_image_review_summary.json": ROOT / "runs/qwen_five_image_review/comparison_summary.json",
    "frontier_review_summary.json": ROOT / "runs/frontier_review/frontier_summary.json",
    "editability_suite_summary.json": ROOT / "runs/frontier_review/editability_suite_summary.json",
    "extract_benchmark_summary.json": ROOT / "runs/extract_benchmark_prompted_grounded/extract_benchmark_summary.json",
    "transparent_benchmark_summary.json": ROOT / "runs/transparent_benchmark/transparent_benchmark_summary.json",
    "effects_demo_metrics.json": ROOT / "runs/effects_groundtruth_demo_cutting_edge/metrics.json",
}


FIGURE_SOURCES = {
    "truck_recomposition_comparison": [
        "runs/qwen_truck_layers_raw_640_20",
        "runs/qwen_truck_enriched_640_20",
        "runs/truck_candidate_search_v2/best",
    ],
    "truck_layer_stack_comparison": [
        "runs/qwen_truck_layers_raw_640_20",
        "runs/qwen_truck_enriched_640_20",
        "runs/truck_candidate_search_v2/best",
    ],
    "truck_metrics_comparison": [
        "runs/qwen_truck_layers_raw_640_20",
        "runs/qwen_truck_enriched_640_20",
        "runs/truck_best_score",
        "runs/truck_best_score_manual",
        "runs/truck_best_score_augment",
        "runs/truck_candidate_search_v2/best",
    ],
    "truck_prompt_ablation": [
        "runs/truck_best_score",
        "runs/truck_best_score_manual",
        "runs/truck_best_score_augment",
        "runs/truck_candidate_search_v2",
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
    "frontier_review": [
        "runs/frontier_review",
    ],
    "prompt_extract_benchmark": [
        "data/layerbenchpp_prompt_benchmark",
        "runs/extract_benchmark_prompted_grounded",
    ],
    "transparent_benchmark": [
        "data/transparent_benchmark",
        "runs/transparent_benchmark",
    ],
    "intrinsic_layer_demo": [
        "runs/truck_candidate_search_v2/best",
    ],
}


COMMAND_LOG = """# Command Log

## Environment metadata

- Python: `{python_version}`
- torch: `{torch_version}`
- transformers: `{transformers_version}`
- diffusers: `{diffusers_version}`
- accelerate: `{accelerate_version}`
- safetensors: `{safetensors_version}`
- archive tag: `{archive_tag}`

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
layerforge enrich-qwen --input data/demo/truck.jpg --layers-dir runs/qwen_truck_layers_raw_640_20 --output runs/qwen_truck_enriched_640_20 --config configs/cutting_edge.yaml --depth depth_pro --preserve-external-order
```

## Five-image Qwen review

```bash
python scripts/run_curated_comparison.py --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png data/qualitative_pack/coffee.png data/qualitative_pack/chelsea_cat.png data/qualitative_pack/image.png --output-root runs/qwen_five_image_review --native-config configs/best_score.yaml --native-segmenter grounded_sam2 --native-depth ensemble --qwen-layers 3,4,6,8 --qwen-steps 10 --qwen-resolution 640 --qwen-device cuda --qwen-dtype bfloat16 --qwen-offload sequential --skip-existing
```

## Frontier comparison and self-evaluation

```bash
python scripts/run_frontier_comparison.py --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png data/qualitative_pack/coffee.png data/qualitative_pack/chelsea_cat.png data/qualitative_pack/image.png --output-root runs/frontier_review --native-config configs/frontier.yaml --peeling-config configs/recursive_peeling.yaml --qwen-layers 4 --qwen-steps 10 --qwen-resolution 640 --qwen-device cuda --qwen-dtype bfloat16 --qwen-offload sequential --skip-existing
python scripts/run_editability_suite.py --frontier-summary runs/frontier_review/frontier_summary.json --output runs/frontier_review/editability_suite_summary.json
```

## Promptable extraction benchmark

```bash
python scripts/make_synthetic_dataset.py --output data/layerbenchpp_prompt_benchmark --count 10 --output-format layerbench_pp --with-effects
python scripts/run_extract_benchmark.py --dataset-dir data/layerbenchpp_prompt_benchmark --output-dir runs/extract_benchmark_prompted_grounded --segmenter grounded_sam2 --depth ensemble --device cuda --max-scenes 10
```

## Transparent decomposition benchmark

```bash
python scripts/make_transparent_dataset.py --output data/transparent_benchmark --count 12
python scripts/run_transparent_benchmark.py --dataset-dir data/transparent_benchmark --output-dir runs/transparent_benchmark
```

## Associated-effect demo

```bash
python scripts/make_synthetic_dataset.py --output runs/effects_demo_source --count 1 --seed 77 --width 640 --height 420 --output-format layerbench_pp --with-effects
python scripts/run_effects_demo.py --scene-dir runs/effects_demo_source/scene_000 --output runs/effects_groundtruth_demo_cutting_edge --config configs/cutting_edge.yaml
```

## Native search run

```bash
layerforge autotune --input data/demo/truck.jpg --output runs/truck_candidate_search_v2 --config configs/best_score.yaml --prompts "truck,road,sky,tree,building,window,wheel,car" --device cuda --no-parallax
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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def copy_snapshots() -> dict[str, object]:
    dst = TARGET / "metrics_snapshots"
    dst.mkdir(parents=True, exist_ok=True)
    copied: dict[str, object] = {}
    for name, src in SNAPSHOTS.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source artifact: {src}")
        payload = json.loads(src.read_text(encoding="utf-8"))
        (dst / name).write_text(json.dumps(_sanitize_json(payload), indent=2, sort_keys=True), encoding="utf-8")
        copied[name] = payload
    return copied


def _aggregate_lookup(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["label"]): row for row in rows}


def _best_image_wins(best_by_image: list[dict[str, object]]) -> dict[str, int]:
    wins: dict[str, int] = {}
    for row in best_by_image:
        label = str(row.get("label"))
        wins[label] = wins.get(label, 0) + 1
    return wins


def refresh_project_manifest(snapshot_payloads: dict[str, object]) -> None:
    manifest_path = ROOT / "PROJECT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    measured = manifest["measured_results"]

    qwen_summary = snapshot_payloads["qwen_five_image_review_summary.json"]
    qwen_rows = _aggregate_lookup(qwen_summary["aggregates"])
    frontier_summary = snapshot_payloads["frontier_review_summary.json"]
    frontier_rows = _aggregate_lookup(frontier_summary["aggregates"])

    qwen = measured["qwen_five_image_review"]
    qwen["qwen_raw_mean_psnr"] = float(qwen_rows["Qwen raw (4)"]["mean_psnr"])
    qwen["qwen_raw_mean_ssim"] = float(qwen_rows["Qwen raw (4)"]["mean_ssim"])
    qwen["qwen_graph_preserve_mean_psnr"] = float(qwen_rows["Qwen + graph preserve (4)"]["mean_psnr"])
    qwen["qwen_graph_preserve_mean_ssim"] = float(qwen_rows["Qwen + graph preserve (4)"]["mean_ssim"])
    qwen["qwen_graph_reorder_mean_psnr"] = float(qwen_rows["Qwen + graph reorder (4)"]["mean_psnr"])
    qwen["qwen_graph_reorder_mean_ssim"] = float(qwen_rows["Qwen + graph reorder (4)"]["mean_ssim"])
    qwen["qwen_raw_mean_psnr_by_layers"] = {
        str(layer): float(qwen_rows[f"Qwen raw ({layer})"]["mean_psnr"])
        for layer in (3, 4, 6, 8)
    }
    qwen["qwen_raw_mean_ssim_by_layers"] = {
        str(layer): float(qwen_rows[f"Qwen raw ({layer})"]["mean_ssim"])
        for layer in (3, 4, 6, 8)
    }
    qwen["qwen_graph_preserve_mean_psnr_by_layers"] = {
        str(layer): float(qwen_rows[f"Qwen + graph preserve ({layer})"]["mean_psnr"])
        for layer in (3, 4, 6, 8)
    }
    qwen["qwen_graph_preserve_mean_ssim_by_layers"] = {
        str(layer): float(qwen_rows[f"Qwen + graph preserve ({layer})"]["mean_ssim"])
        for layer in (3, 4, 6, 8)
    }
    qwen["qwen_graph_reorder_mean_psnr_by_layers"] = {
        str(layer): float(qwen_rows[f"Qwen + graph reorder ({layer})"]["mean_psnr"])
        for layer in (3, 4, 6, 8)
    }
    qwen["qwen_graph_reorder_mean_ssim_by_layers"] = {
        str(layer): float(qwen_rows[f"Qwen + graph reorder ({layer})"]["mean_ssim"])
        for layer in (3, 4, 6, 8)
    }

    frontier = measured["frontier_review"]
    frontier["layerforge_native_mean_psnr"] = float(frontier_rows["LayerForge native"]["mean_psnr"])
    frontier["layerforge_native_mean_ssim"] = float(frontier_rows["LayerForge native"]["mean_ssim"])
    frontier["layerforge_native_mean_self_eval_score"] = float(frontier_rows["LayerForge native"]["mean_self_eval_score"])
    frontier["layerforge_peeling_mean_psnr"] = float(frontier_rows["LayerForge peeling"]["mean_psnr"])
    frontier["layerforge_peeling_mean_ssim"] = float(frontier_rows["LayerForge peeling"]["mean_ssim"])
    frontier["layerforge_peeling_mean_self_eval_score"] = float(frontier_rows["LayerForge peeling"]["mean_self_eval_score"])
    frontier["qwen_raw_mean_psnr"] = float(frontier_rows["Qwen raw (4)"]["mean_psnr"])
    frontier["qwen_raw_mean_ssim"] = float(frontier_rows["Qwen raw (4)"]["mean_ssim"])
    frontier["qwen_raw_mean_self_eval_score"] = float(frontier_rows["Qwen raw (4)"]["mean_self_eval_score"])
    frontier["qwen_graph_preserve_mean_psnr"] = float(frontier_rows["Qwen + graph preserve (4)"]["mean_psnr"])
    frontier["qwen_graph_preserve_mean_ssim"] = float(frontier_rows["Qwen + graph preserve (4)"]["mean_ssim"])
    frontier["qwen_graph_preserve_mean_self_eval_score"] = float(frontier_rows["Qwen + graph preserve (4)"]["mean_self_eval_score"])
    frontier["qwen_graph_reorder_mean_psnr"] = float(frontier_rows["Qwen + graph reorder (4)"]["mean_psnr"])
    frontier["qwen_graph_reorder_mean_ssim"] = float(frontier_rows["Qwen + graph reorder (4)"]["mean_ssim"])
    frontier["qwen_graph_reorder_mean_self_eval_score"] = float(frontier_rows["Qwen + graph reorder (4)"]["mean_self_eval_score"])
    frontier["best_image_wins"] = _best_image_wins(frontier_summary["best_by_image"])

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def _sanitize_string(value: str) -> str:
    root_prefix = str(ROOT) + "/"
    if value == str(ROOT):
        return "."
    cleaned = value.replace(root_prefix, "")
    home_pattern = re.compile(r"/home/[A-Za-z0-9._-]+(?:/[^\s\"']*)?")
    cleaned = home_pattern.sub("[local-path]", cleaned)
    return cleaned


def _summarize_log(value: str) -> str:
    cleaned = _sanitize_string(value)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    kept: list[str] = []
    for line in lines:
        if len(kept) >= 6:
            break
        if any(token in line.lower() for token in ("warning", "error", "deprecated", "hf_", "sam2", "transformers", "resume_download")):
            kept.append(line)
    if not kept:
        kept = lines[:3]
    suffix = "" if len(lines) <= len(kept) else " ... [scrubbed local runtime log]"
    return "\n".join(kept) + suffix


def _sanitize_json(value, key: str | None = None):
    if isinstance(value, dict):
        return {child_key: _sanitize_json(item, key=child_key) for child_key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item, key=key) for item in value]
    if isinstance(value, str):
        if key in {"stderr", "stdout"}:
            return _summarize_log(value)
        return _sanitize_string(value)
    return value


def write_figure_sources() -> None:
    dst = TARGET / "figure_sources"
    dst.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_dependencies_omitted_from_submission_zip": True,
        "generated_figures": {name: f"docs/figures/{name}.png" for name in FIGURE_SOURCES},
        "source_dependencies": FIGURE_SOURCES,
    }
    (dst / "figure_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_readme() -> None:
    text = """# Report Artifacts

This folder collects the compact reported artifacts referenced by the README and `PROJECT_MANIFEST.json`.

- `metrics_snapshots/` contains compact JSON summaries copied from the measured local `runs/` and `results/` directories.
- `figure_sources/figure_manifest.json` records which raw runs and datasets were used to build the report figures and whether those dependencies are omitted from the submission ZIP.
- `command_log.md` lists the command families used to generate the copied artifacts and the verified package/runtime versions used for the current archive refresh.

The goal is to keep the repository reviewable even when heavyweight directories such as `data/`, `runs/`, and `results/` are excluded from the public tree and final archive.

Treat `PROJECT_MANIFEST.json`, `metrics_snapshots/`, and `command_log.md` as the canonical reported artifacts for the repository.
"""
    (TARGET / "README.md").write_text(text, encoding="utf-8")


def write_command_log() -> None:
    rendered = COMMAND_LOG.format(
        python_version=sys.version.split()[0],
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
        diffusers_version=_package_version("diffusers"),
        accelerate_version=_package_version("accelerate"),
        safetensors_version=_package_version("safetensors"),
        archive_tag=ARCHIVE_TAG,
    )
    (TARGET / "command_log.md").write_text(rendered, encoding="utf-8")


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshots = copy_snapshots()
    refresh_project_manifest(snapshots)
    write_figure_sources()
    write_command_log()
    write_readme()
    write_project_site_payload(ROOT, ROOT / "docs" / "site-data" / "project_site.json")
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
