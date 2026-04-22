from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import LayerForgePipeline
from .utils import ensure_dir, write_json


PROMPTABLE_SEGMENTERS = {"grounded_sam2", "grounded-sam2", "open_vocab", "open-vocab-sam2"}


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    description: str
    overrides: dict[str, Any]


def build_autotune_candidates(segmenter: str, prompts: list[str] | None) -> list[CandidateSpec]:
    method = str(segmenter).lower()
    if method not in PROMPTABLE_SEGMENTERS:
        return [
            CandidateSpec(
                name="base",
                description="Base configuration without prompt/threshold search",
                overrides={},
            )
        ]

    threshold_presets = [
        ("balanced", 0.24, 0.20),
        ("recall", 0.20, 0.18),
        ("precision", 0.28, 0.22),
    ]
    prompt_modes = ["gemini"]
    if prompts:
        prompt_modes = ["manual", "augment", "gemini"]

    candidates: list[CandidateSpec] = []
    for prompt_source in prompt_modes:
        for preset_name, box_threshold, text_threshold in threshold_presets:
            if prompt_source == "gemini" and preset_name == "precision":
                continue
            candidates.append(
                CandidateSpec(
                    name=f"{prompt_source}_{preset_name}",
                    description=f"{prompt_source} prompts with {preset_name} thresholds",
                    overrides={
                        "segmentation": {
                            "prompt_source": prompt_source,
                            "model": {
                                "box_threshold": box_threshold,
                                "text_threshold": text_threshold,
                            },
                        }
                    },
                )
            )
    return candidates


def candidate_rank_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    psnr = float(metrics.get("recompose_psnr", float("-inf")))
    ssim = float(metrics.get("recompose_ssim", float("-inf")))
    num_layers = float(metrics.get("num_layers", float("inf")))
    return (psnr, ssim, -num_layers)


def run_autotune(
    pipe: LayerForgePipeline,
    *,
    input_path: str | Path,
    output_dir: str | Path,
    segmenter: str,
    depth_method: str | None,
    prompts: list[str] | None,
    flip_depth: bool | None,
    save_parallax: bool | None,
    ordering_method: str | None,
    ranker_model_path: str | Path | None,
) -> dict[str, Any]:
    root = ensure_dir(output_dir)
    candidates_dir = ensure_dir(root / "candidates")
    specs = build_autotune_candidates(segmenter, prompts)

    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_output_dir: Path | None = None

    for spec in specs:
        candidate_output = candidates_dir / spec.name
        if candidate_output.exists():
            shutil.rmtree(candidate_output)
        result = pipe.run(
            input_path,
            candidate_output,
            segmenter=segmenter,
            depth_method=depth_method,
            prompts=prompts,
            flip_depth=flip_depth,
            save_parallax=save_parallax,
            ordering_method=ordering_method,
            ranker_model_path=ranker_model_path,
            config_overrides=spec.overrides,
        )
        metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
        entry = {
            "name": spec.name,
            "description": spec.description,
            "output_dir": str(result.output_dir),
            "manifest_path": str(result.manifest_path),
            "metrics_path": str(result.metrics_path),
            "metrics": metrics,
            "rank_key": list(candidate_rank_key(metrics)),
        }
        summaries.append(entry)
        if best is None or tuple(entry["rank_key"]) > tuple(best["rank_key"]):
            best = entry
            best_output_dir = result.output_dir

    if best is None or best_output_dir is None:
        raise RuntimeError("Autotune produced no candidate outputs")

    best_dir = root / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best_output_dir, best_dir)

    summary = {
        "input": str(input_path),
        "selected_by": "lexicographic(recompose_psnr, recompose_ssim, -num_layers)",
        "candidates": summaries,
        "best": {
            "name": best["name"],
            "description": best["description"],
            "source_output_dir": best["output_dir"],
            "copied_best_dir": str(best_dir),
            "metrics": best["metrics"],
        },
    }
    write_json(root / "search_summary.json", summary)
    return summary
