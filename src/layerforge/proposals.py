from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CandidateRunSpec:
    label: str
    mode: str
    run_dir: Path
    command: list[str]
    marker_name: str = "metrics.json"
    post_commands: list[list[str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def build_frontier_candidate_specs(
    *,
    image: Path,
    output_root: Path,
    qwen_layers: list[int],
    hybrid_modes: list[str],
    layerforge_bin: str,
    python_bin: str,
    qwen_score_script: Path,
    native_config: str,
    native_segmenter: str,
    native_depth: str,
    peeling_config: str,
    peeling_segmenter: str,
    peeling_depth: str,
    qwen_resolution: int,
    qwen_steps: int,
    qwen_device: str,
    qwen_dtype: str,
    qwen_offload: str,
    qwen_model: str,
    merge_external_layers: bool,
    include_native: bool = True,
    include_peeling: bool = True,
) -> list[CandidateRunSpec]:
    scene_root = output_root / image.stem
    specs: list[CandidateRunSpec] = []

    if include_native:
        specs.append(
            CandidateRunSpec(
                label="LayerForge native",
                mode="native",
                run_dir=scene_root / "native",
                command=[
                    layerforge_bin,
                    "run",
                    "--input",
                    str(image),
                    "--output",
                    str(scene_root / "native"),
                    "--config",
                    native_config,
                    "--segmenter",
                    native_segmenter,
                    "--depth",
                    native_depth,
                ],
                marker_name="metrics.json",
                extra={"candidate_type": "native"},
            )
        )

    if include_peeling:
        specs.append(
            CandidateRunSpec(
                label="LayerForge peeling",
                mode="peeling",
                run_dir=scene_root / "peeling",
                command=[
                    layerforge_bin,
                    "peel",
                    "--input",
                    str(image),
                    "--output",
                    str(scene_root / "peeling"),
                    "--config",
                    peeling_config,
                    "--segmenter",
                    peeling_segmenter,
                    "--depth",
                    peeling_depth,
                ],
                marker_name="metrics.json",
                extra={"candidate_type": "peeling"},
            )
        )

    qwen_script = qwen_score_script.parent / "run_qwen_image_layered.py"
    for layer_count in qwen_layers:
        qwen_dir = scene_root / f"qwen_{layer_count}"
        qwen_cmd = [
            python_bin,
            str(qwen_script),
            "--input",
            str(image),
            "--output-dir",
            str(qwen_dir),
            "--model",
            qwen_model,
            "--layers",
            str(layer_count),
            "--resolution",
            str(qwen_resolution),
            "--steps",
            str(qwen_steps),
            "--device",
            qwen_device,
            "--dtype",
            qwen_dtype,
            "--offload",
            qwen_offload,
        ]
        qwen_score_cmd = [
            python_bin,
            str(qwen_score_script),
            "--input",
            str(image),
            "--layers-dir",
            str(qwen_dir),
        ]
        specs.append(
            CandidateRunSpec(
                label=f"Qwen raw ({layer_count})",
                mode="qwen_raw",
                run_dir=qwen_dir,
                command=qwen_cmd,
                marker_name="manifest.json",
                post_commands=[qwen_score_cmd],
                extra={"requested_layers": layer_count, "candidate_type": "qwen_raw"},
            )
        )
        for hybrid_mode in hybrid_modes:
            hybrid_dir = scene_root / f"qwen_{layer_count}_hybrid_{hybrid_mode}"
            hybrid_cmd = [
                layerforge_bin,
                "enrich-qwen",
                "--input",
                str(image),
                "--layers-dir",
                str(qwen_dir),
                "--output",
                str(hybrid_dir),
                "--config",
                native_config,
                "--depth",
                native_depth,
            ]
            if hybrid_mode == "preserve":
                hybrid_cmd.append("--preserve-external-order")
            if merge_external_layers:
                hybrid_cmd.append("--merge-external-layers")
            specs.append(
                CandidateRunSpec(
                    label=f"Qwen + graph {hybrid_mode} ({layer_count})",
                    mode=f"qwen_hybrid_{hybrid_mode}",
                    run_dir=hybrid_dir,
                    command=hybrid_cmd,
                    marker_name="metrics.json",
                    extra={
                        "requested_layers": layer_count,
                        "hybrid_mode": hybrid_mode,
                        "candidate_type": "qwen_hybrid",
                    },
                )
            )
    return specs
