from __future__ import annotations

from pathlib import Path

from layerforge.proposals import build_frontier_candidate_specs


def test_build_frontier_candidate_specs_includes_native_peeling_and_qwen_modes(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-an-image")
    output_root = tmp_path / "runs"

    specs = build_frontier_candidate_specs(
        image=image,
        output_root=output_root,
        qwen_layers=[4],
        hybrid_modes=["preserve", "reorder"],
        layerforge_bin=".venv/bin/layerforge",
        python_bin="python",
        qwen_score_script=Path("scripts/score_qwen_raw_layers.py"),
        native_config="configs/best_score.yaml",
        native_segmenter="grounded_sam2",
        native_depth="ensemble",
        peeling_config="configs/recursive_peeling.yaml",
        peeling_segmenter="grounded_sam2",
        peeling_depth="ensemble",
        qwen_resolution=640,
        qwen_steps=20,
        qwen_device="cuda",
        qwen_dtype="bfloat16",
        qwen_offload="sequential",
        qwen_model="Qwen/Qwen-Image-Layered",
        merge_external_layers=False,
    )

    labels = [spec.label for spec in specs]
    assert labels == [
        "LayerForge native",
        "LayerForge peeling",
        "Qwen raw (4)",
        "Qwen + graph preserve (4)",
        "Qwen + graph reorder (4)",
    ]
    assert specs[0].marker_name == "metrics.json"
    assert specs[1].command[:2] == [".venv/bin/layerforge", "peel"]
    assert "--preserve-external-order" in specs[3].command
    assert "--preserve-external-order" not in specs[4].command
    assert "--merge-external-layers" not in specs[3].command
    assert specs[2].run_dir == output_root / "sample" / "qwen_4"

