from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_make_synthetic_dataset_layerbench_pp_writes_rich_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "layerbench_pp"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_synthetic_dataset.py",
            "--output",
            str(out_dir),
            "--count",
            "1",
            "--width",
            "256",
            "--height",
            "192",
            "--output-format",
            "layerbench_pp",
            "--with-effects",
        ],
        check=True,
    )

    scene_dir = out_dir / "scene_000"
    metadata = json.loads((scene_dir / "scene_metadata.json").read_text(encoding="utf-8"))
    occlusion = json.loads((scene_dir / "occlusion_graph.json").read_text(encoding="utf-8"))

    assert (scene_dir / "image.png").exists()
    assert (scene_dir / "ground_truth.json").exists()
    assert (scene_dir / "visible_masks").is_dir()
    assert (scene_dir / "amodal_masks").is_dir()
    assert (scene_dir / "alpha_mattes").is_dir()
    assert (scene_dir / "layers_effects_rgba").is_dir()
    assert (scene_dir / "intrinsics" / "albedo.png").exists()
    assert (scene_dir / "intrinsics" / "shading.png").exists()
    assert (scene_dir / "depth.png").exists()
    assert (scene_dir / "depth.npy").exists()
    assert metadata["output_format"] == "layerbench_pp"
    assert metadata["with_effects"] is True
    assert any(layer.get("kind") == "effect" for layer in metadata["layers_near_to_far"])
    assert "edges" in occlusion
    assert len(list((scene_dir / "visible_masks").glob("*.png"))) == len(metadata["layers_near_to_far"])
