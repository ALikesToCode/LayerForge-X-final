from __future__ import annotations

import json
from pathlib import Path

import pytest

from layerforge.pipeline import LayerForgePipeline


@pytest.mark.slow
def test_run_exports_unique_ordered_layer_paths_and_stable_manifest(tmp_path) -> None:
    pipe = LayerForgePipeline("configs/fast.yaml")
    outputs = pipe.run(
        "examples/synth/scene_000/image.png",
        tmp_path / "run",
        segmenter="classical",
        depth_method="geometric_luminance",
        save_parallax=False,
    )

    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    names = [Path(item["path"]).name for item in manifest["ordered_layers_near_to_far"]]
    ranks = [int(item["rank"]) for item in manifest["ordered_layers_near_to_far"]]
    exported = sorted((tmp_path / "run" / "layers_ordered_rgba").glob("*.png"))

    assert names
    assert len(names) == len(set(names))
    assert len(exported) == len(names)
    assert ranks == list(range(len(ranks)))
