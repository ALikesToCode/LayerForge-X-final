from __future__ import annotations

import json

import pytest

from layerforge.dalg import build_dalg_manifest, export_dalg_manifest
from layerforge.pipeline import LayerForgePipeline


@pytest.mark.slow
def test_run_writes_canonical_dalg_manifest(tmp_path) -> None:
    pipe = LayerForgePipeline("configs/fast.yaml")
    outputs = pipe.run(
        "examples/synth/scene_000/image.png",
        tmp_path / "run",
        segmenter="classical",
        depth_method="geometric_luminance",
        save_parallax=False,
    )

    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    dalg_path = tmp_path / "run" / "dalg_manifest.json"
    dalg = json.loads(dalg_path.read_text(encoding="utf-8"))

    assert manifest["canonical_dalg"] == str(dalg_path)
    assert dalg["kind"] == "layerforge.dalg"
    assert dalg["graph"]["ordering"] == "near_to_far"
    assert dalg["graph"]["node_count"] == len(dalg["graph"]["layers"])
    assert dalg["exports"]["source_manifest"] == "manifest.json"
    assert dalg["asset"]["run_dir"] == "."


def test_export_dalg_manifest_cli_shape_from_existing_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "layers_ordered_rgba").mkdir()
    (run_dir / "layers_albedo_rgba").mkdir()
    (run_dir / "layers_shading_rgba").mkdir()
    (run_dir / "layers_amodal_masks").mkdir()
    (run_dir / "debug").mkdir()

    (run_dir / "layers_ordered_rgba" / "000_foreground_person.png").write_bytes(b"png")
    (run_dir / "layers_albedo_rgba" / "000_foreground_person_albedo.png").write_bytes(b"png")
    (run_dir / "layers_shading_rgba" / "000_foreground_person_shading.png").write_bytes(b"png")
    (run_dir / "layers_amodal_masks" / "000_foreground_person_amodal.png").write_bytes(b"png")

    (run_dir / "metrics.json").write_text(
        json.dumps({"segmentation_method": "classical", "depth_method": "geometric_luminance"}),
        encoding="utf-8",
    )
    (run_dir / "debug" / "layer_graph.json").write_text(
        json.dumps(
            {
                "layers_near_to_far": [
                    {
                        "rank": 0,
                        "name": "000_foreground_person",
                        "label": "person",
                        "group": "foreground",
                        "depth_median": 0.1,
                        "area": 42,
                        "bbox": [1, 2, 3, 4],
                        "occludes": [1],
                        "occluded_by": [],
                        "metadata": {"source": "synthetic"},
                    }
                ],
                "occlusion_edges": [{"near_id": 0, "far_id": 1, "confidence": 0.9}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": "input.png",
                "metrics": "metrics.json",
                "layer_graph": "debug/layer_graph.json",
                "ordered_layers_near_to_far": [
                    {
                        "path": "layers_ordered_rgba/000_foreground_person.png",
                        "name": "000_foreground_person",
                        "rank": 0,
                        "label": "person",
                        "group": "foreground",
                        "depth_median": 0.1,
                    }
                ],
                "grouped_layers": [],
                "debug": {"recomposed_rgb": "debug/recomposed_rgb.png"},
            }
        ),
        encoding="utf-8",
    )

    dalg_path = export_dalg_manifest(run_dir)
    dalg = build_dalg_manifest(run_dir)

    assert dalg_path == run_dir / "dalg_manifest.json"
    assert dalg["graph"]["edge_count"] == 1
    assert dalg["graph"]["layers"][0]["paths"]["rgba"] == "layers_ordered_rgba/000_foreground_person.png"
    assert dalg["graph"]["layers"][0]["metadata"]["source"] == "synthetic"
