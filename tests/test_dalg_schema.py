from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from layerforge.dalg import build_dalg_manifest, export_dalg_manifest


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)


def _write_rgba(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (16, 12), color).save(path)


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "layers_ordered_rgba").mkdir(parents=True)
    (run_dir / "layers_albedo_rgba").mkdir()
    (run_dir / "layers_shading_rgba").mkdir()
    (run_dir / "layers_amodal_masks").mkdir()
    (run_dir / "debug").mkdir()

    _write_rgb(run_dir / "input.png", (240, 240, 240))
    _write_rgba(run_dir / "layers_ordered_rgba" / "000_foreground_person.png", (210, 90, 70, 255))
    _write_rgba(run_dir / "layers_albedo_rgba" / "000_foreground_person_albedo.png", (210, 90, 70, 255))
    _write_rgba(run_dir / "layers_shading_rgba" / "000_foreground_person_shading.png", (140, 140, 140, 255))
    Image.new("L", (16, 12), 255).save(run_dir / "layers_amodal_masks" / "000_foreground_person_amodal.png")
    _write_rgb(run_dir / "debug" / "recomposed_rgb.png", (230, 230, 230))

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
                        "bbox": [1, 2, 15, 11],
                        "occludes": [],
                        "occluded_by": [],
                        "metadata": {"source": "synthetic"},
                    }
                ],
                "occlusion_edges": [],
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
                "effect_layers": [],
                "debug": {"recomposed_rgb": "debug/recomposed_rgb.png"},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_dalg_manifest_matches_schema_contract(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    dalg = build_dalg_manifest(run_dir)
    schema = json.loads(Path("schemas/dalg.schema.json").read_text(encoding="utf-8"))

    assert dalg["$schema"] == schema["$id"]
    for key in schema["required"]:
        assert key in dalg

    node_ids = [layer["id"] for layer in dalg["graph"]["layers"]]
    assert len(node_ids) == len(set(node_ids))

    ranks = [layer["rank"] for layer in dalg["graph"]["layers"]]
    assert len(ranks) == len(set(ranks))

    for layer in dalg["graph"]["layers"]:
        assert isinstance(layer["paths"], dict)
        for path in layer["paths"].values():
            assert not str(path).startswith("/")
            assert (run_dir / path).exists()

    for layer in dalg["layers"]:
        if layer["path"] is not None:
            assert not str(layer["path"]).startswith("/")
            assert (run_dir / layer["path"]).exists()

    layer_ids = set(node_ids)
    for edge in dalg["graph"]["edges"]:
        assert edge["near_id"] in layer_ids
        assert edge["far_id"] in layer_ids


def test_exported_dalg_manifest_uses_canonical_schema_url(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    out_path = export_dalg_manifest(run_dir)
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload["$schema"] == "https://layerforge.dev/schemas/dalg.schema.json"
    assert payload["exports"]["source_manifest"] == "manifest.json"
