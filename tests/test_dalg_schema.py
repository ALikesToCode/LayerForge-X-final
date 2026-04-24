from __future__ import annotations

import json
from pathlib import Path

from layerforge.dalg import CANONICAL_DALG_SCHEMA_URL, build_dalg_manifest, export_dalg_manifest, load_dalg_manifest, validate_dalg_manifest


def _write_png_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def test_dalg_manifest_matches_canonical_schema_contract(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_png_stub(run_dir / "input.png")
    _write_png_stub(run_dir / "layers_ordered_rgba" / "000_vehicle_car.png")
    _write_png_stub(run_dir / "layers_ordered_rgba" / "001_background_completed.png")
    _write_png_stub(run_dir / "layers_alpha" / "000_vehicle_car_alpha.png")
    _write_png_stub(run_dir / "layers_alpha" / "001_background_completed_alpha.png")
    _write_png_stub(run_dir / "layers_albedo_rgba" / "000_vehicle_car_albedo.png")
    _write_png_stub(run_dir / "layers_shading_rgba" / "000_vehicle_car_shading.png")
    _write_png_stub(run_dir / "layers_amodal_masks" / "000_vehicle_car_amodal.png")
    _write_png_stub(run_dir / "layers_albedo_rgba" / "001_background_completed_albedo.png")
    _write_png_stub(run_dir / "layers_shading_rgba" / "001_background_completed_shading.png")

    (run_dir / "metrics.json").write_text(
        json.dumps({"segmentation_method": "grounded_sam2", "depth_method": "ensemble", "intrinsic_method": "retinex"}),
        encoding="utf-8",
    )
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    (run_dir / "debug" / "layer_graph.json").write_text(
        json.dumps(
            {
                "layers_near_to_far": [
                    {
                        "rank": 0,
                        "name": "000_vehicle_car",
                        "label": "car",
                        "group": "vehicle",
                        "depth_median": 0.12,
                        "area": 1200,
                        "bbox": [4, 6, 44, 48],
                        "occludes": [1],
                        "occluded_by": [],
                    },
                    {
                        "rank": 1,
                        "name": "001_background_completed",
                        "label": "background completed",
                        "group": "background",
                        "depth_median": 0.88,
                        "area": 9000,
                        "bbox": [0, 0, 64, 64],
                        "occludes": [],
                        "occluded_by": [0],
                    },
                ],
                "occlusion_edges": [{"near_id": 0, "far_id": 1, "confidence": 0.91}],
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
                        "path": "layers_ordered_rgba/000_vehicle_car.png",
                        "name": "000_vehicle_car",
                        "rank": 0,
                        "label": "car",
                        "group": "vehicle",
                        "depth_median": 0.12,
                    },
                    {
                        "path": "layers_ordered_rgba/001_background_completed.png",
                        "name": "001_background_completed",
                        "rank": 1,
                        "label": "background completed",
                        "group": "background",
                        "depth_median": 0.88,
                    },
                ],
                "grouped_layers": [],
                "debug": {"recomposed_rgb": "debug/recomposed_rgb.png"},
            }
        ),
        encoding="utf-8",
    )

    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "dalg.schema.json").read_text(encoding="utf-8"))
    export_dalg_manifest(run_dir)
    dalg = build_dalg_manifest(run_dir)

    assert dalg["$schema"] == CANONICAL_DALG_SCHEMA_URL
    assert dalg["$schema"] == schema["properties"]["$schema"]["const"]
    assert dalg["kind"] == schema["properties"]["kind"]["const"]
    assert dalg["dalg_version"] == "1.1"
    assert dalg["alpha_mode"] == "straight"
    assert dalg["color_space"] == "sRGB"
    assert dalg["config_hash"]
    assert dalg["graph"]["node_count"] == len(dalg["graph"]["layers"])
    assert dalg["graph"]["edge_count"] == len(dalg["graph"]["edges"])
    assert dalg["design_export"]["layer_order"] == "near_to_far"
    assert "vehicle" in dalg["design_export"]["semantic_groups"]
    assert "alpha_masks" in dalg["design_export"]["includes"]
    assert validate_dalg_manifest(dalg, run_dir) == []

    ids = [layer["id"] for layer in dalg["graph"]["layers"]]
    ranks = [layer["rank"] for layer in dalg["graph"]["layers"]]
    names = [layer["name"] for layer in dalg["graph"]["layers"]]

    assert len(ids) == len(set(ids))
    assert len(ranks) == len(set(ranks))
    assert len(names) == len(set(names))

    valid_ids = set(ids)
    for edge in dalg["graph"]["edges"]:
        assert edge["near_id"] in valid_ids
        assert edge["far_id"] in valid_ids
        assert edge["relation"] == "in_front_of"
        assert "boundary_depth_delta" in edge["evidence"]

    for layer in dalg["graph"]["layers"]:
        assert "depth_stats" in layer
        assert "semantic_labels" in layer
        assert "provenance" in layer
        assert "quality_metrics" in layer
        for rel_path in layer["paths"].values():
            assert (run_dir / rel_path).exists(), rel_path


def test_load_dalg_manifest_migrates_v1_manifest(tmp_path) -> None:
    path = tmp_path / "dalg_v1.json"
    path.write_text(
        json.dumps(
            {
                "$schema": CANONICAL_DALG_SCHEMA_URL,
                "kind": "layerforge.dalg",
                "schema_version": "1.0.0",
                "canvas": None,
                "asset": {"input": None, "run_dir": ".", "mode": "native_layerforge"},
                "recipe": {},
                "graph": {"ordering": "near_to_far", "graph_order_available": False, "node_count": 0, "edge_count": 0, "layers": [], "edges": []},
                "layers": [],
                "metrics": {},
                "exports": {"source_manifest": "manifest.json", "metrics": None, "grouped_layers": [], "effect_layers": [], "debug": {}},
            }
        ),
        encoding="utf-8",
    )

    migrated = load_dalg_manifest(path)

    assert migrated["schema_version"] == "1.0.0"
    assert migrated["dalg_version"] == "1.0"
