from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from layerforge.cli import main
from layerforge.design_export import build_design_manifest, export_design_assets, export_design_manifest, export_psd


def _write_png(path: Path, color: tuple[int, int, int, int], size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def _write_mask(path: Path, value: int, size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, value).save(path)


def _make_design_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    for name in (
        "layers_ordered_rgba",
        "layers_alpha",
        "layers_alpha_confidence",
        "layers_completed_rgba",
        "layers_albedo_rgba",
        "layers_shading_rgba",
        "layers_amodal_masks",
        "layers_hidden_masks",
        "debug",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    _write_png(run_dir / "input.png", (20, 30, 40, 255))
    _write_png(run_dir / "layers_ordered_rgba" / "000_person.png", (220, 20, 20, 160))
    _write_png(run_dir / "layers_ordered_rgba" / "001_background.png", (20, 30, 40, 255))
    _write_mask(run_dir / "layers_alpha" / "000_person_alpha.png", 160)
    _write_mask(run_dir / "layers_alpha" / "001_background_alpha.png", 255)
    _write_mask(run_dir / "layers_alpha_confidence" / "000_person_alpha_confidence.png", 220)
    _write_png(run_dir / "layers_completed_rgba" / "000_person_completed.png", (230, 30, 30, 255))
    _write_png(run_dir / "layers_albedo_rgba" / "000_person_albedo.png", (180, 30, 30, 255))
    _write_png(run_dir / "layers_shading_rgba" / "000_person_shading.png", (180, 180, 180, 255))
    _write_mask(run_dir / "layers_amodal_masks" / "000_person_amodal.png", 220)
    _write_mask(run_dir / "layers_hidden_masks" / "000_person_hidden.png", 64)

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
                        "name": "000_person",
                        "label": "person",
                        "group": "people",
                        "depth_median": 0.2,
                        "area": 50,
                        "bbox": [2, 2, 8, 10],
                        "occludes": [1],
                        "occluded_by": [],
                        "metadata": {"source": "synthetic", "hidden_area_ratio": 0.1},
                    },
                    {
                        "rank": 1,
                        "name": "001_background",
                        "label": "background",
                        "group": "background/stuff",
                        "depth_median": 0.8,
                        "area": 192,
                        "bbox": [0, 0, 16, 12],
                        "occludes": [],
                        "occluded_by": [0],
                        "metadata": {"source": "background"},
                    },
                ],
                "occlusion_edges": [
                    {
                        "near_id": 0,
                        "far_id": 1,
                        "relation": "in_front_of",
                        "confidence": 0.9,
                        "overlap_ratio": 0.25,
                    }
                ],
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
                        "path": "layers_ordered_rgba/000_person.png",
                        "name": "000_person",
                        "rank": 0,
                        "label": "person",
                        "group": "people",
                        "depth_median": 0.2,
                    },
                    {
                        "path": "layers_ordered_rgba/001_background.png",
                        "name": "001_background",
                        "rank": 1,
                        "label": "background",
                        "group": "background/stuff",
                        "depth_median": 0.8,
                    },
                ],
                "grouped_layers": [],
                "debug": {},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_design_manifest_projects_dalg_layers_and_groups(tmp_path: Path) -> None:
    run_dir = _make_design_run(tmp_path)

    design_path = export_design_manifest(run_dir)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    rebuilt = build_design_manifest(run_dir)

    assert design["format"] == "layerforge.design_manifest"
    assert design["layer_order"] == "near_to_far"
    assert design["canvas"] == {"height": 12, "width": 16}
    assert design["semantic_groups"]["people"] == ["000_person"]
    assert design["layers"][0]["alpha_mask"] == "layers_alpha/000_person_alpha.png"
    assert design["layers"][0]["alpha_confidence_path"] == "layers_alpha_confidence/000_person_alpha_confidence.png"
    assert design["layers"][0]["hidden_mask"] == "layers_hidden_masks/000_person_hidden.png"
    assert rebuilt["exports"]["dalg"] == "dalg_manifest.json"


def test_export_design_cli_preserves_dalg_default_and_writes_all_formats(tmp_path: Path) -> None:
    pytest.importorskip("psd_tools")
    run_dir = _make_design_run(tmp_path)

    assert main(["export-design", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "dalg_manifest.json").exists()

    assert main(["export-design", "--run-dir", str(run_dir), "--format", "all"]) == 0
    assert (run_dir / "design_manifest.json").exists()
    assert (run_dir / "layers.psd").exists()


def test_export_psd_creates_semantic_groups_and_support_sublayers(tmp_path: Path) -> None:
    PSDImage = pytest.importorskip("psd_tools").PSDImage
    run_dir = _make_design_run(tmp_path)

    outputs = export_design_assets(run_dir, include_design_json=True, include_psd=True)
    psd_path = export_psd(run_dir)
    psd = PSDImage.open(psd_path)
    group_names = [layer.name for layer in psd]
    first_group_children = [layer.name for layer in psd[0]]

    assert outputs["design_json"] == run_dir / "design_manifest.json"
    assert outputs["psd"] == run_dir / "layers.psd"
    assert psd_path == run_dir / "layers.psd"
    assert "people" in group_names
    assert "000 000_person" in first_group_children
    assert "000 000_person / alpha mask" in first_group_children
    assert "000 000_person / alpha confidence" in first_group_children
