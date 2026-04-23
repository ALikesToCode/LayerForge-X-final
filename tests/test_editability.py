from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.editability import evaluate_run_editability, export_target_assets, select_editable_layer


def _write_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _make_run(tmp_path: Path, *, copy_like_background: bool) -> Path:
    image = np.full((32, 32, 3), 240, dtype=np.uint8)
    image[8:24, 10:22] = np.array([210, 70, 60], dtype=np.uint8)
    image_path = tmp_path / ("copy_input.png" if copy_like_background else "honest_input.png")
    Image.fromarray(image, mode="RGB").save(image_path)

    run_dir = tmp_path / ("copy_run" if copy_like_background else "honest_run")
    run_dir.mkdir()
    bg = np.zeros((32, 32, 4), dtype=np.uint8)
    bg[..., 3] = 255
    bg[..., :3] = image if copy_like_background else 240

    fg = np.zeros((32, 32, 4), dtype=np.uint8)
    fg[8:24, 10:22, :3] = np.array([210, 70, 60], dtype=np.uint8)
    fg[8:24, 10:22, 3] = 255

    _write_rgba(run_dir / "background.png", bg)
    _write_rgba(run_dir / "red_car.png", fg)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": str(image_path),
                "ordered_layers_near_to_far": [
                    {
                        "path": str(run_dir / "red_car.png"),
                        "name": "000_red_car",
                        "label": "red car",
                        "group": "vehicle",
                        "rank": 0,
                    },
                    {
                        "path": str(run_dir / "background.png"),
                        "name": "001_background_layer",
                        "label": "background",
                        "group": "background",
                        "rank": 1,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_evaluate_run_editability_penalizes_copy_like_background(tmp_path: Path) -> None:
    honest_run = _make_run(tmp_path, copy_like_background=False)
    copy_run = _make_run(tmp_path, copy_like_background=True)

    honest = evaluate_run_editability(honest_run)
    copy_like = evaluate_run_editability(copy_run)

    assert honest["edit_success_score"] > copy_like["edit_success_score"]
    assert honest["background_hole_ratio"] < copy_like["background_hole_ratio"]
    assert not honest["preview_paths"]["baseline"].startswith("/")
    assert honest["preview_paths"]["baseline"] == "debug/edit_baseline.png"


def test_export_target_assets_selects_prompt_named_layer(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, copy_like_background=False)
    exported = export_target_assets(run_dir, output_dir=tmp_path / "extract", prompt="red car")

    assert exported["selected_target"]["name"] == "000_red_car"
    assert (tmp_path / "extract" / "target_rgba.png").exists()
    assert (tmp_path / "extract" / "background_completed.png").exists()
    assert (tmp_path / "extract" / "target_metadata.json").exists()
    assert exported["exports"]["target_rgba"] == "target_rgba.png"
    assert not exported["exports"]["edit_preview_move"].startswith("/")


def test_select_editable_layer_can_use_gemini_selector(monkeypatch) -> None:
    def make_layer(name: str, label: str, x0: int, x1: int, color: tuple[int, int, int]) -> dict[str, object]:
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[8:24, x0:x1, :3] = np.array(color, dtype=np.uint8)
        rgba[8:24, x0:x1, 3] = 255
        mask = rgba[..., 3] > 0
        return {
            "name": name,
            "label": label,
            "group": "vehicle",
            "rank": 0,
            "rgba": rgba,
            "alpha": rgba[..., 3].astype(np.float32) / 255.0,
            "mask": mask,
            "bbox": (x0, 8, x1, 24),
        }

    layers = [
        make_layer("000_red_car", "red car", 4, 14, (220, 60, 50)),
        make_layer("001_blue_bike", "blue bike", 18, 28, (50, 80, 220)),
    ]

    monkeypatch.setattr(
        "layerforge.editability._gemini_select_layer_name",
        lambda *args, **kwargs: "001_blue_bike",
    )

    selected = select_editable_layer(
        layers,
        prompt="blue bike",
        cfg={"backend": "gemini", "candidate_limit": 2},
    )

    assert selected is not None
    assert selected["name"] == "001_blue_bike"


def test_select_editable_layer_auto_falls_back_to_heuristic_on_gemini_error(monkeypatch) -> None:
    rgba = np.zeros((24, 24, 4), dtype=np.uint8)
    rgba[4:20, 4:20, :3] = np.array([200, 60, 50], dtype=np.uint8)
    rgba[4:20, 4:20, 3] = 255
    mask = rgba[..., 3] > 0
    layer = {
        "name": "000_red_car",
        "label": "red car",
        "group": "vehicle",
        "rank": 0,
        "rgba": rgba,
        "alpha": rgba[..., 3].astype(np.float32) / 255.0,
        "mask": mask,
        "bbox": (4, 4, 20, 20),
    }

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("gemini unavailable")

    monkeypatch.setattr("layerforge.editability._gemini_select_layer_name", boom)

    selected = select_editable_layer(
        [layer],
        prompt="red car",
        cfg={"backend": "auto", "candidate_limit": 1},
    )

    assert selected is not None
    assert selected["name"] == "000_red_car"


def test_export_target_assets_point_only_records_inferred_semantic_identity(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path, copy_like_background=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["ordered_layers_near_to_far"][0]["name"] = "000_object_region"
    manifest["ordered_layers_near_to_far"][0]["label"] = "object region"
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        "layerforge.editability._gemini_infer_target_prompt",
        lambda *args, **kwargs: "red car",
    )

    exported = export_target_assets(run_dir, output_dir=tmp_path / "extract_point", point=(16, 16))

    assert exported["selected_target"]["name"] == "000_object_region"
    assert exported["selected_target"]["semantic_label"] == "red car"
    assert exported["selected_target"]["semantic_name"] == "red_car"
    assert exported["resolved_prompt"] == "red car"
