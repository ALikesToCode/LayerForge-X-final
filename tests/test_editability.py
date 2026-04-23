from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.editability import evaluate_run_editability, export_target_assets


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
