from __future__ import annotations

import json

import numpy as np
from PIL import Image

from layerforge.validation import validate_run_outputs


def _rgba(path, alpha: int = 255) -> None:
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[..., :3] = 120
    arr[..., 3] = alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGBA").save(path)


def test_validate_run_outputs_reports_ok_for_minimal_valid_run(tmp_path) -> None:
    run = tmp_path / "run"
    _rgba(run / "layers_ordered_rgba" / "000_object.png")
    _rgba(run / "layers_alpha" / "000_object_alpha.png")
    (run / "layers_alpha_confidence").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), 220, dtype=np.uint8), mode="L").save(run / "layers_alpha_confidence" / "000_object_alpha_confidence.png")
    _rgba(run / "layers_completed_rgba" / "000_object_completed.png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(run / "input.png")
    (run / "debug").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(run / "debug" / "recomposed_rgb.png")
    (run / "debug" / "layer_graph.json").write_text(
        json.dumps(
            {
                "occlusion_edges": [
                    {
                        "near_id": 0,
                        "far_id": 1,
                        "confidence": 0.8,
                        "relation": "in_front_of",
                        "evidence": {"boundary_depth_delta": 0.2},
                    }
                ],
                "segment_nodes": [{"segment_id": 0, "removed_edges": []}],
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "input": "input.png",
                "layer_graph": "debug/layer_graph.json",
                "ordered_layers_near_to_far": [
                    {
                        "path": "layers_ordered_rgba/000_object.png",
                        "alpha_path": "layers_alpha/000_object_alpha.png",
                        "alpha_confidence_path": "layers_alpha_confidence/000_object_alpha_confidence.png",
                        "completed_path": "layers_completed_rgba/000_object_completed.png",
                    }
                ],
                "debug": {"recomposed_rgb": "debug/recomposed_rgb.png"},
            }
        ),
        encoding="utf-8",
    )

    report = validate_run_outputs(run)

    assert report["ok"] is True
    assert report["metrics"]["recomposition_residual"] == 0.0


def test_validate_run_outputs_reports_missing_layer_path(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"ordered_layers_near_to_far": [{"path": "missing.png"}], "debug": {}}),
        encoding="utf-8",
    )

    report = validate_run_outputs(run)

    assert report["ok"] is False
    assert "missing layer file path" in report["errors"][0]


def test_validate_run_outputs_requires_graph_edge_evidence(tmp_path) -> None:
    run = tmp_path / "run"
    _rgba(run / "layers_ordered_rgba" / "000_object.png")
    (run / "debug").mkdir(parents=True, exist_ok=True)
    (run / "debug" / "layer_graph.json").write_text(
        json.dumps(
            {
                "occlusion_edges": [{"near_id": 0, "far_id": 1, "confidence": 0.8, "relation": "in_front_of"}],
                "segment_nodes": [],
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "layer_graph": "debug/layer_graph.json",
                "ordered_layers_near_to_far": [{"path": "layers_ordered_rgba/000_object.png"}],
                "debug": {},
            }
        ),
        encoding="utf-8",
    )

    report = validate_run_outputs(run)

    assert report["ok"] is False
    assert any("graph edge missing evidence" in error for error in report["errors"])
