from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import layerforge.qwen_io as qwen_io
from layerforge.pipeline import LayerForgePipeline


def _write_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def test_enrich_qwen_preserve_and_reorder_respect_manifest_and_disable_merge_by_default(tmp_path: Path) -> None:
    image = np.full((32, 48, 3), 220, dtype=np.uint8)
    image[8:24, 8:24] = np.array([60, 90, 180], dtype=np.uint8)
    image[14:28, 20:36] = np.array([220, 90, 60], dtype=np.uint8)
    image_path = tmp_path / "input.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    layers_dir = tmp_path / "qwen_layers"
    layers_dir.mkdir()

    far = np.zeros((32, 48, 4), dtype=np.uint8)
    far[..., :3] = 220
    far[..., 3] = 255
    far[8:24, 8:24, :3] = np.array([60, 90, 180], dtype=np.uint8)

    near = np.zeros((32, 48, 4), dtype=np.uint8)
    near[14:28, 20:36, :3] = np.array([220, 90, 60], dtype=np.uint8)
    near[14:28, 20:36, 3] = 255

    stray = np.zeros((32, 48, 4), dtype=np.uint8)
    stray[2:10, 2:10, :3] = np.array([20, 200, 80], dtype=np.uint8)
    stray[2:10, 2:10, 3] = 255

    _write_rgba(layers_dir / "00.png", far)
    _write_rgba(layers_dir / "01.png", near)
    _write_rgba(layers_dir / "extra_debug.png", stray)
    (layers_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": str(image_path),
                "model": "Qwen/Qwen-Image-Layered",
                "layer_paths": [str(layers_dir / "00.png"), str(layers_dir / "01.png")],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pipe = LayerForgePipeline("configs/fast.yaml")

    preserve_outputs = pipe.enrich_rgba_layers(
        image_path,
        layers_dir,
        tmp_path / "preserve",
        depth_method="geometric_luminance",
        preserve_external_order=True,
    )
    reorder_outputs = pipe.enrich_rgba_layers(
        image_path,
        layers_dir,
        tmp_path / "reorder",
        depth_method="geometric_luminance",
        preserve_external_order=False,
    )

    preserve_metrics = json.loads(preserve_outputs.metrics_path.read_text(encoding="utf-8"))
    preserve_manifest = json.loads(preserve_outputs.manifest_path.read_text(encoding="utf-8"))
    reorder_metrics = json.loads(reorder_outputs.metrics_path.read_text(encoding="utf-8"))

    assert preserve_metrics["mode"] == "external_rgba_enrichment"
    assert preserve_metrics["visual_order_mode"] == "preserve_external_order"
    assert preserve_metrics["preserve_external_order"] is True
    assert preserve_metrics["merge_external_layers"] is False
    assert preserve_metrics["num_layers"] == 2.0
    assert preserve_metrics["premerge_semantic_layers"] == 2.0
    assert preserve_metrics["merge_reduction"] == 0.0
    assert preserve_metrics["selected_external_visual_order"] in {"manifest_order", "reversed_manifest_order"}
    assert len(preserve_manifest["ordered_layers_near_to_far"]) == 2
    assert len(preserve_manifest["external_manifest"]["layer_paths"]) == 2
    assert all("extra_debug.png" not in item["path"] for item in preserve_manifest["ordered_layers_near_to_far"])

    assert reorder_metrics["visual_order_mode"] == "graph_order"
    assert reorder_metrics["preserve_external_order"] is False
    assert reorder_metrics["graph_order_available"] is True
    assert reorder_metrics["merge_external_layers"] is False
    assert reorder_metrics["num_layers"] == 2.0


def test_enrich_qwen_reorder_guardrail_falls_back_when_graph_order_is_much_worse(tmp_path: Path, monkeypatch) -> None:
    image = np.full((24, 24, 3), 220, dtype=np.uint8)
    image[4:16, 4:16] = np.array([40, 80, 180], dtype=np.uint8)
    image[8:20, 8:20] = np.array([220, 90, 60], dtype=np.uint8)
    image_path = tmp_path / "input.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    layers_dir = tmp_path / "qwen_layers"
    layers_dir.mkdir()

    far = np.zeros((24, 24, 4), dtype=np.uint8)
    far[..., :3] = 220
    far[..., 3] = 255
    far[4:16, 4:16, :3] = np.array([40, 80, 180], dtype=np.uint8)

    near = np.zeros((24, 24, 4), dtype=np.uint8)
    near[8:20, 8:20, :3] = np.array([220, 90, 60], dtype=np.uint8)
    near[8:20, 8:20, 3] = 255

    _write_rgba(layers_dir / "00.png", far)
    _write_rgba(layers_dir / "01.png", near)
    (layers_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": str(image_path),
                "model": "Qwen/Qwen-Image-Layered",
                "layer_paths": [str(layers_dir / "00.png"), str(layers_dir / "01.png")],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_pick_best_visual_order(*args, **kwargs):
        orders = {"manifest_order": [0, 1], "reversed_manifest_order": [1, 0]}
        scores = {
            "manifest_order": {"recompose_psnr": 30.0, "recompose_ssim": 0.90},
            "reversed_manifest_order": {"recompose_psnr": 12.0, "recompose_ssim": 0.55},
        }
        return "manifest_order", orders, scores

    def fake_topo_order(nodes):
        return [1, 0]

    def fake_score_layer_stack(rgb, layers):
        order = [layer.label for layer in layers]
        if order == ["01", "00"]:
            return {"recompose_psnr": 12.0, "recompose_ssim": 0.55, "num_layers": float(len(layers))}
        return {"recompose_psnr": 30.0, "recompose_ssim": 0.90, "num_layers": float(len(layers))}

    monkeypatch.setattr(qwen_io, "_pick_best_visual_order", fake_pick_best_visual_order)
    monkeypatch.setattr(qwen_io, "topo_order", fake_topo_order)
    monkeypatch.setattr(qwen_io, "_score_layer_stack", fake_score_layer_stack)

    pipe = LayerForgePipeline("configs/fast.yaml")
    outputs = pipe.enrich_rgba_layers(
        image_path,
        layers_dir,
        tmp_path / "reorder_guarded",
        depth_method="geometric_luminance",
        preserve_external_order=False,
    )

    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))

    assert metrics["graph_order_available"] is True
    assert metrics["graph_order_psnr"] == 12.0
    assert metrics["selected_external_order_psnr"] == 30.0
    assert metrics["graph_order_applied"] is False
    assert metrics["graph_order_guardrail_triggered"] is True
    assert metrics["visual_order_mode"] == "graph_order_guarded_fallback"
    ordered_labels = [item["label"] for item in manifest["ordered_layers_near_to_far"]]
    assert ordered_labels == ["00", "01"]
