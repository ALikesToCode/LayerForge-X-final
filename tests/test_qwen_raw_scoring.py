from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from layerforge.qwen_io import score_raw_rgba_layers


def test_score_qwen_raw_layers_writes_metrics_and_recomposition(tmp_path: Path) -> None:
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

    Image.fromarray(far, mode="RGBA").save(layers_dir / "00.png")
    Image.fromarray(near, mode="RGBA").save(layers_dir / "01.png")
    (layers_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": str(image_path),
                "model": "Qwen/Qwen-Image-Layered",
                "resolution": 640,
                "num_inference_steps": 10,
                "offload": "sequential",
                "layer_paths": [str(layers_dir / "00.png"), str(layers_dir / "01.png")],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_path, recomposed_path = score_raw_rgba_layers(image_path, layers_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert recomposed_path.exists()
    assert (layers_dir / "recomposed_rgba.png").exists()
    assert (layers_dir / "ordered_layer_contact_sheet.png").exists()
    assert metrics["mode"] == "qwen_raw_rgba"
    assert metrics["ordering_assumption"] == "manifest_order_interpreted_as_far_to_near"
    assert metrics["num_layers"] == 2.0
    assert metrics["recompose_psnr"] > 40.0
