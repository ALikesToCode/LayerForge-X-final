from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from layerforge.gemini_io import decode_segmentation_item, parse_jsonish_text, prepare_gemini_image


def test_parse_jsonish_text_accepts_markdown_fence() -> None:
    text = """```json
    ["person", "car", "road"]
    ```"""
    assert parse_jsonish_text(text) == ["person", "car", "road"]


def test_decode_segmentation_item_restores_full_mask() -> None:
    mask = Image.fromarray(np.array([[0, 255], [255, 255]], dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    payload = {
        "label": "window",
        "box_2d": [250, 250, 750, 750],
        "mask": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
    }
    label, full_mask = decode_segmentation_item(payload, (8, 8))
    assert label == "window"
    assert full_mask.shape == (8, 8)
    assert full_mask.sum() > 0
    assert full_mask[0, 0] == 0


def test_prepare_gemini_image_resizes_large_inputs() -> None:
    image = Image.new("RGB", (2400, 1200), color=(10, 20, 30))
    prepared = prepare_gemini_image(image, 1024)
    assert max(prepared.size) == 1024
    assert prepared.size == (1024, 512)
