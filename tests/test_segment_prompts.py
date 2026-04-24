from __future__ import annotations

import pytest
import numpy as np
from PIL import Image

from layerforge.segment import merge_prompt_labels, segment_image


def test_resolve_grounded_sam2_prompts_uses_gemini_for_auto_without_manual_prompts(monkeypatch) -> None:
    from layerforge import segment as segment_mod

    pil = Image.new("RGB", (8, 8), color="black")
    monkeypatch.setattr(segment_mod, "_gemini_suggest_labels", lambda _pil, _cfg: ["plane", "tree"])
    prompts = segment_mod._resolve_grounded_sam2_prompts(pil, {"prompt_source": "auto", "prompts": []})
    assert prompts == ["plane", "tree"]


def test_resolve_grounded_sam2_prompts_raises_for_explicit_gemini_without_fallback(monkeypatch) -> None:
    from layerforge import segment as segment_mod

    pil = Image.new("RGB", (8, 8), color="black")

    def boom(_pil, _cfg):
        raise RuntimeError("gemini offline")

    monkeypatch.setattr(segment_mod, "_gemini_suggest_labels", boom)
    with pytest.raises(RuntimeError, match="gemini offline"):
        segment_mod._resolve_grounded_sam2_prompts(pil, {"prompt_source": "gemini", "prompts": []})


def test_resolve_grounded_sam2_prompts_falls_back_to_explicit_manual_prompts_on_gemini_error(monkeypatch) -> None:
    from layerforge import segment as segment_mod

    pil = Image.new("RGB", (8, 8), color="black")

    def boom(_pil, _cfg):
        raise RuntimeError("gemini offline")

    monkeypatch.setattr(segment_mod, "_gemini_suggest_labels", boom)
    prompts = segment_mod._resolve_grounded_sam2_prompts(pil, {"prompt_source": "gemini", "prompts": ["truck", "road"]})
    assert prompts == ["truck", "road"]


def test_merge_prompt_labels_keeps_manual_order_and_deduplicates() -> None:
    merged = merge_prompt_labels(["truck", "road", "car"], ["vehicle", "road", "window", "truck"])
    assert merged == ["truck", "road", "car", "vehicle", "window"]


def test_segment_image_falls_back_to_classical_when_optional_backend_fails(monkeypatch) -> None:
    from layerforge import segment as segment_mod

    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")

    def boom(_pil, _cfg, _device):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(segment_mod, "grounded_sam2_segments", boom)
    with pytest.warns(RuntimeWarning, match="falling back to classical"):
        segments = segment_image(rgb, pil, {"method": "grounded_sam2", "min_area_ratio": 0.001}, device="cpu")

    assert segments
    assert any(seg.source == "classical-grid" for seg in segments)
    assert any(seg.metadata.get("fallback_from") == "grounded_sam2" for seg in segments)
