from __future__ import annotations

import pytest
from PIL import Image

from layerforge.segment import merge_prompt_labels


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
