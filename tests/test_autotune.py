from __future__ import annotations

from layerforge.autotune import build_autotune_candidates, candidate_rank_key


def test_build_autotune_candidates_for_grounded_sam2_with_prompts() -> None:
    names = [candidate.name for candidate in build_autotune_candidates("grounded_sam2", ["truck", "road"])]
    assert "manual_balanced" in names
    assert "augment_balanced" in names
    assert "gemini_balanced" in names
    assert "gemini_precision" not in names


def test_candidate_rank_key_prefers_psnr_then_ssim_then_fewer_layers() -> None:
    a = {"recompose_psnr": 30.5, "recompose_ssim": 0.98, "num_layers": 25}
    b = {"recompose_psnr": 30.5, "recompose_ssim": 0.98, "num_layers": 19}
    c = {"recompose_psnr": 31.0, "recompose_ssim": 0.97, "num_layers": 30}
    assert candidate_rank_key(c) > candidate_rank_key(b)
    assert candidate_rank_key(b) > candidate_rank_key(a)
