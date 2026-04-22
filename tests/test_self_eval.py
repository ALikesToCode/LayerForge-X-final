from __future__ import annotations

from layerforge.self_eval import choose_best_candidates


def test_choose_best_candidates_prefers_editable_candidate_over_copy_like_fidelity() -> None:
    rows = [
        {
            "image": "demo/truck.png",
            "label": "Copy-like background stack",
            "status": "ok",
            "num_layers": 4,
            "recompose_psnr": 36.0,
            "recompose_ssim": 0.995,
            "has_graph": True,
            "has_ordered_layers": True,
            "effect_layer_count": 0,
            "duration_sec": 20.0,
            "semantic_purity": 0.22,
            "alpha_quality_score": 0.18,
            "edit_success_score": 0.10,
            "background_hole_ratio": 0.95,
            "non_edited_region_preservation": 0.98,
            "occlusion_edge_count": 2,
        },
        {
            "image": "demo/truck.png",
            "label": "Structured but middling hybrid",
            "status": "ok",
            "num_layers": 4,
            "recompose_psnr": 28.7,
            "recompose_ssim": 0.876,
            "has_graph": True,
            "has_ordered_layers": True,
            "effect_layer_count": 0,
            "duration_sec": 34.0,
            "semantic_purity": 0.58,
            "alpha_quality_score": 0.45,
            "edit_success_score": 0.48,
            "background_hole_ratio": 0.28,
            "non_edited_region_preservation": 0.71,
            "occlusion_edge_count": 6,
        },
        {
            "image": "demo/truck.png",
            "label": "LayerForge peeling",
            "status": "ok",
            "num_layers": 6,
            "recompose_psnr": 27.5,
            "recompose_ssim": 0.902,
            "has_graph": True,
            "has_ordered_layers": True,
            "effect_layer_count": 1,
            "duration_sec": 18.0,
            "semantic_purity": 0.72,
            "alpha_quality_score": 0.59,
            "edit_success_score": 0.81,
            "background_hole_ratio": 0.06,
            "non_edited_region_preservation": 0.84,
            "occlusion_edge_count": 12,
        },
    ]

    scored_rows, best_by_image = choose_best_candidates(rows)

    assert len(scored_rows) == 3
    assert len(best_by_image) == 1
    best = best_by_image[0]
    assert best["image"] == "demo/truck.png"
    assert best["label"] == "LayerForge peeling"
    assert best["self_eval_score"] == max(row["self_eval_score"] for row in scored_rows)
    assert best["self_eval_reason"]
