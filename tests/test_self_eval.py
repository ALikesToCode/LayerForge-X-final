from __future__ import annotations

from layerforge.self_eval import choose_best_candidates


def test_choose_best_candidates_prefers_structured_high_fidelity_candidate() -> None:
    rows = [
        {
            "image": "demo/truck.png",
            "label": "Qwen raw (4)",
            "status": "ok",
            "num_layers": 4,
            "recompose_psnr": 29.1,
            "recompose_ssim": 0.885,
            "has_graph": False,
            "has_ordered_layers": True,
            "effect_layer_count": 0,
            "duration_sec": 35.0,
        },
        {
            "image": "demo/truck.png",
            "label": "Qwen + graph preserve (4)",
            "status": "ok",
            "num_layers": 4,
            "recompose_psnr": 28.7,
            "recompose_ssim": 0.876,
            "has_graph": True,
            "has_ordered_layers": True,
            "effect_layer_count": 0,
            "duration_sec": 34.0,
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

