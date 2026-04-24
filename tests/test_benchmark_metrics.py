from __future__ import annotations

import json

from layerforge.benchmark import summarize_benchmark_rows, write_markdown_benchmark_report


def test_summarize_benchmark_rows_exposes_world_class_metric_surfaces(tmp_path) -> None:
    rows = [
        {
            "mean_best_iou": 0.5,
            "pairwise_layer_order_accuracy": 1.0,
            "recompose_ssim": 0.9,
            "mean_alpha_quality_score": 0.8,
            "mean_hidden_area_ratio": 0.25,
            "mean_completion_consistency": 0.75,
            "mean_intrinsic_residual": 0.1,
            "runtime_sec": 2.0,
            "peak_memory_mb": 128.0,
        }
    ]

    summary = summarize_benchmark_rows(rows)
    md_path = write_markdown_benchmark_report(tmp_path, summary)

    assert summary["semantic_segmentation_quality"] == 0.5
    assert summary["depth_ordering_quality"] == 1.0
    assert summary["amodal_mask_quality"] == 0.8
    assert summary["intrinsic_decomposition_quality"] == 0.9
    assert summary["ci_passed"] is True
    assert "alpha_matting_quality" in md_path.read_text(encoding="utf-8")


def test_benchmark_summary_is_json_serializable() -> None:
    summary = summarize_benchmark_rows([])
    payload = json.loads(json.dumps(summary))
    assert payload["num_scenes"] == 0
    assert "ci_gates" in payload
