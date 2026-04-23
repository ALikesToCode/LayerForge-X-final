from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_project_manifest_submission_paths_exist() -> None:
    manifest = _load_json(ROOT / "PROJECT_MANIFEST.json")
    shipped_paths = []
    shipped_paths.extend(manifest["figure_pack"].values())
    shipped_paths.extend(manifest["submission_artifacts"].values())
    shipped_paths.extend(
        [
            manifest["measured_results"]["qwen_five_image_review"]["summary"],
            manifest["measured_results"]["frontier_review"]["summary"],
            manifest["measured_results"]["frontier_review"]["editability_suite"],
            manifest["measured_results"]["extract_benchmark_prompted_grounded"]["summary"],
            manifest["measured_results"]["transparent_benchmark"]["summary"],
        ]
    )

    for rel_path in shipped_paths:
        assert (ROOT / rel_path).exists(), rel_path


def test_report_artifact_snapshots_and_figures_are_valid() -> None:
    snapshot_dir = ROOT / "report_artifacts" / "metrics_snapshots"
    for json_path in snapshot_dir.glob("*.json"):
        payload = _load_json(json_path)
        assert payload is not None

    figure_manifest = _load_json(ROOT / "report_artifacts" / "figure_sources" / "figure_manifest.json")
    for rel_path in figure_manifest["generated_figures"].values():
        assert (ROOT / rel_path).exists(), rel_path

    figures_doc = (ROOT / "docs" / "FIGURES.md").read_text(encoding="utf-8")
    for rel_path in figure_manifest["generated_figures"].values():
        figure_name = Path(rel_path).name
        assert figure_name in figures_doc


def test_project_manifest_matches_frontier_and_qwen_snapshots() -> None:
    manifest = _load_json(ROOT / "PROJECT_MANIFEST.json")
    measured = manifest["measured_results"]

    qwen_summary = _load_json(ROOT / "report_artifacts" / "metrics_snapshots" / "qwen_five_image_review_summary.json")
    qwen_rows = {row["label"]: row for row in qwen_summary["aggregates"]}
    qwen_measured = measured["qwen_five_image_review"]
    assert qwen_measured["qwen_graph_reorder_mean_psnr_by_layers"]["6"] == qwen_rows["Qwen + graph reorder (6)"]["mean_psnr"]
    assert qwen_measured["qwen_graph_reorder_mean_psnr_by_layers"]["8"] == qwen_rows["Qwen + graph reorder (8)"]["mean_psnr"]
    assert qwen_measured["qwen_graph_reorder_mean_ssim_by_layers"]["6"] == qwen_rows["Qwen + graph reorder (6)"]["mean_ssim"]
    assert qwen_measured["qwen_graph_reorder_mean_ssim_by_layers"]["8"] == qwen_rows["Qwen + graph reorder (8)"]["mean_ssim"]

    frontier_summary = _load_json(ROOT / "report_artifacts" / "metrics_snapshots" / "frontier_review_summary.json")
    frontier_rows = {row["label"]: row for row in frontier_summary["aggregates"]}
    frontier_measured = measured["frontier_review"]
    assert frontier_measured["layerforge_native_mean_self_eval_score"] == frontier_rows["LayerForge native"]["mean_self_eval_score"]
    assert frontier_measured["layerforge_peeling_mean_self_eval_score"] == frontier_rows["LayerForge peeling"]["mean_self_eval_score"]
    assert frontier_measured["qwen_graph_preserve_mean_self_eval_score"] == frontier_rows["Qwen + graph preserve (4)"]["mean_self_eval_score"]
    assert frontier_measured["qwen_graph_reorder_mean_self_eval_score"] == frontier_rows["Qwen + graph reorder (4)"]["mean_self_eval_score"]
    assert frontier_measured["qwen_raw_mean_self_eval_score"] == frontier_rows["Qwen raw (4)"]["mean_self_eval_score"]


def test_report_source_shell_links_resolve() -> None:
    source = ROOT / "docs" / "final_report_pack" / "LayerForge_X_Final_Report_SOURCE.md"
    links = re.findall(r"\(([^)]+)\)", source.read_text(encoding="utf-8"))
    checked = 0
    for link in links:
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if link.startswith("<") and link.endswith(">"):
            link = link[1:-1]
        if not any(link.endswith(ext) for ext in (".png", ".md", ".json", ".yaml", ".pdf", ".docx", ".bib")):
            continue
        checked += 1
        assert (source.parent / link).resolve().exists(), link
    assert checked > 0
