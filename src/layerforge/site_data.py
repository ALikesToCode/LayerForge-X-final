from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


FIGURE_META: dict[str, dict[str, str]] = {
    "frontier_review": {
        "title": "Frontier comparison",
        "caption": "Five-image comparison across native LayerForge, recursive peeling, raw Qwen, and Qwen-plus-graph hybrids.",
    },
    "truck_recomposition_comparison": {
        "title": "Truck recomposition study",
        "caption": "Direct truck comparison between the strongest native search result, raw Qwen layers, and the graph-enriched Qwen export.",
    },
    "prompt_extract_benchmark": {
        "title": "Prompt-conditioned extraction benchmark",
        "caption": "Measured prompt extraction behavior for text, point, box, and text-plus-geometry queries.",
    },
    "transparent_benchmark": {
        "title": "Transparent-layer benchmark",
        "caption": "Alpha-quality and clean-background measurements on synthetic transparent and semi-transparent scenes.",
    },
    "effects_layer_demo": {
        "title": "Associated-effect prototype",
        "caption": "Prototype effect-layer extraction on a controlled synthetic scene with a measured IoU readout.",
    },
    "intrinsic_layer_demo": {
        "title": "Intrinsic layer export",
        "caption": "Per-layer albedo and shading export shown as a stretch-level intrinsic decomposition capability.",
    },
    "public_benchmark_comparison": {
        "title": "Public semantic grouping benchmarks",
        "caption": "Visible semantic grouping on COCO Panoptic and ADE20K using coarse LayerForge groups.",
    },
    "public_depth_comparison": {
        "title": "Public depth benchmarks",
        "caption": "DIODE validation comparisons for geometric depth, Depth Pro, and scaled Depth Pro alignment.",
    },
}


AUDIENCE_CARDS = [
    {
        "title": "Researchers",
        "body": "Use the DALG manifest, report figures, and metrics snapshots to inspect the structural contribution rather than only the final RGB reconstruction.",
    },
    {
        "title": "Editors and art directors",
        "body": "Use the local web UI to upload an image, run a decomposition mode, inspect grouped layers, and export a canonical design manifest without working directly from the CLI.",
    },
    {
        "title": "Developers",
        "body": "Use the Python CLI and the canonical DALG export as the integration surface for external tools, Qwen enrichment, prompt extraction, and benchmark automation.",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_remote_url(text: str) -> tuple[str, str] | None:
    value = text.strip()
    if not value:
        return None
    patterns = [
        r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$",
        r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group("owner"), match.group("repo")
    return None


def _discover_repo_identity(repo_root: Path) -> tuple[str, str] | None:
    try:
        proc = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return _parse_remote_url(proc.stdout)


def _blob_url(repo_url: str, path: str) -> str:
    return f"{repo_url}/blob/main/{path}"


def _frontier_comparisons(frontier_summary: dict[str, Any]) -> list[dict[str, Any]]:
    aggregates = frontier_summary.get("aggregates", [])
    rows = sorted(aggregates, key=lambda item: float(item.get("mean_self_eval_score", 0.0)), reverse=True)
    descriptions = {
        "LayerForge native": "Full native graph pipeline with semantic grouping, amodal completion, intrinsics, and ordered RGBA export.",
        "LayerForge peeling": "Graph-guided recursive peeling with residual inpainting and explicit front-to-back layer logging.",
        "Qwen raw (4)": "Direct Qwen-Image-Layered RGBA decomposition used as the frontier generative baseline.",
        "Qwen + graph preserve (4)": "Raw Qwen layers with DALG metadata while preserving the best external visual stack.",
        "Qwen + graph reorder (4)": "Raw Qwen layers re-exported through the LayerForge depth graph for graph-ordered layering.",
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label"))
        output.append(
            {
                "label": label,
                "images": int(row.get("images", 0)),
                "mean_psnr": float(row.get("mean_psnr", 0.0)),
                "mean_ssim": float(row.get("mean_ssim", 0.0)),
                "mean_self_eval_score": float(row.get("mean_self_eval_score", 0.0)),
                "description": descriptions.get(label, "Measured comparison candidate in the shipped frontier review."),
            }
        )
    return output


def build_project_site_payload(repo_root: Path) -> dict[str, Any]:
    manifest = _read_json(repo_root / "PROJECT_MANIFEST.json")
    frontier = _read_json(repo_root / "report_artifacts/metrics_snapshots/frontier_review_summary.json")
    extract = _read_json(repo_root / "report_artifacts/metrics_snapshots/extract_benchmark_summary.json")
    transparent = _read_json(repo_root / "report_artifacts/metrics_snapshots/transparent_benchmark_summary.json")
    effects = _read_json(repo_root / "report_artifacts/metrics_snapshots/effects_demo_metrics.json")

    repo_identity = _discover_repo_identity(repo_root)
    owner, repo = repo_identity if repo_identity is not None else ("ALikesToCode", "LayerForge-X-final")
    repo_url = f"https://github.com/{owner}/{repo}"
    pages_url = f"https://{owner.lower()}.github.io/{repo}/"

    measured = manifest["measured_results"]
    validated = manifest["validated"]

    figures: list[dict[str, Any]] = []
    for key, raw_path in manifest["figure_pack"].items():
        if key in {"figure_index", "candidate_search_summary"}:
            continue
        rel = raw_path.removeprefix("docs/")
        meta = FIGURE_META.get(key, {"title": key.replace("_", " ").title(), "caption": "Generated figure from the measured LayerForge evidence pack."})
        figures.append({"key": key, "path": rel, **meta})

    docs_links = [
        {"label": "Final report (DOCX)", "href": "final_report_pack/LayerForge_X_Final_Report_2026_04_22.docx"},
        {"label": "Final report (Markdown)", "href": "final_report_pack/LayerForge_X_Final_Report_FULL.md"},
        {"label": "Figure index", "href": "FIGURES.md"},
        {"label": "Current results summary", "href": "RESULTS_SUMMARY_CURRENT.md"},
        {"label": "Project specification", "href": "FINAL_PROJECT_SPEC.md"},
        {"label": "API contract", "href": "api/README.md"},
        {"label": "Canonical manifest on GitHub", "href": _blob_url(repo_url, "PROJECT_MANIFEST.json")},
        {"label": "Submission evidence pack on GitHub", "href": _blob_url(repo_url, "report_artifacts/README.md")},
    ]

    return {
        "project": {
            "name": manifest["project"],
            "tagline": "Depth-aware amodal layered image decomposition from a single RGB image",
            "abstract": (
                "LayerForge-X represents a single RGB image as a depth-aware amodal layer graph (DALG) whose nodes are "
                "editable RGBA layers and whose edges encode pairwise occlusion evidence. The repository couples a native "
                "graph pipeline with Qwen-Image-Layered enrichment, recursive peeling, prompt-conditioned extraction, "
                "transparent-layer recovery, and editability-aware evaluation."
            ),
            "repo_url": repo_url,
            "pages_url": pages_url,
        },
        "validated": validated,
        "hero_metrics": [
            {"label": "Pytest", "value": str(validated.get("pytest", "verified"))},
            {"label": "Frontier native self-eval", "value": f"{measured['frontier_review']['layerforge_native_mean_self_eval_score']:.4f}"},
            {"label": "Prompt text hit rate", "value": f"{measured['extract_benchmark_prompted_grounded']['text_target_hit_rate']:.2f}"},
            {"label": "Transparent alpha MAE", "value": f"{measured['transparent_benchmark']['mean_transparent_alpha_mae']:.4f}"},
            {"label": "Effect IoU", "value": f"{measured['effects_groundtruth_demo_cutting_edge']['effect_iou']:.4f}"},
        ],
        "contributions": [
            "Depth-Aware Amodal Layer Graph (DALG) export as the canonical scene representation.",
            "Native layered decomposition with semantic grouping, occlusion ordering, and optional per-layer intrinsics.",
            "Qwen-Image-Layered import and graph enrichment in preserve-order and reorder modes.",
            "Recursive semantic peeling, prompt-conditioned extraction, transparent-layer recovery, and effect-layer prototyping.",
            "Submission-safe evidence pack with measured JSON snapshots, figures, and a reproducible report build path.",
        ],
        "audiences": AUDIENCE_CARDS,
        "comparisons": _frontier_comparisons(frontier),
        "benchmarks": {
            "prompt_extract": {
                "queries_per_type": int(measured["extract_benchmark_prompted_grounded"]["queries_per_prompt_type"]),
                "text_hit_rate": float(measured["extract_benchmark_prompted_grounded"]["text_target_hit_rate"]),
                "point_hit_rate": float(measured["extract_benchmark_prompted_grounded"]["point_target_hit_rate"]),
                "text_mean_iou": float(measured["extract_benchmark_prompted_grounded"]["text_mean_target_iou"]),
                "point_mean_iou": float(measured["extract_benchmark_prompted_grounded"]["point_mean_target_iou"]),
                "note": "Point-only and box-only prompts localize geometry well but do not satisfy the semantic hit criterion without text guidance.",
            },
            "transparent": {
                "alpha_mae": float(transparent["mean_transparent_alpha_mae"]),
                "background_psnr": float(transparent["mean_background_psnr"]),
                "background_ssim": float(transparent["mean_background_ssim"]),
                "recompose_psnr": float(transparent["mean_recompose_psnr"]),
                "note": "Transparent recomposition is treated as a sanity check. Alpha quality and clean-background quality remain the primary transparent-layer metrics.",
            },
            "effects": {
                "effect_iou": float(effects["effect_iou"]),
                "predicted_pixels": int(effects["predicted_effect_pixels"]),
                "ground_truth_pixels": int(effects["ground_truth_effect_pixels"]),
                "note": "The associated-effect extractor is a heuristic prototype rather than a trained visual-effects decomposer.",
            },
        },
        "public_benchmarks": {
            "coco_panoptic_miou": float(measured["coco_panoptic_mask2former_512"]["miou_supported_groups"]),
            "ade20k_miou": float(measured["ade20k_mask2former_512"]["miou_supported_groups"]),
            "diode_depthpro_scale_abs_rel": float(measured["diode_depthpro_scale_full"]["abs_rel"]),
        },
        "figures": figures,
        "docs_links": docs_links,
        "local_lab": {
            "entrypoint": "layerforge webui --open-browser",
            "modes": [
                {"key": "run", "label": "Layered decomposition", "description": "Run the native LayerForge pipeline and inspect ordered layers, contact sheets, and recomposition."},
                {"key": "extract", "label": "Prompt extraction", "description": "Run the decomposition and export one selected editable target layer plus background-completed previews."},
                {"key": "transparent", "label": "Transparent target recovery", "description": "Approximate a transparent or semi-transparent foreground and estimate a clean background."},
                {"key": "peel", "label": "Recursive peeling", "description": "Iteratively peel the front-most editable layer while logging the residual canvas."},
            ],
        },
    }


def write_project_site_payload(repo_root: Path, output_path: Path) -> Path:
    output_root = ensure_dir(output_path.parent)
    payload = build_project_site_payload(repo_root)
    return write_json(output_root / output_path.name, payload)
