from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dalg import load_dalg_manifest, validate_dalg_manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _alpha_range_ok(path: Path) -> bool:
    arr = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    alpha = arr[..., 3].astype(np.float32) / 255.0
    return bool(np.isfinite(alpha).all() and alpha.min() >= 0.0 and alpha.max() <= 1.0)


def _gray_range_ok(path: Path) -> bool:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return bool(np.isfinite(arr).all() and arr.min() >= 0.0 and arr.max() <= 1.0)


def _recomposition_residual(root: Path, manifest: dict[str, Any]) -> float | None:
    input_path = _resolve(root, manifest.get("input"))
    recomposed_path = _resolve(root, dict(manifest.get("debug", {})).get("recomposed_rgb"))
    if input_path is None or recomposed_path is None or not input_path.exists() or not recomposed_path.exists():
        return None
    src = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.float32)
    rec = np.asarray(Image.open(recomposed_path).convert("RGB").resize((src.shape[1], src.shape[0])), dtype=np.float32)
    return float(np.mean(np.abs(src - rec)) / 255.0)


def validate_run_outputs(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.exists():
        return {"ok": False, "errors": [f"missing manifest: {manifest_path}"], "warnings": [], "metrics": {}}

    manifest = _read_json(manifest_path)
    layer_rows = list(manifest.get("ordered_layers_near_to_far", []))
    for row in layer_rows:
        for key in ["path", "alpha_path", "alpha_confidence_path", "completed_path", "hidden_mask_path"]:
            raw = row.get(key)
            if raw is None:
                continue
            path = _resolve(root, raw)
            if path is None or not path.exists():
                errors.append(f"missing layer file {key}: {raw}")
        rgba_path = _resolve(root, row.get("path"))
        if rgba_path is not None and rgba_path.exists() and not _alpha_range_ok(rgba_path):
            errors.append(f"invalid alpha values: {row.get('path')}")
        alpha_confidence_path = _resolve(root, row.get("alpha_confidence_path"))
        if alpha_confidence_path is not None and alpha_confidence_path.exists() and not _gray_range_ok(alpha_confidence_path):
            errors.append(f"invalid alpha confidence values: {row.get('alpha_confidence_path')}")

    graph_path = _resolve(root, manifest.get("layer_graph"))
    if graph_path is not None and graph_path.exists():
        graph = _read_json(graph_path)
        for edge in graph.get("occlusion_edges", []):
            if "relation" not in edge:
                errors.append(f"graph edge missing relation: {edge}")
            if "confidence" not in edge:
                errors.append(f"graph edge missing confidence: {edge}")
            evidence = edge.get("evidence")
            if not isinstance(evidence, dict):
                errors.append(f"graph edge missing evidence: {edge}")
            elif not any(evidence.get(key) is not None for key in ("boundary_depth_delta", "overlap_ratio", "contact_score", "semantic_prior", "model_confidence")):
                errors.append(f"graph edge evidence is empty: {edge}")
        removed = [edge for node in graph.get("segment_nodes", []) for edge in node.get("removed_edges", [])]
        if removed:
            warnings.append(f"cycle_resolution_removed_edges={len(removed)}")

    dalg_path = _resolve(root, manifest.get("canonical_dalg"))
    if dalg_path is not None and dalg_path.exists():
        dalg = load_dalg_manifest(dalg_path)
        errors.extend(validate_dalg_manifest(dalg, root))
    elif manifest.get("canonical_dalg"):
        errors.append(f"missing canonical_dalg: {manifest.get('canonical_dalg')}")

    residual = _recomposition_residual(root, manifest)
    metrics = {"recomposition_residual": residual}
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }
