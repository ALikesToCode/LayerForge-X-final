from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


CANONICAL_DALG_SCHEMA_VERSION = "1.1.0"
CANONICAL_DALG_VERSION = "1.1"
CANONICAL_DALG_SCHEMA_URL = "https://layerforge.dev/schemas/dalg.schema.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rel_path(base: Path, raw_path: str | Path | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _resolve_run_path(run_dir: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return run_dir / path


def _load_graph_rows(run_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    graph_path = _resolve_run_path(run_dir, manifest.get("layer_graph"))
    if graph_path is None or not graph_path.exists():
        return {}, []
    graph = _read_json(graph_path)
    layer_rows = {
        str(item.get("name")): item
        for item in graph.get("layers_near_to_far", [])
        if item.get("name") is not None
    }
    return layer_rows, list(graph.get("occlusion_edges", []))


def _layer_support_paths(run_dir: Path, rgba_path: Path) -> dict[str, str | None]:
    stem = rgba_path.stem
    return {
        "rgba": _rel_path(run_dir, rgba_path),
        "alpha": _rel_path(run_dir, run_dir / "layers_alpha" / f"{stem}_alpha.png"),
        "completed_rgba": _rel_path(run_dir, run_dir / "layers_completed_rgba" / f"{stem}_completed.png"),
        "albedo_rgba": _rel_path(run_dir, run_dir / "layers_albedo_rgba" / f"{stem}_albedo.png"),
        "shading_rgba": _rel_path(run_dir, run_dir / "layers_shading_rgba" / f"{stem}_shading.png"),
        "amodal_mask": _rel_path(run_dir, run_dir / "layers_amodal_masks" / f"{stem}_amodal.png"),
        "hidden_mask": _rel_path(run_dir, run_dir / "layers_hidden_masks" / f"{stem}_hidden.png"),
    }


def _layer_v11_fields(
    *,
    row: dict[str, Any],
    graph_row: dict[str, Any],
    support_paths: dict[str, str],
) -> dict[str, Any]:
    metadata = dict(graph_row.get("metadata", {}))
    alpha_meta = metadata.get("alpha", {}) if isinstance(metadata.get("alpha"), dict) else {}
    depth_median = row.get("depth_median", graph_row.get("depth_median"))
    depth_p10 = graph_row.get("depth_p10")
    depth_p90 = graph_row.get("depth_p90")
    return {
        "visible_mask_path": support_paths.get("alpha"),
        "amodal_mask_path": support_paths.get("amodal_mask"),
        "hidden_mask_path": support_paths.get("hidden_mask"),
        "rgba_path": support_paths.get("rgba"),
        "completed_rgba_path": support_paths.get("completed_rgba"),
        "albedo_path": support_paths.get("albedo_rgba"),
        "shading_path": support_paths.get("shading_rgba"),
        "alpha_confidence_path": support_paths.get("alpha_confidence"),
        "depth_stats": {
            "median": depth_median,
            "p10": depth_p10,
            "p90": depth_p90,
            "trimmed_mean": graph_row.get("depth_trimmed_mean"),
            "boundary_median": graph_row.get("depth_boundary_median"),
            "variance": graph_row.get("depth_variance"),
            "confidence": graph_row.get("depth_confidence"),
        },
        "semantic_labels": [x for x in [row.get("group"), row.get("label")] if x],
        "source_backend": metadata.get("source"),
        "provenance": {
            "source_segment_ids": metadata.get("source_segment_ids", graph_row.get("source_segment_ids", [])),
            "segment_source": metadata.get("source"),
            "ordering_method": metadata.get("ordering_method"),
            "alpha_backend": alpha_meta.get("backend"),
        },
        "quality_metrics": {
            "alpha_quality_score": row.get("alpha_quality_score", metadata.get("alpha_quality_score")),
            "hidden_area_ratio": metadata.get("hidden_area_ratio"),
            "completion_consistency": metadata.get("completion_consistency"),
            "edge_continuity_score": metadata.get("edge_continuity_score"),
            "intrinsic_residual": metadata.get("intrinsic_residual"),
            "area": graph_row.get("area"),
        },
    }


def _edge_v11_fields(edge: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "boundary_depth_delta": edge.get("local_depth_gap"),
        "overlap_ratio": edge.get("overlap_ratio"),
        "contact_score": edge.get("shared_boundary_length"),
        "semantic_prior": edge.get("semantic_prior"),
        "model_confidence": edge.get("confidence"),
    }
    return {
        "relation": edge.get("relation", "in_front_of"),
        "evidence": evidence,
        "confidence": edge.get("confidence"),
        "conflict_notes": edge.get("conflict_notes"),
    }


def _existing_only(run_dir: Path, mapping: dict[str, str | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in mapping.items():
        if value is None:
            continue
        path = run_dir / value
        if path.exists():
            out[key] = value
    return out


def _infer_canvas(root: Path, manifest: dict[str, Any]) -> dict[str, int] | None:
    candidates: list[Path] = []
    input_path = _resolve_run_path(root, manifest.get("input"))
    if input_path is not None:
        candidates.append(input_path)
    for row in manifest.get("ordered_layers_near_to_far", []):
        rgba_path = _resolve_run_path(root, row.get("path"))
        if rgba_path is not None:
            candidates.append(rgba_path)
            break
    for candidate in candidates:
        if candidate.exists():
            try:
                with Image.open(candidate) as image:
                    width, height = image.size
                return {"width": int(width), "height": int(height)}
            except OSError:
                continue
    return None


def build_dalg_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {root}")

    manifest = _read_json(manifest_path)
    metrics_path = _resolve_run_path(root, manifest.get("metrics"))
    metrics = _read_json(metrics_path) if metrics_path and metrics_path.exists() else {}
    graph_layers, graph_edges = _load_graph_rows(root, manifest)

    ordered_layers = []
    for row in manifest.get("ordered_layers_near_to_far", []):
        rgba_path = _resolve_run_path(root, row.get("path"))
        if rgba_path is None:
            continue
        graph_row = graph_layers.get(str(row.get("name")), {})
        support_paths = _existing_only(root, _layer_support_paths(root, rgba_path))
        metadata = dict(graph_row.get("metadata", {}))
        layer_row = {
            "id": int(row.get("rank", 0)),
            "name": row.get("name"),
            "label": row.get("label"),
            "group": row.get("group"),
            "rank": int(row.get("rank", 0)),
            "editable": str(row.get("group")) != "background",
            "bbox": graph_row.get("bbox"),
            "area": graph_row.get("area"),
            "depth_median": row.get("depth_median", graph_row.get("depth_median")),
            "occludes": graph_row.get("occludes", []),
            "occluded_by": graph_row.get("occluded_by", []),
            "paths": support_paths,
            "metadata": metadata,
        }
        layer_row.update(_layer_v11_fields(row=row, graph_row=graph_row, support_paths=support_paths))
        ordered_layers.append(layer_row)

    debug_paths = {
        key: rel
        for key, rel in (
            (key, _rel_path(root, value)) for key, value in dict(manifest.get("debug", {})).items()
        )
        if rel is not None
    }
    grouped_layers = [_rel_path(root, item) for item in manifest.get("grouped_layers", [])]
    effect_layers = [_rel_path(root, item) for item in manifest.get("effect_layers", [])]

    canvas = _infer_canvas(root, manifest)
    design_layers = [
        {
            "name": layer["name"],
            "path": layer["paths"].get("rgba"),
            "rank": layer["rank"],
            "group": layer["group"],
            "bbox": layer["bbox"],
            "depth_median": layer["depth_median"],
            "editable": layer["editable"],
        }
        for layer in ordered_layers
    ]

    enriched_edges = []
    for edge in graph_edges:
        edge_row = dict(edge)
        edge_row.update(_edge_v11_fields(edge_row))
        enriched_edges.append(edge_row)

    input_path = _resolve_run_path(root, manifest.get("input"))
    config = manifest.get("config", {})
    metrics_model_manifest = {
        "segmentation": metrics.get("segmentation_method"),
        "depth": metrics.get("depth_method"),
        "depth_source": metrics.get("depth_source"),
        "matting": config.get("matting", {}).get("method") if isinstance(config, dict) else None,
        "intrinsics": metrics.get("intrinsic_method"),
        "inpainting": metrics.get("inpaint_method"),
    }

    dalg = {
        "$schema": CANONICAL_DALG_SCHEMA_URL,
        "kind": "layerforge.dalg",
        "schema_version": CANONICAL_DALG_SCHEMA_VERSION,
        "dalg_version": CANONICAL_DALG_VERSION,
        "alpha_mode": "straight",
        "color_space": "sRGB",
        "input_hash": _sha256_file(input_path),
        "model_manifest": metrics_model_manifest,
        "creation_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_hash": _sha256_json(config),
        "canvas": canvas,
        "asset": {
            "input": _rel_path(root, manifest.get("input")),
            "run_dir": ".",
            "mode": manifest.get("mode", "native_layerforge"),
        },
        "recipe": {
            "segmentation_method": metrics.get("segmentation_method"),
            "depth_method": metrics.get("depth_method"),
            "depth_source": metrics.get("depth_source"),
            "ordering_method": manifest.get("ordering_method", metrics.get("ordering_method")),
            "visual_order_mode": manifest.get("visual_order_mode"),
            "selected_external_visual_order": manifest.get("selected_external_visual_order"),
            "intrinsic_method": metrics.get("intrinsic_method"),
            "inpaint_method": metrics.get("inpaint_method"),
        },
        "graph": {
            "ordering": "near_to_far",
            "graph_order_available": bool(graph_edges or manifest.get("graph_order_near_to_far")),
            "node_count": len(ordered_layers),
            "edge_count": len(enriched_edges),
            "layers": ordered_layers,
            "edges": enriched_edges,
        },
        "layers": design_layers,
        "metrics": metrics,
        "exports": {
            "source_manifest": "manifest.json",
            "metrics": _rel_path(root, metrics_path) if metrics_path is not None else None,
            "grouped_layers": [path for path in grouped_layers if path is not None],
            "effect_layers": [path for path in effect_layers if path is not None],
            "debug": debug_paths,
        },
    }
    return dalg


def migrate_dalg_manifest(dalg: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(dalg)
    if "dalg_version" not in migrated:
        schema_version = str(migrated.get("schema_version", "1.0.0"))
        migrated["dalg_version"] = "1.0" if schema_version.startswith("1.0") else CANONICAL_DALG_VERSION
    return migrated


def load_dalg_manifest(path: str | Path) -> dict[str, Any]:
    return migrate_dalg_manifest(_read_json(Path(path)))


def validate_dalg_manifest(dalg: dict[str, Any], run_dir: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if dalg.get("$schema") != CANONICAL_DALG_SCHEMA_URL:
        errors.append("invalid or missing $schema")
    if dalg.get("kind") != "layerforge.dalg":
        errors.append("invalid or missing kind")
    if not dalg.get("schema_version"):
        errors.append("missing schema_version")
    if not dalg.get("dalg_version"):
        errors.append("missing dalg_version")
    graph = dalg.get("graph", {})
    layers = graph.get("layers", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    if graph.get("node_count") != len(layers):
        errors.append("graph.node_count does not match graph.layers")
    if graph.get("edge_count") != len(edges):
        errors.append("graph.edge_count does not match graph.edges")
    endpoint_ids = {layer.get("id") for layer in layers if isinstance(layer, dict)}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        provenance = layer.get("provenance", {})
        source_ids = provenance.get("source_segment_ids", []) if isinstance(provenance, dict) else []
        endpoint_ids.update(source_ids)
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("graph edge is not an object")
            continue
        if edge.get("near_id") not in endpoint_ids:
            errors.append(f"edge near_id is not a known layer/source id: {edge.get('near_id')}")
        if edge.get("far_id") not in endpoint_ids:
            errors.append(f"edge far_id is not a known layer/source id: {edge.get('far_id')}")
        if "evidence" not in edge:
            errors.append(f"edge missing evidence: {edge.get('near_id')}->{edge.get('far_id')}")
    if run_dir is not None:
        root = Path(run_dir)
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            for key in (
                "visible_mask_path",
                "amodal_mask_path",
                "hidden_mask_path",
                "rgba_path",
                "completed_rgba_path",
                "albedo_path",
                "shading_path",
                "alpha_confidence_path",
            ):
                rel = layer.get(key)
                if rel and not (root / rel).exists():
                    errors.append(f"missing layer file for {layer.get('name')}.{key}: {rel}")
            for rel in dict(layer.get("paths", {})).values():
                if rel and not (root / rel).exists():
                    errors.append(f"missing layer path for {layer.get('name')}: {rel}")
    return errors


def export_dalg_manifest(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(run_dir)
    out_path = Path(output_path) if output_path is not None else root / "dalg_manifest.json"
    dalg = build_dalg_manifest(root)
    validation_errors = validate_dalg_manifest(dalg, root)
    if validation_errors:
        detail = "; ".join(validation_errors[:5])
        raise RuntimeError(f"DALG manifest validation failed: {detail}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dalg, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
