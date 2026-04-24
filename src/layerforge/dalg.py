from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


CANONICAL_DALG_SCHEMA_VERSION = "1.0.0"
CANONICAL_DALG_SCHEMA_URL = "https://layerforge.dev/schemas/dalg.schema.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "albedo_rgba": _rel_path(run_dir, run_dir / "layers_albedo_rgba" / f"{stem}_albedo.png"),
        "shading_rgba": _rel_path(run_dir, run_dir / "layers_shading_rgba" / f"{stem}_shading.png"),
        "amodal_mask": _rel_path(run_dir, run_dir / "layers_amodal_masks" / f"{stem}_amodal.png"),
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
        ordered_layers.append(
            {
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
        )

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

    dalg = {
        "$schema": CANONICAL_DALG_SCHEMA_URL,
        "kind": "layerforge.dalg",
        "schema_version": CANONICAL_DALG_SCHEMA_VERSION,
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
            "edge_count": len(graph_edges),
            "layers": ordered_layers,
            "edges": graph_edges,
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


def export_dalg_manifest(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(run_dir)
    out_path = Path(output_path) if output_path is not None else root / "dalg_manifest.json"
    dalg = build_dalg_manifest(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dalg, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
