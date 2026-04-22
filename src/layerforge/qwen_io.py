from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .compose import composite_layers_near_to_far
from .depth import estimate_depth
from .graph import amodal_complete, build_nodes, graph_json, merge_compatible_layers, renumber_layers_in_place, topo_order
from .image_io import load_rgb, save_depth16, save_gray, save_rgb, save_rgba
from .intrinsics import decompose_intrinsics, intrinsic_rgba
from .metrics import compute_run_metrics
from .semantic import label_to_group
from .types import Layer, PipelineOutputs, Segment
from .utils import bbox_from_mask, ensure_dir, safe_name, write_json
from .visualize import depth_to_rgb, save_layer_contact_sheet


def _candidate_layer_paths(layers_dir: str | Path) -> list[Path]:
    root = Path(layers_dir)
    exts = {".png", ".webp", ".tif", ".tiff"}
    paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    bad_tokens = ("albedo", "shading", "mask", "depth", "debug", "contact_sheet", "preview", "recomposed")
    return [p for p in paths if not any(tok in p.stem.lower() for tok in bad_tokens)]


def _resolve_manifest_layer_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([root / path.name, root / path, path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _manifest_layer_paths(layers_dir: str | Path) -> tuple[dict[str, Any], list[Path]]:
    root = Path(layers_dir)
    manifest_path = root / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layer_paths = [_resolve_manifest_layer_path(root, item) for item in manifest.get("layer_paths", [])]
        layer_paths = [path for path in layer_paths if path.exists()]
        if layer_paths:
            return manifest, layer_paths
    return manifest, _candidate_layer_paths(root)


def _validate_rgba(arr: np.ndarray) -> np.ndarray:
    rgba = np.asarray(arr, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected an RGBA image")
    return rgba.copy()


def load_external_rgba_segments(layers_dir: str | Path, image_shape: tuple[int, int], min_alpha: float = 0.02) -> tuple[list[Segment], dict[int, np.ndarray]]:
    h, w = image_shape
    segments: list[Segment] = []
    rgba_by_sid: dict[int, np.ndarray] = {}
    for p in _candidate_layer_paths(layers_dir):
        im = Image.open(p).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
        rgba = _validate_rgba(np.asarray(im, dtype=np.uint8))
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        mask = alpha > min_alpha
        if not mask.any():
            continue
        label = p.stem.replace("_", " ").replace("-", " ")
        sid = len(segments)
        segments.append(Segment(sid, label, label_to_group(label), mask, 1.0, bbox_from_mask(mask), "external-rgba", {"path": str(p)}))
        rgba_by_sid[sid] = rgba
    if not segments:
        raise ValueError(f"No usable RGBA layer files found in {layers_dir}")
    return segments, rgba_by_sid


def score_raw_rgba_layers(
    image_path: str | Path,
    layers_dir: str | Path,
    *,
    min_alpha: float = 0.02,
) -> tuple[Path, Path]:
    root = Path(layers_dir)
    manifest, layer_paths = _manifest_layer_paths(root)
    if not layer_paths:
        raise ValueError(f"No usable RGBA layer files found in {layers_dir}")

    first = Image.open(layer_paths[0]).convert("RGBA")
    target_size = first.size
    rgb = np.asarray(Image.open(image_path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS), dtype=np.uint8)

    ordered_layers: list[Layer] = []
    manifest_far_to_near = list(layer_paths)
    for rank, path in enumerate(reversed(manifest_far_to_near)):
        rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        visible = alpha > float(min_alpha)
        label = path.stem.replace("_", " ").replace("-", " ")
        ordered_layers.append(
            Layer(
                id=rank,
                name=f"{rank:03d}_{safe_name(path.stem)}",
                label=label,
                group=label_to_group(label),
                rank=rank,
                depth_median=float(rank),
                depth_p10=float(rank),
                depth_p90=float(rank),
                area=int(visible.sum()),
                bbox=bbox_from_mask(visible),
                alpha=alpha,
                rgba=rgba,
                albedo_rgba=rgba.copy(),
                shading_rgba=rgba.copy(),
                visible_mask=visible,
                amodal_mask=None,
                source_segment_ids=[rank],
                metadata={"source": "qwen_raw_rgba", "path": str(path)},
            )
        )

    recomposed = composite_layers_near_to_far(ordered_layers)
    save_rgba(root / "recomposed_rgba.png", recomposed)
    save_rgb(root / "recomposed_rgb.png", recomposed[..., :3])
    save_layer_contact_sheet(root / "ordered_layer_contact_sheet.png", ordered_layers)

    metrics = compute_run_metrics(rgb, ordered_layers, cfg={})
    metrics.update(
        {
            "mode": "qwen_raw_rgba",
            "model": manifest.get("model"),
            "ordering_assumption": "manifest_order_interpreted_as_far_to_near",
            "resolution": manifest.get("resolution"),
            "num_inference_steps": manifest.get("num_inference_steps"),
            "offload": manifest.get("offload"),
            "input_resized_to": [int(target_size[0]), int(target_size[1])],
        }
    )
    metrics_path = write_json(root / "metrics.json", metrics)
    return metrics_path, root / "recomposed_rgb.png"


def enrich_rgba_layers(
    image_path: str | Path,
    layers_dir: str | Path,
    output_dir: str | Path,
    cfg: dict[str, Any],
    device: str = "auto",
    depth_method: str | None = None,
    flip_depth: bool | None = None,
) -> PipelineOutputs:
    if depth_method:
        cfg["depth"]["method"] = depth_method
    if flip_depth is not None:
        cfg["depth"]["flip"] = bool(flip_depth)

    out = ensure_dir(output_dir)
    dirs = {k: ensure_dir(out / k) for k in ["layers_ordered_rgba", "layers_albedo_rgba", "layers_shading_rgba", "layers_amodal_masks", "debug"]}

    rgb, pil = load_rgb(image_path, cfg.get("io", {}).get("max_side"))
    save_rgb(dirs["debug"] / "input_rgb.png", rgb)

    depth_pred = estimate_depth(pil, rgb, cfg["depth"], device)
    depth = depth_pred.depth.astype(np.float32)
    save_rgb(dirs["debug"] / "depth_gray.png", depth_to_rgb(depth))
    save_depth16(dirs["debug"] / "depth_16bit.png", depth)

    albedo, shading, iid_method = decompose_intrinsics(rgb, cfg["intrinsics"])
    save_rgb(dirs["debug"] / "intrinsic_albedo.png", albedo)
    save_rgb(dirs["debug"] / "intrinsic_shading.png", shading)

    segments, rgba_by_sid = load_external_rgba_segments(layers_dir, depth.shape, float(cfg.get("qwen", {}).get("min_alpha", 0.02)))
    nodes = build_nodes(segments, depth, cfg["layering"])
    ordering_method = str(cfg["layering"].get("ordering_method", "boundary")).lower()
    ordering_scores: dict[int, float] = {}
    if ordering_method in {"learned", "ranker"}:
        from .ranker import learned_order, load_ranker

        model_path = str(cfg["layering"].get("ranker_model_path", "")).strip()
        if not model_path:
            raise RuntimeError("Learned ordering requires layering.ranker_model_path")
        order, ordering_scores = learned_order(nodes, load_ranker(model_path))
    else:
        order = topo_order(nodes)

    layers: list[Layer] = []
    min_area = max(8, int(depth.size * float(cfg["layering"].get("min_layer_area_ratio", 0.001))))
    for rank, sid in enumerate(order):
        seg = next(s for s in segments if s.id == sid)
        rgba = rgba_by_sid[sid]
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        visible = alpha > float(cfg.get("qwen", {}).get("min_alpha", 0.02))
        if int(visible.sum()) < min_area:
            continue
        node = nodes[sid]
        ar, sr = intrinsic_rgba(albedo, shading, alpha)
        amodal = amodal_complete(visible, int(cfg["layering"].get("amodal_expand_px", 16))) if bool(cfg["layering"].get("amodal_enabled", True)) else None
        name = f"{rank:03d}_{safe_name(seg.group)}_{safe_name(seg.label)}"
        layers.append(Layer(
            len(layers), name, seg.label, seg.group, rank,
            node.depth_median, node.depth_p10, node.depth_p90, int(visible.sum()), bbox_from_mask(visible),
            alpha, rgba, ar, sr, visible, amodal, [seg.id], sorted(node.occludes), sorted(node.occluded_by),
            {"source": "external-rgba", "external_path": seg.metadata.get("path"), "depth_source": depth_pred.source, "ordering_method": ordering_method, "ordering_score": ordering_scores.get(sid)},
        ))
    premerge_layer_count = len(layers)
    layers = renumber_layers_in_place(merge_compatible_layers(layers, cfg["layering"]))
    ordered_layers = list(layers)

    ordered_paths = []
    for l in ordered_layers:
        ordered_paths.append(save_rgba(dirs["layers_ordered_rgba"] / f"{l.name}.png", l.rgba))
        save_rgba(dirs["layers_albedo_rgba"] / f"{l.name}_albedo.png", l.albedo_rgba)
        save_rgba(dirs["layers_shading_rgba"] / f"{l.name}_shading.png", l.shading_rgba)
        save_gray(dirs["debug"] / f"{l.name}_alpha.png", l.alpha)
        if l.amodal_mask is not None:
            save_gray(dirs["layers_amodal_masks"] / f"{l.name}_amodal.png", l.amodal_mask.astype(np.float32))

    recomposed = composite_layers_near_to_far(ordered_layers)
    save_rgba(dirs["debug"] / "recomposed_rgba.png", recomposed)
    save_rgb(dirs["debug"] / "recomposed_rgb.png", recomposed[..., :3])
    save_layer_contact_sheet(dirs["debug"] / "ordered_layer_contact_sheet.png", ordered_layers)

    metrics = compute_run_metrics(rgb, ordered_layers, cfg)
    metrics.update({
        "mode": "external_rgba_enrichment",
        "depth_method": cfg["depth"]["method"],
        "depth_source": depth_pred.source,
        "intrinsic_method": iid_method,
        "external_layers_dir": str(layers_dir),
        "ordering_method": ordering_method,
        "premerge_semantic_layers": float(premerge_layer_count),
        "merge_reduction": float(max(0, premerge_layer_count - len(ordered_layers))),
    })
    metrics_path = write_json(out / "metrics.json", metrics)
    graph_path = write_json(dirs["debug"] / "layer_graph.json", graph_json(ordered_layers, nodes))
    manifest = {
        "input": str(image_path),
        "external_layers_dir": str(layers_dir),
        "output_dir": str(out),
        "metrics": str(metrics_path),
        "layer_graph": str(graph_path),
        "ordered_layers_near_to_far": [{"path": str(p), "name": l.name, "rank": l.rank, "label": l.label, "group": l.group, "depth_median": l.depth_median} for p, l in zip(ordered_paths, ordered_layers)],
        "debug": {"depth_gray": str(dirs["debug"] / "depth_gray.png"), "recomposed_rgb": str(dirs["debug"] / "recomposed_rgb.png")},
    }
    manifest_path = write_json(out / "manifest.json", manifest)
    return PipelineOutputs(out, manifest_path, metrics_path, ordered_paths, [], {k: Path(v) for k, v in manifest["debug"].items()})
