from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .compose import composite_layers_near_to_far
from .depth import estimate_depth
from .graph import amodal_complete, build_nodes, graph_json, topo_order
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
            {"source": "external-rgba", "external_path": seg.metadata.get("path"), "depth_source": depth_pred.source},
        ))

    ordered_paths = []
    for l in sorted(layers, key=lambda x: x.rank):
        ordered_paths.append(save_rgba(dirs["layers_ordered_rgba"] / f"{l.name}.png", l.rgba))
        save_rgba(dirs["layers_albedo_rgba"] / f"{l.name}_albedo.png", l.albedo_rgba)
        save_rgba(dirs["layers_shading_rgba"] / f"{l.name}_shading.png", l.shading_rgba)
        save_gray(dirs["debug"] / f"{l.name}_alpha.png", l.alpha)
        if l.amodal_mask is not None:
            save_gray(dirs["layers_amodal_masks"] / f"{l.name}_amodal.png", l.amodal_mask.astype(np.float32))

    recomposed = composite_layers_near_to_far(layers)
    save_rgba(dirs["debug"] / "recomposed_rgba.png", recomposed)
    save_rgb(dirs["debug"] / "recomposed_rgb.png", recomposed[..., :3])
    save_layer_contact_sheet(dirs["debug"] / "ordered_layer_contact_sheet.png", layers)

    metrics = compute_run_metrics(rgb, layers, cfg)
    metrics.update({"mode": "external_rgba_enrichment", "depth_method": cfg["depth"]["method"], "depth_source": depth_pred.source, "intrinsic_method": iid_method, "external_layers_dir": str(layers_dir)})
    metrics_path = write_json(out / "metrics.json", metrics)
    graph_path = write_json(dirs["debug"] / "layer_graph.json", graph_json(layers, nodes))
    manifest = {
        "input": str(image_path),
        "external_layers_dir": str(layers_dir),
        "output_dir": str(out),
        "metrics": str(metrics_path),
        "layer_graph": str(graph_path),
        "ordered_layers_near_to_far": [{"path": str(p), "name": l.name, "rank": l.rank, "label": l.label, "group": l.group, "depth_median": l.depth_median} for p, l in zip(ordered_paths, sorted(layers, key=lambda x: x.rank))],
        "debug": {"depth_gray": str(dirs["debug"] / "depth_gray.png"), "recomposed_rgb": str(dirs["debug"] / "recomposed_rgb.png")},
    }
    manifest_path = write_json(out / "manifest.json", manifest)
    return PipelineOutputs(out, manifest_path, metrics_path, ordered_paths, [], {k: Path(v) for k, v in manifest["debug"].items()})
