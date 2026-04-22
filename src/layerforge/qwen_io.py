from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .compose import composite_layers_near_to_far
from .dalg import export_dalg_manifest
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


def _label_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ")


def _build_layers_for_sid_order(
    sid_order: list[int],
    *,
    segments: list[Segment],
    rgba_by_sid: dict[int, np.ndarray],
    min_alpha: float,
    nodes: dict[int, Any] | None = None,
    albedo: np.ndarray | None = None,
    shading: np.ndarray | None = None,
    amodal_enabled: bool = False,
    amodal_expand_px: int = 16,
    ordering_method: str = "external",
    ordering_scores: dict[int, float] | None = None,
    depth_source: str | None = None,
) -> list[Layer]:
    ordering_scores = ordering_scores or {}
    out: list[Layer] = []
    for rank, sid in enumerate(sid_order):
        seg = next(s for s in segments if s.id == sid)
        rgba = rgba_by_sid[sid]
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        visible = alpha > float(min_alpha)
        if not visible.any():
            continue
        if nodes is None:
            depth_median = depth_p10 = depth_p90 = float(rank)
            occludes: list[int] = []
            occluded_by: list[int] = []
        else:
            node = nodes[sid]
            depth_median = node.depth_median
            depth_p10 = node.depth_p10
            depth_p90 = node.depth_p90
            occludes = sorted(node.occludes)
            occluded_by = sorted(node.occluded_by)
        if albedo is not None and shading is not None:
            ar, sr = intrinsic_rgba(albedo, shading, alpha)
        else:
            ar = rgba.copy()
            sr = rgba.copy()
        amodal = amodal_complete(visible, int(amodal_expand_px)) if amodal_enabled else None
        name = f"{rank:03d}_{safe_name(seg.group)}_{safe_name(seg.label)}"
        out.append(
            Layer(
                len(out),
                name,
                seg.label,
                seg.group,
                rank,
                depth_median,
                depth_p10,
                depth_p90,
                int(visible.sum()),
                bbox_from_mask(visible),
                alpha,
                rgba,
                ar,
                sr,
                visible,
                amodal,
                [seg.id],
                occludes,
                occluded_by,
                {
                    "source": "external-rgba",
                    "external_path": seg.metadata.get("path"),
                    "ordering_method": ordering_method,
                    "ordering_score": ordering_scores.get(sid),
                    "depth_source": depth_source,
                },
            )
        )
    return out


def _score_layer_stack(rgb: np.ndarray, layers: list[Layer]) -> dict[str, float]:
    return compute_run_metrics(rgb, layers, cfg={})


def _pick_best_visual_order(
    rgb: np.ndarray,
    *,
    segments: list[Segment],
    rgba_by_sid: dict[int, np.ndarray],
    min_alpha: float,
) -> tuple[str, dict[str, list[int]], dict[str, dict[str, float]]]:
    orders = {
        "manifest_order": list(range(len(segments))),
        "reversed_manifest_order": list(reversed(range(len(segments)))),
    }
    scores: dict[str, dict[str, float]] = {}
    for name, sid_order in orders.items():
        layers = _build_layers_for_sid_order(
            sid_order,
            segments=segments,
            rgba_by_sid=rgba_by_sid,
            min_alpha=min_alpha,
        )
        scores[name] = _score_layer_stack(rgb, layers)
    best_name = max(
        orders,
        key=lambda key: (
            float(scores[key].get("recompose_psnr", -math.inf)),
            float(scores[key].get("recompose_ssim", -math.inf)),
        ),
    )
    return best_name, orders, scores


def load_external_rgba_segments(layers_dir: str | Path, image_shape: tuple[int, int], min_alpha: float = 0.02) -> tuple[list[Segment], dict[int, np.ndarray], dict[str, Any]]:
    h, w = image_shape
    segments: list[Segment] = []
    rgba_by_sid: dict[int, np.ndarray] = {}
    manifest, layer_paths = _manifest_layer_paths(layers_dir)
    for p in layer_paths:
        im = Image.open(p).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
        rgba = _validate_rgba(np.asarray(im, dtype=np.uint8))
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        mask = alpha > min_alpha
        if not mask.any():
            continue
        label = _label_from_path(p)
        sid = len(segments)
        segments.append(Segment(sid, label, label_to_group(label), mask, 1.0, bbox_from_mask(mask), "external-rgba", {"path": str(p)}))
        rgba_by_sid[sid] = rgba
    if not segments:
        raise ValueError(f"No usable RGBA layer files found in {layers_dir}")
    return segments, rgba_by_sid, manifest


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

    segments, rgba_by_sid, manifest = load_external_rgba_segments(root, rgb.shape[:2], float(min_alpha))
    selected_order_name, candidate_orders, candidate_scores = _pick_best_visual_order(
        rgb,
        segments=segments,
        rgba_by_sid=rgba_by_sid,
        min_alpha=float(min_alpha),
    )
    ordered_layers = _build_layers_for_sid_order(
        candidate_orders[selected_order_name],
        segments=segments,
        rgba_by_sid=rgba_by_sid,
        min_alpha=float(min_alpha),
    )

    recomposed = composite_layers_near_to_far(ordered_layers)
    save_rgba(root / "recomposed_rgba.png", recomposed)
    save_rgb(root / "recomposed_rgb.png", recomposed[..., :3])
    save_layer_contact_sheet(root / "ordered_layer_contact_sheet.png", ordered_layers)

    metrics = _score_layer_stack(rgb, ordered_layers)
    metrics.update(
        {
            "mode": "qwen_raw_rgba",
            "model": manifest.get("model"),
            "ordering_assumption": "best_of_manifest_and_reversed_manifest",
            "selected_visual_order": selected_order_name,
            "manifest_order_psnr": float(candidate_scores["manifest_order"]["recompose_psnr"]),
            "manifest_order_ssim": float(candidate_scores["manifest_order"]["recompose_ssim"]),
            "reversed_manifest_order_psnr": float(candidate_scores["reversed_manifest_order"]["recompose_psnr"]),
            "reversed_manifest_order_ssim": float(candidate_scores["reversed_manifest_order"]["recompose_ssim"]),
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

    qwen_cfg = cfg.get("qwen", {})
    min_alpha = float(qwen_cfg.get("min_alpha", 0.02))
    preserve_external_order = bool(qwen_cfg.get("preserve_external_order", False))
    merge_external_layers = bool(qwen_cfg.get("merge_external_layers", False))

    segments, rgba_by_sid, external_manifest = load_external_rgba_segments(layers_dir, depth.shape, min_alpha)
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

    selected_external_order_name, external_orders, external_scores = _pick_best_visual_order(
        rgb,
        segments=segments,
        rgba_by_sid=rgba_by_sid,
        min_alpha=min_alpha,
    )
    graph_layers_premerge = _build_layers_for_sid_order(
        order,
        segments=segments,
        rgba_by_sid=rgba_by_sid,
        min_alpha=min_alpha,
        nodes=nodes,
        albedo=albedo,
        shading=shading,
        amodal_enabled=bool(cfg["layering"].get("amodal_enabled", True)),
        amodal_expand_px=int(cfg["layering"].get("amodal_expand_px", 16)),
        ordering_method=ordering_method,
        ordering_scores=ordering_scores,
        depth_source=depth_pred.source,
    )
    external_layers_premerge = _build_layers_for_sid_order(
        external_orders[selected_external_order_name],
        segments=segments,
        rgba_by_sid=rgba_by_sid,
        min_alpha=min_alpha,
        nodes=nodes,
        albedo=albedo,
        shading=shading,
        amodal_enabled=bool(cfg["layering"].get("amodal_enabled", True)),
        amodal_expand_px=int(cfg["layering"].get("amodal_expand_px", 16)),
        ordering_method="preserve_external_order",
        depth_source=depth_pred.source,
    )

    if merge_external_layers:
        graph_layers = renumber_layers_in_place(merge_compatible_layers(graph_layers_premerge, cfg["layering"]))
        external_layers = renumber_layers_in_place(merge_compatible_layers(external_layers_premerge, cfg["layering"]))
    else:
        graph_layers = renumber_layers_in_place(list(graph_layers_premerge))
        external_layers = renumber_layers_in_place(list(external_layers_premerge))

    ordered_layers = external_layers if preserve_external_order else graph_layers
    visual_order_mode = "preserve_external_order" if preserve_external_order else "graph_order"
    premerge_layer_count = len(external_layers_premerge if preserve_external_order else graph_layers_premerge)

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

    graph_metrics = _score_layer_stack(rgb, graph_layers)
    external_metrics = _score_layer_stack(rgb, external_layers)
    metrics = _score_layer_stack(rgb, ordered_layers)
    metrics.update({
        "mode": "external_rgba_enrichment",
        "depth_method": cfg["depth"]["method"],
        "depth_source": depth_pred.source,
        "intrinsic_method": iid_method,
        "external_layers_dir": str(layers_dir),
        "ordering_method": ordering_method,
        "visual_order_mode": visual_order_mode,
        "selected_external_visual_order": selected_external_order_name,
        "graph_order_psnr": float(graph_metrics["recompose_psnr"]),
        "graph_order_ssim": float(graph_metrics["recompose_ssim"]),
        "selected_external_order_psnr": float(external_metrics["recompose_psnr"]),
        "selected_external_order_ssim": float(external_metrics["recompose_ssim"]),
        "manifest_order_psnr": float(external_scores["manifest_order"]["recompose_psnr"]),
        "manifest_order_ssim": float(external_scores["manifest_order"]["recompose_ssim"]),
        "reversed_manifest_order_psnr": float(external_scores["reversed_manifest_order"]["recompose_psnr"]),
        "reversed_manifest_order_ssim": float(external_scores["reversed_manifest_order"]["recompose_ssim"]),
        "graph_order_available": True,
        "merge_external_layers": merge_external_layers,
        "preserve_external_order": preserve_external_order,
        "premerge_semantic_layers": float(premerge_layer_count),
        "merge_reduction": float(max(0, premerge_layer_count - len(ordered_layers))),
    })
    metrics_path = write_json(out / "metrics.json", metrics)
    graph_path = write_json(dirs["debug"] / "layer_graph.json", graph_json(ordered_layers, nodes))
    manifest = {
        "input": str(image_path),
        "external_layers_dir": str(layers_dir),
        "output_dir": str(out),
        "visual_order_mode": visual_order_mode,
        "selected_external_visual_order": selected_external_order_name,
        "metrics": str(metrics_path),
        "layer_graph": str(graph_path),
        "external_manifest": external_manifest,
        "graph_order_near_to_far": [{"segment_id": sid, "label": next(s.label for s in segments if s.id == sid), "group": next(s.group for s in segments if s.id == sid)} for sid in order],
        "ordered_layers_near_to_far": [{"path": str(p), "name": l.name, "rank": l.rank, "label": l.label, "group": l.group, "depth_median": l.depth_median} for p, l in zip(ordered_paths, ordered_layers)],
        "debug": {"depth_gray": str(dirs["debug"] / "depth_gray.png"), "recomposed_rgb": str(dirs["debug"] / "recomposed_rgb.png")},
    }
    manifest_path = write_json(out / "manifest.json", manifest)
    canonical_dalg_path = export_dalg_manifest(out)
    manifest["canonical_dalg"] = str(canonical_dalg_path)
    manifest_path = write_json(out / "manifest.json", manifest)
    return PipelineOutputs(out, manifest_path, metrics_path, ordered_paths, [], {k: Path(v) for k, v in manifest["debug"].items()})
