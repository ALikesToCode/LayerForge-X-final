from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .compose import composite_layers_near_to_far, rgba_from_rgb_alpha
from .dalg import export_dalg_manifest
from .depth import estimate_depth
from .graph import build_completed_background_layer, build_layers, grouped_layers, renumber_layers_in_place
from .image_io import load_rgb, save_depth16, save_gray, save_rgb, save_rgba
from .inpaint import inpaint_background
from .intrinsics import decompose_intrinsics, intrinsic_rgba
from .matting import predict_alpha_matte
from .metrics import compute_run_metrics
from .segment import resolve_disjoint_masks, segment_image, summarize_segments
from .types import Layer, PipelineOutputs
from .utils import bbox_from_mask, ensure_dir, safe_name, write_json
from .visualize import depth_to_rgb, draw_segment_labels, save_layer_contact_sheet, segmentation_overlay


def _preview_rgb(rgba: np.ndarray) -> np.ndarray:
    arr = rgba.astype(np.float32)
    alpha = arr[..., 3:4] / 255.0
    base = np.full(arr[..., :3].shape, 245.0, dtype=np.float32)
    return np.clip(arr[..., :3] * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)


def _save_peeling_strip(path: str | Path, frames: list[np.ndarray]) -> Path:
    if not frames:
        return Path(path)
    imgs = [Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), mode="RGB") for frame in frames]
    h = max(im.height for im in imgs)
    margin = 8
    canvas = Image.new("RGB", (sum(im.width for im in imgs) + margin * (len(imgs) + 1), h + 2 * margin), "white")
    x = margin
    for im in imgs:
        y = margin + (h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + margin
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def extract_associated_effect_layer(
    *,
    current_rgb: np.ndarray,
    inpainted_rgb: np.ndarray,
    core_mask: np.ndarray,
    label: str,
    rank: int,
    cfg: dict[str, Any],
    depth_median: float = 0.5,
    depth_p10: float | None = None,
    depth_p90: float | None = None,
    albedo: np.ndarray | None = None,
    shading: np.ndarray | None = None,
    device: str = "auto",
) -> Layer | None:
    if not bool(cfg.get("enabled", True)):
        return None
    core = np.asarray(core_mask, dtype=bool)
    if not core.any():
        return None

    dilate_px = max(1, int(cfg.get("dilate_px", 22)))
    inner_dilate_px = max(0, int(cfg.get("inner_dilate_px", 4)))
    outer = ndi.binary_dilation(core, iterations=dilate_px)
    inner = ndi.binary_dilation(core, iterations=inner_dilate_px) if inner_dilate_px > 0 else core
    ring = outer & ~inner

    if bool(cfg.get("prefer_downward", True)):
        ys, _ = np.where(core)
        if ys.size:
            center_y = int(np.median(ys))
            y_coords = np.arange(core.shape[0], dtype=np.int32)[:, None]
            ring &= y_coords >= center_y

    reference_rgb = np.asarray(inpainted_rgb, dtype=np.uint8)
    use_provided_reference = bool(cfg.get("use_provided_reference", False))
    if not use_provided_reference:
        support_dilate_px = max(1, int(cfg.get("support_dilate_px", max(6, dilate_px // 2))))
        support_mask = ndi.binary_dilation(core, iterations=support_dilate_px)
        if bool(cfg.get("prefer_downward", True)):
            ys, _ = np.where(core)
            if ys.size:
                center_y = int(np.median(ys))
                y_coords = np.arange(core.shape[0], dtype=np.int32)[:, None]
                support_mask &= y_coords >= center_y
        if support_mask.any():
            local_inpainted, _, _ = inpaint_background(
                current_rgb,
                support_mask,
                {
                    "method": "opencv_telea",
                    "radius": float(cfg.get("support_inpaint_radius", 5)),
                },
            )
            reference_rgb = local_inpainted
    else:
        support_dilate_px = 0
    delta = np.mean(np.abs(current_rgb.astype(np.float32) - reference_rgb.astype(np.float32)), axis=2) / 255.0
    candidate = ring & (delta >= float(cfg.get("delta_threshold", 0.05)))
    if int(candidate.sum()) < int(cfg.get("min_area_px", 80)):
        return None

    alpha_scale = max(1e-3, float(cfg.get("alpha_scale", 0.18)))
    residual_alpha = np.clip(delta / alpha_scale, 0.0, 1.0).astype(np.float32)
    residual_alpha *= candidate.astype(np.float32)
    alpha, alpha_backend_meta = _refine_effect_alpha(
        current_rgb=current_rgb,
        residual_alpha=residual_alpha,
        core_mask=core,
        candidate_mask=candidate,
        cfg=cfg,
        device=device,
    )
    visible = alpha > 0.05
    if int(visible.sum()) < int(cfg.get("min_area_px", 80)):
        return None

    rgba = rgba_from_rgb_alpha(current_rgb, alpha)
    if albedo is not None and shading is not None:
        albedo_rgba, shading_rgba = intrinsic_rgba(albedo, shading, alpha)
    else:
        albedo_rgba = rgba.copy()
        shading_rgba = rgba.copy()

    clean_label = f"{label} effect"
    return Layer(
        id=rank,
        name=f"{rank:03d}_effect_{safe_name(label)}",
        label=clean_label,
        group="effect",
        rank=rank,
        depth_median=float(depth_median),
        depth_p10=float(depth_p10 if depth_p10 is not None else depth_median),
        depth_p90=float(depth_p90 if depth_p90 is not None else depth_median),
        area=int(visible.sum()),
        bbox=bbox_from_mask(visible),
        alpha=alpha,
        rgba=rgba,
        albedo_rgba=albedo_rgba,
        shading_rgba=shading_rgba,
        visible_mask=visible,
        amodal_mask=visible.copy(),
        metadata={
            "effect_type": "associated_residual",
            "delta_mean": float(delta[visible].mean()) if visible.any() else 0.0,
            "support_dilate_px": support_dilate_px,
            "use_provided_reference": use_provided_reference,
            "alpha_backend": alpha_backend_meta,
        },
    )


def _refine_effect_alpha(
    *,
    current_rgb: np.ndarray,
    residual_alpha: np.ndarray,
    core_mask: np.ndarray,
    candidate_mask: np.ndarray,
    cfg: dict[str, Any],
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    alpha = np.clip(np.asarray(residual_alpha, dtype=np.float32), 0.0, 1.0)
    backend_alpha, backend_meta = predict_alpha_matte(
        current_rgb,
        np.logical_or(core_mask, candidate_mask),
        {
            "backend": cfg.get("alpha_backend", "auto"),
            "model": cfg.get("alpha_backend_model", "ZhengPeng7/BiRefNet-matting"),
            "max_side": cfg.get("alpha_backend_max_side", 1024),
            "crop_expand_px": cfg.get("alpha_backend_crop_expand_px", 48),
            "support_expand_px": cfg.get("alpha_backend_support_expand_px", 16),
            "respect_support_mask": cfg.get("alpha_backend_respect_support_mask", True),
            "prefer_half": cfg.get("alpha_backend_prefer_half", True),
        },
        device=device,
    )
    if backend_alpha is None:
        return alpha, backend_meta

    core_alpha = core_mask.astype(np.float32)
    effect_alpha = np.clip(backend_alpha.astype(np.float32) - core_alpha, 0.0, 1.0)
    effect_alpha *= candidate_mask.astype(np.float32)
    backend_weight = float(cfg.get("alpha_backend_weight", 0.75))
    alpha = np.clip(np.maximum(alpha, effect_alpha * backend_weight), 0.0, 1.0)
    alpha *= candidate_mask.astype(np.float32)
    return alpha, backend_meta


def run_recursive_peeling(image_path: str | Path, output_dir: str | Path, cfg: dict[str, Any], device: str = "auto") -> PipelineOutputs:
    out = ensure_dir(output_dir)
    dirs = {
        key: ensure_dir(out / key)
        for key in [
            "iterations",
            "layers_ordered_rgba",
            "layers_grouped_rgba",
            "layers_effects_rgba",
            "layers_albedo_rgba",
            "layers_shading_rgba",
            "layers_amodal_masks",
            "debug",
        ]
    }

    rgb, pil = load_rgb(image_path, cfg.get("io", {}).get("max_side"))
    save_rgb(dirs["debug"] / "input_rgb.png", rgb)

    current_rgb = rgb.copy()
    current_pil = pil
    peel_layers: list[Layer] = []
    iteration_rows: list[dict[str, Any]] = []
    strip_frames: list[np.ndarray] = [rgb]
    max_layers = max(1, int(cfg.get("peeling", {}).get("max_layers", 6)))
    min_remaining_ratio = float(cfg.get("peeling", {}).get("min_remaining_foreground_ratio", 0.001))
    min_remaining_area = max(8, int(current_rgb.shape[0] * current_rgb.shape[1] * min_remaining_ratio))

    for iteration in range(max_layers):
        iteration_dir = ensure_dir(dirs["iterations"] / f"iteration_{iteration:02d}")
        save_rgb(iteration_dir / "input.png", current_rgb)

        raw_segments = segment_image(current_rgb, current_pil, cfg["segmentation"], device)
        if not raw_segments:
            break
        depth_pred = estimate_depth(current_pil, current_rgb, cfg["depth"], device)
        depth = depth_pred.depth.astype(np.float32)
        segments = resolve_disjoint_masks(raw_segments, depth)
        if not segments:
            break

        save_rgb(iteration_dir / "segmentation_overlay.png", segmentation_overlay(current_rgb, segments))
        save_rgb(iteration_dir / "segments_labeled.png", draw_segment_labels(current_rgb, segments))
        save_rgb(iteration_dir / "depth_gray.png", depth_to_rgb(depth))
        save_depth16(iteration_dir / "depth_16bit.png", depth)

        albedo, shading, iid_method = decompose_intrinsics(current_rgb, cfg["intrinsics"])
        semantic_layers, nodes = build_layers(current_rgb, segments, depth, albedo, shading, cfg["layering"], cfg["matting"])
        selected = next((layer for layer in semantic_layers if layer.group != "background"), None)
        if selected is None or selected.area < min_remaining_area:
            break

        remove_mask = selected.amodal_mask.astype(bool) if selected.amodal_mask is not None else selected.visible_mask.astype(bool)
        save_gray(iteration_dir / "selected_mask.png", remove_mask.astype(np.float32))
        save_rgba(iteration_dir / "selected_layer.png", selected.rgba)
        write_json(iteration_dir / "segments.json", summarize_segments(segments))

        next_rgb, bg_mask, inpaint_method = inpaint_background(current_rgb, remove_mask, cfg["inpainting"], device)
        save_rgb(iteration_dir / "residual_inpainted.png", next_rgb)
        save_gray(iteration_dir / "residual_mask.png", bg_mask.astype(np.float32))

        peel_layers.append(selected)
        effect_layer = extract_associated_effect_layer(
            current_rgb=current_rgb,
            inpainted_rgb=next_rgb,
            core_mask=remove_mask,
            label=selected.label,
            rank=len(peel_layers),
            cfg=cfg.get("effects", {}),
            depth_median=selected.depth_median,
            depth_p10=selected.depth_p10,
            depth_p90=selected.depth_p90,
            albedo=albedo,
            shading=shading,
            device=device,
        )
        if effect_layer is not None:
            peel_layers.append(effect_layer)
            save_rgba(iteration_dir / "effect_layer.png", effect_layer.rgba)

        iteration_rows.append(
            {
                "iteration": iteration,
                "selected_label": selected.label,
                "selected_group": selected.group,
                "selected_area": int(selected.area),
                "depth_source": depth_pred.source,
                "intrinsic_method": iid_method,
                "inpaint_method": inpaint_method,
                "effect_layer": effect_layer.name if effect_layer is not None else None,
            }
        )
        strip_frames.extend([_preview_rgb(selected.rgba), next_rgb])
        current_rgb = next_rgb
        current_pil = Image.fromarray(current_rgb, mode="RGB")

    bg_albedo, bg_shading, iid_method = decompose_intrinsics(current_rgb, cfg["intrinsics"])
    background = build_completed_background_layer(current_rgb, bg_albedo, bg_shading, len(peel_layers), "recursive_peeling_background")
    ordered_layers = renumber_layers_in_place(peel_layers + [background])

    ordered_paths: list[Path] = []
    effect_paths: list[Path] = []
    for layer in ordered_layers:
        ordered_paths.append(save_rgba(dirs["layers_ordered_rgba"] / f"{layer.name}.png", layer.rgba))
        save_rgba(dirs["layers_albedo_rgba"] / f"{layer.name}_albedo.png", layer.albedo_rgba)
        save_rgba(dirs["layers_shading_rgba"] / f"{layer.name}_shading.png", layer.shading_rgba)
        if layer.amodal_mask is not None:
            save_gray(dirs["layers_amodal_masks"] / f"{layer.name}_amodal.png", layer.amodal_mask.astype(np.float32))
        if layer.group == "effect":
            effect_paths.append(save_rgba(dirs["layers_effects_rgba"] / f"{layer.name}.png", layer.rgba))

    grouped = grouped_layers(ordered_layers, bins=3)
    grouped_paths = [save_rgba(dirs["layers_grouped_rgba"] / f"{g.name}.png", g.rgba) for g in grouped]
    recomposed = composite_layers_near_to_far(ordered_layers)
    save_rgba(dirs["debug"] / "recomposed_rgba.png", recomposed)
    save_rgb(dirs["debug"] / "recomposed_rgb.png", recomposed[..., :3])
    save_rgb(dirs["debug"] / "background_completion.png", current_rgb)
    save_layer_contact_sheet(dirs["debug"] / "ordered_layer_contact_sheet.png", ordered_layers)
    save_layer_contact_sheet(dirs["debug"] / "grouped_layer_contact_sheet.png", grouped)
    strip_path = _save_peeling_strip(dirs["debug"] / "peeling_strip.png", strip_frames)

    metrics = compute_run_metrics(rgb, ordered_layers, cfg)
    metrics.update(
        {
            "mode": "recursive_peeling",
            "iteration_count": float(len(iteration_rows)),
            "effect_layer_count": float(sum(1 for layer in ordered_layers if layer.group == "effect")),
            "core_layer_count": float(sum(1 for layer in ordered_layers if layer.group not in {"effect", "background"})),
            "intrinsic_method": iid_method,
        }
    )
    metrics_path = write_json(out / "metrics.json", metrics)

    peel_graph = {
        "mode": "recursive_peeling",
        "input": str(image_path),
        "iterations": iteration_rows,
        "layers_near_to_far": [
            {
                "rank": layer.rank,
                "name": layer.name,
                "label": layer.label,
                "group": layer.group,
                "depth_median": layer.depth_median,
                "area": layer.area,
            }
            for layer in ordered_layers
        ],
    }
    graph_path = write_json(dirs["debug"] / "layer_graph.json", peel_graph)

    manifest = {
        "mode": "recursive_peeling",
        "input": str(image_path),
        "output_dir": str(out),
        "metrics": str(metrics_path),
        "layer_graph": str(graph_path),
        "ordered_layers_near_to_far": [
            {"path": str(path), "name": layer.name, "rank": layer.rank, "label": layer.label, "group": layer.group, "depth_median": layer.depth_median}
            for path, layer in zip(ordered_paths, ordered_layers)
        ],
        "grouped_layers": [str(path) for path in grouped_paths],
        "effect_layers": [str(path) for path in effect_paths],
        "iterations": iteration_rows,
        "debug": {
            "input_rgb": str(dirs["debug"] / "input_rgb.png"),
            "background_completion": str(dirs["debug"] / "background_completion.png"),
            "recomposed_rgb": str(dirs["debug"] / "recomposed_rgb.png"),
            "peeling_strip": str(strip_path),
        },
    }
    manifest_path = write_json(out / "manifest.json", manifest)
    canonical_dalg_path = export_dalg_manifest(out)
    manifest["canonical_dalg"] = str(canonical_dalg_path)
    manifest_path = write_json(out / "manifest.json", manifest)
    return PipelineOutputs(out, manifest_path, metrics_path, ordered_paths, grouped_paths, {k: Path(v) for k, v in manifest["debug"].items() if v})
