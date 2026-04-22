from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

from .image_io import save_gray, save_rgb, save_rgba
from .semantic import BACKGROUND_GROUPS
from .utils import ensure_dir, write_json


EDIT_EXCLUDED_GROUPS = set(BACKGROUND_GROUPS) | {"effect"}


def _resolve_manifest_path(path_or_dir: str | Path) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find manifest at {path}")
    return path


def _resolve_path(base: Path, raw: str | Path) -> Path:
    value = Path(raw)
    if value.is_absolute():
        return value
    candidate = (base / value).resolve()
    if candidate.exists():
        return candidate
    return value.resolve()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _alpha_mask(rgba: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    return (rgba[..., 3].astype(np.float32) / 255.0) > threshold


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rgba_float(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgba_f = rgba.astype(np.float32) / 255.0
    return rgba_f[..., :3], rgba_f[..., 3:4]


def _composite_rgba_layers(layers: list[dict[str, Any]]) -> np.ndarray:
    if not layers:
        raise ValueError("At least one layer is required to composite")
    h, w = layers[0]["rgba"].shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha_out = np.zeros((h, w, 1), dtype=np.float32)
    for layer in sorted(layers, key=lambda item: int(item["rank"]), reverse=True):
        src_rgb, src_alpha = _rgba_float(layer["rgba"])
        rgb = src_rgb * src_alpha + rgb * (1.0 - src_alpha)
        alpha_out = src_alpha + alpha_out * (1.0 - src_alpha)
    return np.dstack([np.clip(rgb * 255.0, 0, 255), np.clip(alpha_out[..., 0] * 255.0, 0, 255)]).astype(np.uint8)


def _shift_rgba(rgba: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return rgba.copy()
    h, w = rgba.shape[:2]
    out = np.zeros_like(rgba)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(w, w - dx) if dx >= 0 else min(w, w + dx)
    src_y1 = min(h, h - dy) if dy >= 0 else min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    out[dst_y0:dst_y1, dst_x0:dst_x1] = rgba[src_y0:src_y1, src_x0:src_x1]
    return out


def _recolor_rgba(rgba: np.ndarray, color: tuple[int, int, int] = (245, 122, 70)) -> np.ndarray:
    out = rgba.copy()
    alpha = out[..., 3:4].astype(np.float32) / 255.0
    tint = np.asarray(color, dtype=np.float32)
    rgb = out[..., :3].astype(np.float32)
    out[..., :3] = np.clip(rgb * 0.35 + tint * 0.65 * alpha + rgb * 0.25 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return out


def _blur_rgba(rgba: np.ndarray, radius: float = 5.0) -> np.ndarray:
    pil = Image.fromarray(rgba, mode="RGBA")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGBA"), dtype=np.uint8)


def _delta_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2) / 255.0


def _mean_in_region(delta: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(delta[mask].mean())


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    return ndi.binary_dilation(mask.astype(bool), iterations=iterations)


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if box is None:
        return mask
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return mask
    mask[y1:y2, x1:x2] = True
    return mask


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def _layer_text_score(layer: dict[str, Any], prompt: str | None) -> float:
    tokens = _tokenize(prompt)
    if not tokens:
        return 0.0
    hay = " ".join(
        [
            str(layer.get("name", "")),
            str(layer.get("label", "")),
            str(layer.get("group", "")),
        ]
    ).lower()
    matches = sum(1 for token in tokens if token in hay)
    return float(matches) / float(max(1, len(tokens)))


def _load_graph_edge_count(run_dir: Path) -> int:
    for name in ("layer_graph.json", "peeling_graph.json"):
        path = run_dir / "debug" / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(len(payload.get("occlusion_edges", [])))
    return 0


def load_ordered_layers(path_or_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    manifest_path = _resolve_manifest_path(path_or_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    input_path = _resolve_path(base_dir, manifest["input"])
    input_rgb = _load_rgb(input_path)
    ordered_layers = manifest.get("ordered_layers_near_to_far")
    if ordered_layers:
        raw_layers = ordered_layers
    else:
        raw_layers = [
            {
                "path": path,
                "name": Path(str(path)).stem,
                "label": Path(str(path)).stem,
                "group": "external",
                "rank": idx,
            }
            for idx, path in enumerate(manifest.get("layer_paths", []))
        ]
    layers: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_layers):
        rgba_path = _resolve_path(base_dir, item["path"])
        rgba = _load_rgba(rgba_path)
        mask = _alpha_mask(rgba)
        layers.append(
            {
                "path": rgba_path,
                "name": item.get("name", rgba_path.stem),
                "label": item.get("label", item.get("name", rgba_path.stem)),
                "group": item.get("group", "external"),
                "rank": int(item.get("rank", idx)),
                "depth_median": item.get("depth_median"),
                "rgba": rgba,
                "alpha": rgba[..., 3].astype(np.float32) / 255.0,
                "mask": mask,
                "bbox": _bbox(mask),
            }
        )
    return manifest, sorted(layers, key=lambda item: item["rank"]), input_rgb


def select_editable_layer(
    layers: list[dict[str, Any]],
    *,
    prompt: str | None = None,
    point: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    target_name: str | None = None,
) -> dict[str, Any] | None:
    if not layers:
        return None
    h, w = layers[0]["rgba"].shape[:2]
    point_x, point_y = point if point is not None else (None, None)
    box_mask = _box_mask((h, w), box)
    candidates = []
    non_background = [layer for layer in layers if str(layer.get("group")) not in EDIT_EXCLUDED_GROUPS]
    pool = non_background or [layer for layer in layers if str(layer.get("group")) != "effect"] or layers
    for layer in pool:
        mask = layer["mask"]
        alpha_area = float(mask.mean())
        score = alpha_area
        if target_name and str(layer.get("name")) == target_name:
            score += 10.0
        score += 2.5 * _layer_text_score(layer, prompt)
        if point_x is not None and point_y is not None and 0 <= point_x < w and 0 <= point_y < h and mask[point_y, point_x]:
            score += 4.0
        if np.any(box_mask):
            intersection = np.logical_and(mask, box_mask).sum()
            union = np.logical_or(mask, box_mask).sum()
            if union > 0:
                score += 3.0 * float(intersection / union)
        if alpha_area > 0.92 and len(pool) > 1:
            score -= 1.25
        candidates.append((score, layer))
    candidates.sort(key=lambda item: (item[0], float(item[1]["mask"].sum())), reverse=True)
    return candidates[0][1] if candidates else None


def evaluate_run_editability(
    path_or_dir: str | Path,
    *,
    prompt: str | None = None,
    point: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    target_name: str | None = None,
    output_dir: str | Path | None = None,
    write_json_metrics: bool = True,
) -> dict[str, Any]:
    manifest, layers, input_rgb = load_ordered_layers(path_or_dir)
    run_dir = _resolve_manifest_path(path_or_dir).parent
    output_root = ensure_dir(Path(output_dir) if output_dir is not None else run_dir / "debug")
    baseline = _composite_rgba_layers(layers)[..., :3]
    target = select_editable_layer(layers, prompt=prompt, point=point, box=box, target_name=target_name)
    if target is None:
        metrics = {
            "selected_target": None,
            "edit_success_score": 0.0,
            "semantic_purity": 0.0,
            "alpha_quality_score": 0.0,
            "background_hole_ratio": 1.0,
            "non_edited_region_preservation": 0.0,
            "foreground_edit_coverage": 0.0,
            "layer_alpha_overlap": 1.0,
            "occlusion_edge_count": _load_graph_edge_count(run_dir),
        }
        if write_json_metrics:
            write_json(run_dir / "editability_metrics.json", metrics)
        return metrics

    fg_layers = [layer for layer in layers if str(layer.get("group")) not in EDIT_EXCLUDED_GROUPS]
    fg_masks = [layer["mask"] for layer in fg_layers if np.any(layer["mask"])]
    union_fg = np.logical_or.reduce(fg_masks) if fg_masks else np.zeros_like(target["mask"], dtype=bool)
    foreground_edit_coverage = float(union_fg.mean()) if np.any(union_fg) else 0.0
    if fg_masks:
        denom = max(1, int(union_fg.sum()))
        largest_foreground_share = max(float(mask.sum()) / float(denom) for mask in fg_masks)
    else:
        largest_foreground_share = 1.0
    overlap_values: list[float] = []
    for idx, a in enumerate(fg_masks):
        for b in fg_masks[idx + 1 :]:
            denom = max(1, int(min(a.sum(), b.sum())))
            overlap_values.append(float(np.logical_and(a, b).sum()) / float(denom))
    layer_alpha_overlap = _safe_mean(overlap_values)

    target_mask = target["mask"]
    target_support = _dilate(target_mask, 6)
    other_layers = [layer for layer in layers if layer["name"] != target["name"]]
    remove_comp = _composite_rgba_layers(other_layers)[..., :3]
    remove_delta = _delta_map(baseline, remove_comp)
    remove_response = _mean_in_region(remove_delta, target_mask)
    background_hole_ratio = float(np.mean(remove_delta[target_mask] < 0.035)) if np.any(target_mask) else 1.0
    non_edited_region_mae_remove = _mean_in_region(remove_delta, ~target_support)

    x1, y1, x2, y2 = target["bbox"]
    h, w = target_mask.shape
    shift = max(8, min(w, h) // 10)
    room_left = x1
    room_right = w - x2
    dx = shift if room_right >= room_left else -shift
    dy = 0
    moved = dict(target)
    moved["rgba"] = _shift_rgba(target["rgba"], dx, dy)
    moved["alpha"] = moved["rgba"][..., 3].astype(np.float32) / 255.0
    moved["mask"] = _alpha_mask(moved["rgba"])
    moved_layers = [dict(layer) if layer["name"] != target["name"] else moved for layer in layers]
    move_comp = _composite_rgba_layers(moved_layers)[..., :3]
    moved_support = _dilate(np.logical_or(target_mask, moved["mask"]), 6)
    move_delta = _delta_map(baseline, move_comp)
    move_response = _mean_in_region(move_delta, moved_support)
    non_edited_region_mae_move = _mean_in_region(move_delta, ~moved_support)

    recolored = dict(target)
    recolored["rgba"] = _recolor_rgba(target["rgba"])
    recolored["alpha"] = recolored["rgba"][..., 3].astype(np.float32) / 255.0
    recolored["mask"] = target_mask
    recolor_layers = [dict(layer) if layer["name"] != target["name"] else recolored for layer in layers]
    recolor_comp = _composite_rgba_layers(recolor_layers)[..., :3]
    recolor_delta = _delta_map(baseline, recolor_comp)
    recolor_response = _mean_in_region(recolor_delta, target_support)
    non_edited_region_mae_recolor = _mean_in_region(recolor_delta, ~target_support)

    background_candidates = [layer for layer in layers if str(layer.get("group")) in BACKGROUND_GROUPS]
    if background_candidates:
        background_layer = max(background_candidates, key=lambda layer: int(layer["mask"].sum()))
        blurred_background = dict(background_layer)
        blurred_background["rgba"] = _blur_rgba(background_layer["rgba"], radius=5.0)
        blurred_background["alpha"] = blurred_background["rgba"][..., 3].astype(np.float32) / 255.0
        blur_layers = [dict(layer) if layer["name"] != background_layer["name"] else blurred_background for layer in layers]
        blur_comp = _composite_rgba_layers(blur_layers)[..., :3]
    else:
        blur_comp = baseline.copy()

    soft_pixels = 0
    fg_alpha_pixels = 0
    for layer in fg_layers:
        alpha = layer["alpha"]
        fg_alpha_pixels += int(np.count_nonzero(alpha > 0.05))
        soft_pixels += int(np.count_nonzero((alpha > 0.05) & (alpha < 0.95)))
    foreground_soft_alpha_ratio = float(soft_pixels) / float(max(1, fg_alpha_pixels))
    semantic_purity = _clip01(
        0.40 * min(1.0, foreground_edit_coverage / 0.35)
        + 0.35 * (1.0 - min(1.0, layer_alpha_overlap / 0.25))
        + 0.25 * (1.0 - max(0.0, largest_foreground_share - 0.80) / 0.20)
    )
    alpha_quality_score = _clip01(0.30 + 0.70 * min(1.0, foreground_soft_alpha_ratio / 0.08)) if fg_alpha_pixels else 0.0
    non_edited_region_preservation = _clip01(
        1.0 - _safe_mean([non_edited_region_mae_remove, non_edited_region_mae_move, non_edited_region_mae_recolor]) / 0.10
    )
    edit_success_score = _clip01(
        0.30 * min(1.0, remove_response / 0.20)
        + 0.25 * min(1.0, move_response / 0.18)
        + 0.20 * min(1.0, recolor_response / 0.12)
        + 0.15 * non_edited_region_preservation
        + 0.10 * (1.0 - background_hole_ratio)
    )

    metrics = {
        "selected_target": {
            "name": target["name"],
            "label": target["label"],
            "group": target["group"],
            "rank": int(target["rank"]),
            "bbox": [int(v) for v in target["bbox"]],
        },
        "edit_response_remove": round(remove_response, 6),
        "edit_response_move": round(move_response, 6),
        "edit_response_recolor": round(recolor_response, 6),
        "background_hole_ratio": round(background_hole_ratio, 6),
        "non_edited_region_mae_remove": round(non_edited_region_mae_remove, 6),
        "non_edited_region_mae_move": round(non_edited_region_mae_move, 6),
        "non_edited_region_mae_recolor": round(non_edited_region_mae_recolor, 6),
        "non_edited_region_preservation": round(non_edited_region_preservation, 6),
        "foreground_edit_coverage": round(foreground_edit_coverage, 6),
        "largest_foreground_share": round(largest_foreground_share, 6),
        "layer_alpha_overlap": round(layer_alpha_overlap, 6),
        "foreground_soft_alpha_ratio": round(foreground_soft_alpha_ratio, 6),
        "semantic_purity": round(semantic_purity, 6),
        "alpha_quality_score": round(alpha_quality_score, 6),
        "edit_success_score": round(edit_success_score, 6),
        "occlusion_edge_count": int(_load_graph_edge_count(run_dir)),
        "preview_paths": {
            "baseline": str((output_root / "edit_baseline.png").resolve()),
            "remove": str((output_root / "edit_remove.png").resolve()),
            "move": str((output_root / "edit_move.png").resolve()),
            "recolor": str((output_root / "edit_recolor.png").resolve()),
            "background_blur": str((output_root / "edit_background_blur.png").resolve()),
        },
    }

    save_rgb(output_root / "edit_baseline.png", baseline)
    save_rgb(output_root / "edit_remove.png", remove_comp)
    save_rgb(output_root / "edit_move.png", move_comp)
    save_rgb(output_root / "edit_recolor.png", recolor_comp)
    save_rgb(output_root / "edit_background_blur.png", blur_comp)

    if write_json_metrics:
        write_json(run_dir / "editability_metrics.json", metrics)
    return metrics


def export_target_assets(
    path_or_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    prompt: str | None = None,
    point: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    manifest, layers, _ = load_ordered_layers(path_or_dir)
    run_dir = _resolve_manifest_path(path_or_dir).parent
    output_root = ensure_dir(Path(output_dir) if output_dir is not None else run_dir / "target_extract")
    target = select_editable_layer(layers, prompt=prompt, point=point, box=box, target_name=target_name)
    if target is None:
        raise ValueError("No editable target layer could be selected")
    save_rgba(output_root / "target_rgba.png", target["rgba"])
    save_gray(output_root / "target_alpha.png", target["alpha"])

    amodal_path = run_dir / "layers_amodal_masks" / f"{target['name']}_amodal.png"
    if amodal_path.exists():
        amodal = np.asarray(Image.open(amodal_path).convert("L"), dtype=np.uint8)
        save_gray(output_root / "target_amodal_mask.png", amodal.astype(np.float32) / 255.0)

    remaining = [layer for layer in layers if layer["name"] != target["name"]]
    background_completed = _composite_rgba_layers(remaining)[..., :3]
    save_rgb(output_root / "background_completed.png", background_completed)

    moved = dict(target)
    moved["rgba"] = _shift_rgba(target["rgba"], max(8, target["rgba"].shape[1] // 10), 0)
    moved["alpha"] = moved["rgba"][..., 3].astype(np.float32) / 255.0
    moved["mask"] = _alpha_mask(moved["rgba"])
    move_layers = [dict(layer) if layer["name"] != target["name"] else moved for layer in layers]
    save_rgb(output_root / "edit_preview_move.png", _composite_rgba_layers(move_layers)[..., :3])
    save_rgb(output_root / "edit_preview_remove.png", background_completed)

    metadata = {
        "input": manifest["input"],
        "run_dir": str(run_dir),
        "selected_target": {
            "name": target["name"],
            "label": target["label"],
            "group": target["group"],
            "rank": int(target["rank"]),
            "bbox": [int(v) for v in target["bbox"]],
        },
        "prompt": prompt,
        "point": list(point) if point is not None else None,
        "box": list(box) if box is not None else None,
        "exports": {
            "target_rgba": str((output_root / "target_rgba.png").resolve()),
            "target_alpha": str((output_root / "target_alpha.png").resolve()),
            "background_completed": str((output_root / "background_completed.png").resolve()),
            "edit_preview_move": str((output_root / "edit_preview_move.png").resolve()),
            "edit_preview_remove": str((output_root / "edit_preview_remove.png").resolve()),
        },
    }
    write_json(output_root / "target_metadata.json", metadata)
    return metadata
