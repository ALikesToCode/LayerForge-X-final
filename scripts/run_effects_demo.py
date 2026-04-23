#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from layerforge.config import load_config
from layerforge.peeling import extract_associated_effect_layer
from layerforge.utils import mask_iou, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an auditable associated-effect-layer demo on a layerbench_pp scene.")
    p.add_argument("--scene-dir", default="runs/effects_demo_source/scene_000")
    p.add_argument("--output", default="runs/effects_groundtruth_demo")
    p.add_argument("--config", default="configs/fast.yaml")
    return p.parse_args()


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def composite_rgba_stack(paths_far_to_near: list[Path]) -> np.ndarray:
    canvas = np.zeros((*load_rgba(paths_far_to_near[0]).shape[:2], 3), dtype=np.float32)
    for path in paths_far_to_near:
        rgba = load_rgba(path).astype(np.float32)
        alpha = rgba[..., 3:4] / 255.0
        canvas = rgba[..., :3] * alpha + canvas * (1.0 - alpha)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_gray(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(path)


def build_strip(frames: list[np.ndarray]) -> Image.Image:
    imgs = [Image.fromarray(frame, mode="RGB") for frame in frames]
    margin = 8
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
    canvas = Image.new("RGB", (w, h + 2 * margin), "white")
    x = margin
    for im in imgs:
        y = margin + (h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + margin
    return canvas


def preview_rgba(rgba: np.ndarray) -> np.ndarray:
    arr = rgba.astype(np.float32)
    alpha = arr[..., 3:4] / 255.0
    bg = np.full(arr[..., :3].shape, 245.0, dtype=np.float32)
    return np.clip(arr[..., :3] * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    effect_cfg = dict(cfg.get("effects", {}))
    effect_cfg["use_provided_reference"] = True
    effect_cfg["delta_threshold"] = min(float(effect_cfg.get("delta_threshold", 0.015)), 0.015)
    effect_cfg["alpha_scale"] = min(float(effect_cfg.get("alpha_scale", 0.08)), 0.08)
    effect_cfg["dilate_px"] = max(int(effect_cfg.get("dilate_px", 36)), 36)
    effect_cfg["support_dilate_px"] = 0
    scene_dir = Path(args.scene_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((scene_dir / "scene_metadata.json").read_text(encoding="utf-8"))
    layer_entries = metadata["layers_near_to_far"]
    name_to_entry = {entry["name"]: entry for entry in layer_entries}

    core_entry = name_to_entry["near_person"]
    effect_entry = name_to_entry["near_person_shadow"]
    background_entries = [entry for entry in layer_entries if entry["name"] not in {"near_person", "near_person_shadow"}]

    core_path = scene_dir / core_entry["rgba_path"]
    effect_path = scene_dir / effect_entry["rgba_path"]
    background_paths = [scene_dir / entry["rgba_path"] for entry in background_entries]

    current_rgb = np.asarray(Image.open(scene_dir / "image.png").convert("RGB"), dtype=np.uint8)
    clean_reference = composite_rgba_stack(list(reversed(background_paths)))
    core_rgba = load_rgba(core_path)
    effect_rgba = load_rgba(effect_path)
    core_mask = (core_rgba[..., 3] > 12).astype(bool)
    gt_effect_mask = effect_rgba[..., 3] > 12

    effect_layer = extract_associated_effect_layer(
        current_rgb=current_rgb,
        inpainted_rgb=clean_reference,
        core_mask=core_mask,
        label="near_person",
        rank=1,
        cfg=effect_cfg,
    )

    save_rgb(out / "input_rgb.png", current_rgb)
    save_rgb(out / "reference_without_object.png", clean_reference)
    save_gray(out / "core_mask.png", core_mask)
    Image.fromarray(effect_rgba, mode="RGBA").save(out / "ground_truth_effect_rgba.png")
    save_gray(out / "ground_truth_effect_mask.png", gt_effect_mask)

    metrics = {
        "scene_dir": str(scene_dir),
        "effect_detected": bool(effect_layer is not None),
        "ground_truth_effect_pixels": int(gt_effect_mask.sum()),
        "effect_config": effect_cfg,
    }
    strip_frames = [current_rgb, clean_reference, preview_rgba(effect_rgba)]

    if effect_layer is not None:
        Image.fromarray(effect_layer.rgba, mode="RGBA").save(out / "predicted_effect_rgba.png")
        save_gray(out / "predicted_effect_mask.png", effect_layer.visible_mask)
        metrics.update(
            {
                "predicted_effect_pixels": int(effect_layer.visible_mask.sum()),
                "effect_iou": float(mask_iou(effect_layer.visible_mask, gt_effect_mask)),
                "effect_delta_mean": float(effect_layer.metadata.get("delta_mean", 0.0)),
            }
        )
        strip_frames.append(preview_rgba(effect_layer.rgba))
    else:
        metrics.update(
            {
                "predicted_effect_pixels": 0,
                "effect_iou": 0.0,
                "effect_delta_mean": 0.0,
            }
        )
        strip_frames.append(np.full_like(current_rgb, 245, dtype=np.uint8))

    strip = build_strip(strip_frames)
    strip.save(out / "effect_demo_strip.png")
    write_json(out / "metrics.json", metrics)
    print(json.dumps({"output_dir": str(out), "metrics": str(out / "metrics.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
