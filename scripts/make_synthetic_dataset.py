#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class LayerSpec:
    name: str
    shape: str
    color: tuple[int, int, int]
    box: tuple[int, int, int, int]
    depth: float
    kind: str = "object"
    alpha: int = 255


def over_straight(bg: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    src = rgba[..., :3].astype(np.float32) / 255.0
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    base = bg.astype(np.float32) / 255.0
    return np.clip((src * a + base * (1 - a)) * 255, 0, 255).astype(np.uint8)


def make_layer(h: int, w: int, shape: str, color: tuple[int, int, int], box: tuple[int, int, int, int], alpha: int = 255) -> np.ndarray:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = (*color, int(np.clip(alpha, 0, 255)))
    if shape == "ellipse":
        d.ellipse(box, fill=fill)
    elif shape == "triangle":
        x0, y0, x1, y1 = box
        d.polygon([(x0 + (x1 - x0) // 2, y0), (x0, y1), (x1, y1)], fill=fill)
    else:
        d.rounded_rectangle(box, radius=18, fill=fill)
    return np.array(img, dtype=np.uint8, copy=True)


def make_background(h: int, w: int) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[..., 0] = (185 - 55 * y).astype(np.uint8)
    bg[..., 1] = (215 - 60 * y).astype(np.uint8)
    bg[..., 2] = (240 - 100 * y).astype(np.uint8)
    bg[int(h * 0.62) :] = np.array([95, 140, 92], dtype=np.uint8)
    return bg


def rgba_full(rgb: np.ndarray) -> np.ndarray:
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])


def build_specs(w: int, h: int, *, with_effects: bool, output_format: str) -> list[LayerSpec]:
    if output_format == "basic":
        return [
            LayerSpec("front_table", "rect", (116, 70, 42), (int(0.18 * w), int(0.72 * h), int(0.80 * w), int(0.95 * h)), 0.10),
            LayerSpec("near_person", "ellipse", (218, 80, 65), (int(0.34 * w), int(0.38 * h), int(0.48 * w), int(0.86 * h)), 0.22),
            LayerSpec("mid_building", "rect", (160, 150, 142), (int(0.52 * w), int(0.25 * h), int(0.88 * w), int(0.77 * h)), 0.58),
            LayerSpec("far_tree", "triangle", (50, 110, 60), (int(0.08 * w), int(0.18 * h), int(0.28 * w), int(0.72 * h)), 0.85),
            LayerSpec("background_sky_ground", "background", (0, 0, 0), (0, 0, w, h), 1.00, kind="background"),
        ]

    specs = [
        LayerSpec("near_person", "ellipse", (220, 88, 70), (int(0.34 * w), int(0.34 * h), int(0.50 * w), int(0.86 * h)), 0.10, kind="core"),
        LayerSpec("front_table", "rect", (126, 78, 46), (int(0.16 * w), int(0.72 * h), int(0.82 * w), int(0.95 * h)), 0.18),
        LayerSpec("mid_building", "rect", (162, 152, 144), (int(0.52 * w), int(0.22 * h), int(0.90 * w), int(0.77 * h)), 0.58),
        LayerSpec("far_tree", "triangle", (54, 112, 65), (int(0.06 * w), int(0.18 * h), int(0.28 * w), int(0.74 * h)), 0.85),
        LayerSpec("background_sky_ground", "background", (0, 0, 0), (0, 0, w, h), 1.00, kind="background"),
    ]
    if with_effects:
        specs.insert(
            1,
            LayerSpec(
                "near_person_shadow",
                "ellipse",
                (32, 24, 20),
                (int(0.28 * w), int(0.72 * h), int(0.60 * w), int(0.92 * h)),
                0.14,
                kind="effect",
                alpha=104,
            ),
        )
    return specs


def make_shading_field(h: int, w: int) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    field = 0.78 + 0.18 * (1.0 - y) + 0.04 * np.sin(2.5 * np.pi * x)
    return np.clip(field, 0.65, 1.0).astype(np.float32)


def apply_shading(rgba: np.ndarray, shading_field: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    out[..., :3] = np.clip(out[..., :3].astype(np.float32) * shading_field[..., None], 0, 255).astype(np.uint8)
    return out


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_alpha(path: Path, alpha: np.ndarray) -> None:
    Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def compute_visible_masks(layer_arrays: list[np.ndarray]) -> list[np.ndarray]:
    occupied = np.zeros(layer_arrays[0].shape[:2], dtype=bool) if layer_arrays else np.zeros((1, 1), dtype=bool)
    visible: list[np.ndarray] = []
    for rgba in layer_arrays:
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        amodal = alpha > 0.05
        vis = amodal & ~occupied
        visible.append(vis)
        occupied |= alpha > 0.5
    return visible


def build_occlusion_edges(specs: list[LayerSpec], visible_masks: list[np.ndarray], amodal_masks: list[np.ndarray]) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for near_idx, near_spec in enumerate(specs):
        for far_idx in range(near_idx + 1, len(specs)):
            overlap = visible_masks[near_idx] & amodal_masks[far_idx]
            if not overlap.any():
                continue
            edges.append(
                {
                    "near": near_spec.name,
                    "far": specs[far_idx].name,
                    "overlap_pixels": int(overlap.sum()),
                }
            )
    return edges


def build_depth_map(specs: list[LayerSpec], layer_arrays: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    depth = np.ones((h, w), dtype=np.float32)
    for spec, rgba in reversed(list(zip(specs, layer_arrays))):
        mask = rgba[..., 3] > 0
        depth[mask] = np.float32(spec.depth)
    return depth


def export_layerbench_pp(
    out: Path,
    *,
    specs: list[LayerSpec],
    albedo_layers: list[np.ndarray],
    shaded_layers: list[np.ndarray],
    image: np.ndarray,
    albedo_rgb: np.ndarray,
    shading_field: np.ndarray,
) -> None:
    visible_dir = out / "visible_masks"
    amodal_dir = out / "amodal_masks"
    alpha_dir = out / "alpha_mattes"
    effects_dir = out / "layers_effects_rgba"
    intrinsics_dir = out / "intrinsics"
    for path in [visible_dir, amodal_dir, alpha_dir, effects_dir, intrinsics_dir]:
        path.mkdir(parents=True, exist_ok=True)

    visible_masks = compute_visible_masks(shaded_layers)
    amodal_masks = [(layer[..., 3].astype(np.float32) / 255.0) > 0.05 for layer in shaded_layers]
    occlusion_edges = build_occlusion_edges(specs, visible_masks, amodal_masks)

    layers_meta: list[dict[str, object]] = []
    for idx, (spec, albedo_rgba, shaded_rgba, visible, amodal) in enumerate(zip(specs, albedo_layers, shaded_layers, visible_masks, amodal_masks)):
        alpha = shaded_rgba[..., 3].astype(np.float32) / 255.0
        base_name = f"{idx:03d}_{spec.name}"
        save_mask(visible_dir / f"{base_name}.png", visible)
        save_mask(amodal_dir / f"{base_name}.png", amodal)
        save_alpha(alpha_dir / f"{base_name}.png", alpha)
        if spec.kind == "effect":
            Image.fromarray(shaded_rgba, mode="RGBA").save(effects_dir / f"{base_name}.png")
        layers_meta.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "rank_near_to_far": idx,
                "depth": spec.depth,
                "visible_pixels": int(visible.sum()),
                "amodal_pixels": int(amodal.sum()),
                "alpha_path": str(Path("alpha_mattes") / f"{base_name}.png"),
                "visible_mask_path": str(Path("visible_masks") / f"{base_name}.png"),
                "amodal_mask_path": str(Path("amodal_masks") / f"{base_name}.png"),
                "rgba_path": str(Path("gt_layers") / f"{base_name}.png"),
            }
        )

    depth = build_depth_map(specs, shaded_layers, image.shape[:2])
    depth_png = np.clip(depth * 65535, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_png).save(out / "depth.png")
    np.save(out / "depth.npy", depth)

    Image.fromarray(albedo_rgb, mode="RGB").save(intrinsics_dir / "albedo.png")
    shade_rgb = np.clip(shading_field[..., None] * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(np.repeat(shade_rgb, 3, axis=2), mode="RGB").save(intrinsics_dir / "shading.png")

    with (out / "occlusion_graph.json").open("w", encoding="utf-8") as f:
        json.dump({"edges": occlusion_edges}, f, indent=2)
    with (out / "scene_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_format": "layerbench_pp",
                "with_effects": any(spec.kind == "effect" for spec in specs),
                "image_path": "image.png",
                "depth_path": "depth.npy",
                "layers_near_to_far": layers_meta,
            },
            f,
            indent=2,
        )


def make_one(out: Path, seed: int, size: tuple[int, int], *, output_format: str, with_effects: bool) -> None:
    rng = np.random.default_rng(seed)
    w, h = size
    bg = make_background(h, w)
    specs = build_specs(w, h, with_effects=with_effects and output_format != "basic", output_format=output_format)

    shading_field = make_shading_field(h, w) if output_format != "basic" else np.ones((h, w), dtype=np.float32)
    layers = []
    gt_dir = out / "gt_layers"
    gt_dir.mkdir(parents=True, exist_ok=True)
    albedo_layers: list[np.ndarray] = []
    shaded_layers: list[np.ndarray] = []

    for rank, spec in enumerate(specs):
        if spec.shape == "background":
            albedo_rgba = rgba_full(bg)
        else:
            albedo_rgba = make_layer(h, w, spec.shape, spec.color, spec.box, alpha=spec.alpha)
            noise = rng.normal(0, 4, size=(h, w, 1)).astype(np.float32)
            albedo_rgba[..., :3] = np.clip(albedo_rgba[..., :3].astype(np.float32) + noise * (albedo_rgba[..., 3:4] > 0), 0, 255).astype(np.uint8)
        shaded_rgba = apply_shading(albedo_rgba, shading_field) if spec.kind != "effect" else albedo_rgba.copy()
        path = gt_dir / f"{rank:03d}_{spec.name}.png"
        Image.fromarray(shaded_rgba, mode="RGBA").save(path)
        albedo_layers.append(albedo_rgba)
        shaded_layers.append(shaded_rgba)
        layers.append(
            {
                "name": spec.name,
                "path": str(Path("gt_layers") / path.name),
                "rank_near_to_far": rank,
                "depth": spec.depth,
                "kind": spec.kind,
            }
        )

    image = np.zeros_like(bg)
    albedo_rgb = np.zeros_like(bg)
    for rgba in reversed(shaded_layers):
        image = over_straight(image, rgba)
    for rgba in reversed(albedo_layers):
        albedo_rgb = over_straight(albedo_rgb, rgba)

    Image.fromarray(image, mode="RGB").save(out / "image.png")
    with (out / "ground_truth.json").open("w", encoding="utf-8") as f:
        json.dump({"image_path": "image.png", "layers_near_to_far": layers}, f, indent=2)

    if output_format != "basic":
        export_layerbench_pp(
            out,
            specs=specs,
            albedo_layers=albedo_layers,
            shaded_layers=shaded_layers,
            image=image,
            albedo_rgb=albedo_rgb,
            shading_field=shading_field,
        )


def make_dataset(root: Path, count: int, seed: int, size: tuple[int, int], *, output_format: str, with_effects: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        make_one(root / f"scene_{i:03d}", seed + i, size, output_format=output_format, with_effects=with_effects)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--output-format", default="basic", choices=["basic", "layerbench_pp"])
    ap.add_argument("--with-effects", action="store_true", help="Only used with --output-format layerbench_pp")
    args = ap.parse_args()
    make_dataset(
        Path(args.output),
        args.count,
        args.seed,
        (args.width, args.height),
        output_format=args.output_format,
        with_effects=bool(args.with_effects),
    )
    print(args.output)


if __name__ == "__main__":
    main()
