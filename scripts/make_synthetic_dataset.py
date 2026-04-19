#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def over_straight(bg: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    src = rgba[..., :3].astype(np.float32) / 255.0
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    base = bg.astype(np.float32) / 255.0
    return np.clip((src * a + base * (1 - a)) * 255, 0, 255).astype(np.uint8)


def make_layer(h: int, w: int, shape: str, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if shape == "ellipse":
        d.ellipse(box, fill=(*color, 255))
    elif shape == "triangle":
        x0, y0, x1, y1 = box
        d.polygon([(x0 + (x1-x0)//2, y0), (x0, y1), (x1, y1)], fill=(*color, 255))
    else:
        d.rounded_rectangle(box, radius=18, fill=(*color, 255))
    return np.array(img, dtype=np.uint8, copy=True)


def make_background(h: int, w: int) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[..., 0] = (185 - 55 * y).astype(np.uint8)
    bg[..., 1] = (215 - 60 * y).astype(np.uint8)
    bg[..., 2] = (240 - 100 * y).astype(np.uint8)
    bg[int(h*0.62):] = np.array([95, 140, 92], dtype=np.uint8)
    return bg


def rgba_full(rgb: np.ndarray) -> np.ndarray:
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])


def make_one(out: Path, seed: int, size: tuple[int, int]) -> None:
    rng = np.random.default_rng(seed)
    w, h = size
    bg = make_background(h, w)
    # Ground truth is near -> far. Lower depth means closer.
    specs = [
        ("front_table", "rect", (116, 70, 42), (int(.18*w), int(.72*h), int(.80*w), int(.95*h)), 0.10),
        ("near_person", "ellipse", (218, 80, 65), (int(.34*w), int(.38*h), int(.48*w), int(.86*h)), 0.22),
        ("mid_building", "rect", (160, 150, 142), (int(.52*w), int(.25*h), int(.88*w), int(.77*h)), 0.58),
        ("far_tree", "triangle", (50, 110, 60), (int(.08*w), int(.18*h), int(.28*w), int(.72*h)), 0.85),
        ("background_sky_ground", "background", (0, 0, 0), (0, 0, w, h), 1.00),
    ]
    layers = []
    gt_dir = out / "gt_layers"
    gt_dir.mkdir(parents=True, exist_ok=True)
    layer_arrays: list[np.ndarray] = []
    for rank, (name, shape, color, box, depth) in enumerate(specs):
        if shape == "background":
            rgba = rgba_full(bg)
        else:
            rgba = make_layer(h, w, shape, color, box)
            noise = rng.normal(0, 4, size=(h, w, 1)).astype(np.float32)
            rgba[..., :3] = np.clip(rgba[..., :3].astype(np.float32) + noise * (rgba[..., 3:4] > 0), 0, 255).astype(np.uint8)
        path = gt_dir / f"{rank:03d}_{name}.png"
        Image.fromarray(rgba, mode="RGBA").save(path)
        layer_arrays.append(rgba)
        layers.append({"name": name, "path": str(Path("gt_layers") / path.name), "rank_near_to_far": rank, "depth": depth})
    image = np.zeros_like(bg)
    # Draw far -> near: background first, closest object last.
    for rgba in reversed(layer_arrays):
        image = over_straight(image, rgba)
    Image.fromarray(image, mode="RGB").save(out / "image.png")
    with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump({"image_path": "image.png", "layers_near_to_far": layers}, f, indent=2)


def make_dataset(root: Path, count: int, seed: int, size: tuple[int, int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        make_one(root / f"scene_{i:03d}", seed + i, size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=420)
    args = ap.parse_args()
    make_dataset(Path(args.output), args.count, args.seed, (args.width, args.height))
    print(args.output)


if __name__ == "__main__":
    main()
