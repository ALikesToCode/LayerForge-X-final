#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class SceneSpec:
    label: str
    shape: str
    alpha: int
    color: tuple[int, int, int]
    box: tuple[int, int, int, int]


def make_background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[..., 0] = np.clip(150 + 55 * x - 20 * y, 0, 255).astype(np.uint8)
    background[..., 1] = np.clip(175 + 35 * y, 0, 255).astype(np.uint8)
    background[..., 2] = np.clip(205 + 25 * (1.0 - x), 0, 255).astype(np.uint8)
    block_color = np.array([55, 70, 95], dtype=np.uint8)
    x0 = int(width * 0.12)
    x1 = int(width * 0.36)
    y0 = int(height * 0.18)
    y1 = int(height * 0.64)
    background[y0:y1, x0:x1] = block_color
    yy = slice(int(height * 0.62), int(height * 0.88))
    xx = slice(int(width * 0.52), int(width * 0.84))
    background[yy, xx] = np.array([235, 210, 160], dtype=np.uint8)
    noise = rng.normal(0, 3, size=background.shape).astype(np.float32)
    return np.clip(background.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_foreground(width: int, height: int, spec: SceneSpec) -> np.ndarray:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = (*spec.color, spec.alpha)
    if spec.shape == "ellipse":
        draw.ellipse(spec.box, fill=fill)
    elif spec.shape == "rounded_rect":
        draw.rounded_rectangle(spec.box, radius=18, fill=fill)
    elif spec.shape == "ring":
        draw.ellipse(spec.box, fill=fill)
        inner = (
            spec.box[0] + 18,
            spec.box[1] + 18,
            spec.box[2] - 18,
            spec.box[3] - 18,
        )
        draw.ellipse(inner, fill=(0, 0, 0, 0))
    else:
        draw.rectangle(spec.box, fill=fill)
    return np.asarray(img, dtype=np.uint8)


def composite(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    bg = background.astype(np.float32) / 255.0
    fg = foreground[..., :3].astype(np.float32) / 255.0
    alpha = foreground[..., 3:4].astype(np.float32) / 255.0
    return np.clip((fg * alpha + bg * (1.0 - alpha)) * 255.0, 0, 255).astype(np.uint8)


def scene_spec(index: int, width: int, height: int) -> SceneSpec:
    variants = [
        SceneSpec("glass_overlay", "ellipse", 96, (188, 222, 255), (int(width * 0.22), int(height * 0.18), int(width * 0.56), int(height * 0.78))),
        SceneSpec("transparent_sticker", "rounded_rect", 122, (245, 146, 72), (int(width * 0.30), int(height * 0.24), int(width * 0.72), int(height * 0.72))),
        SceneSpec("flare_ring", "ring", 88, (255, 225, 140), (int(width * 0.42), int(height * 0.12), int(width * 0.90), int(height * 0.70))),
        SceneSpec("semi_transparent_panel", "rect", 110, (126, 184, 255), (int(width * 0.16), int(height * 0.28), int(width * 0.82), int(height * 0.82))),
    ]
    return variants[index % len(variants)]


def make_dataset(root: Path, count: int, width: int, height: int, seed: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for idx in range(count):
        scene_dir = root / f"scene_{idx:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        bg = make_background(width, height, rng)
        spec = scene_spec(idx, width, height)
        fg = make_foreground(width, height, spec)
        alpha = fg[..., 3].astype(np.float32) / 255.0
        image = composite(bg, fg)

        Image.fromarray(image, mode="RGB").save(scene_dir / "input.png")
        Image.fromarray(bg, mode="RGB").save(scene_dir / "background.png")
        Image.fromarray(fg, mode="RGBA").save(scene_dir / "foreground_rgba.png")
        Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L").save(scene_dir / "alpha_map.png")
        (scene_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "label": spec.label,
                    "shape": spec.shape,
                    "alpha": spec.alpha / 255.0,
                    "box": list(spec.box),
                    "input": "input.png",
                    "background": "background.png",
                    "foreground_rgba": "foreground_rgba.png",
                    "alpha_map": "alpha_map.png",
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a small synthetic transparent/alpha-composited benchmark dataset.")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    make_dataset(Path(args.output), args.count, args.width, args.height, args.seed)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
