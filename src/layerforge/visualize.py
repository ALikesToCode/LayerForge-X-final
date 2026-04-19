from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import Layer, Segment
from .utils import normalize01


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    g = (normalize01(depth, robust=False) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def segmentation_overlay(rgb: np.ndarray, segments: list[Segment]) -> np.ndarray:
    import cv2
    out = rgb.copy()
    palette = np.array([[255,80,80],[80,220,120],[80,120,255],[255,210,80],[230,80,230],[80,220,230],[220,150,80]], dtype=np.uint8)
    for i, s in enumerate(segments):
        cnts, _ = cv2.findContours(s.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, palette[i % len(palette)].tolist(), 2)
    return out


def draw_segment_labels(rgb: np.ndarray, segments: list[Segment]) -> np.ndarray:
    import cv2
    out = rgb.copy()
    for s in segments:
        x0, y0, _, _ = s.bbox
        cv2.putText(out, f"{s.id}:{s.group}", (x0, max(15, y0 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{s.id}:{s.group}", (x0, max(15, y0 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    return out


def save_layer_contact_sheet(path: str | Path, layers: list[Layer], thumb: int = 160) -> Path:
    if not layers:
        return Path(path)
    cols = min(4, len(layers))
    rows = int(np.ceil(len(layers) / cols))
    pad, title_h = 12, 28
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + title_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for i, layer in enumerate(layers):
        im = Image.fromarray(layer.rgba, mode="RGBA")
        im.thumbnail((thumb, thumb))
        x = pad + (i % cols) * (thumb + pad)
        y = pad + (i // cols) * (thumb + title_h + pad)
        checker = Image.new("RGB", (thumb, thumb), "white")
        cd = ImageDraw.Draw(checker)
        t = 12
        for yy in range(0, thumb, t):
            for xx in range(0, thumb, t):
                if (xx//t + yy//t) % 2 == 0:
                    cd.rectangle([xx, yy, xx+t-1, yy+t-1], fill=(225,225,225))
        checker.paste(im, ((thumb - im.width)//2, (thumb - im.height)//2), im)
        sheet.paste(checker, (x, y + title_h))
        draw.text((x, y), layer.name[:28], fill=(0,0,0), font=font)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(p)
    return p
