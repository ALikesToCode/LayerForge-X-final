from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import Layer, Segment
from .utils import normalize01


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    g = (normalize01(depth, robust=False) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def heatmap_rgb(values: np.ndarray) -> np.ndarray:
    v = normalize01(values.astype(np.float32), robust=True)
    red = np.clip(v * 255.0, 0, 255)
    green = np.clip((1.0 - np.abs(v - 0.5) * 2.0) * 180.0, 0, 180)
    blue = np.clip((1.0 - v) * 255.0, 0, 255)
    return np.dstack([red, green, blue]).astype(np.uint8)


def recomposition_error_heatmap(input_rgb: np.ndarray, recomposed_rgb: np.ndarray) -> np.ndarray:
    src = input_rgb.astype(np.float32)
    rec = recomposed_rgb.astype(np.float32)
    if src.shape != rec.shape:
        raise ValueError(f"recomposition heatmap shape mismatch: {src.shape} != {rec.shape}")
    err = np.mean(np.abs(src - rec), axis=2) / 255.0
    return heatmap_rgb(err)


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


def _mask_to_rgba(mask: np.ndarray, rgb: tuple[int, int, int] = (40, 120, 220)) -> np.ndarray:
    alpha = (np.clip(mask.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    out = np.zeros((*mask.shape, 4), dtype=np.uint8)
    out[..., :3] = np.array(rgb, dtype=np.uint8)
    out[..., 3] = alpha
    return out


def layer_surface_rgba(layer: Layer, surface: str) -> np.ndarray:
    key = str(surface).lower()
    if key in {"rgba", "visible"}:
        return layer.rgba
    if key == "completed":
        return layer.completed_rgba if layer.completed_rgba is not None else layer.rgba
    if key == "alpha":
        return _mask_to_rgba(layer.alpha, (30, 30, 30))
    if key == "amodal":
        return _mask_to_rgba(layer.amodal_mask.astype(np.float32), (40, 120, 220)) if layer.amodal_mask is not None else _mask_to_rgba(np.zeros_like(layer.alpha), (40, 120, 220))
    if key == "hidden":
        return _mask_to_rgba(layer.hidden_mask.astype(np.float32), (220, 90, 40)) if layer.hidden_mask is not None else _mask_to_rgba(np.zeros_like(layer.alpha), (220, 90, 40))
    if key == "albedo":
        return layer.albedo_rgba
    if key == "shading":
        return layer.shading_rgba
    if key == "depth":
        depth_rgba = layer.metadata.get("depth_crop_rgba")
        if isinstance(depth_rgba, np.ndarray):
            return depth_rgba
    raise ValueError(f"Unknown layer contact-sheet surface: {surface}")


def save_layer_surface_contact_sheet(path: str | Path, layers: list[Layer], surface: str, thumb: int = 160) -> Path:
    proxy_layers: list[Layer] = []
    for layer in layers:
        proxy_layers.append(
            Layer(
                id=layer.id,
                name=layer.name,
                label=layer.label,
                group=layer.group,
                rank=layer.rank,
                depth_median=layer.depth_median,
                depth_p10=layer.depth_p10,
                depth_p90=layer.depth_p90,
                area=layer.area,
                bbox=layer.bbox,
                alpha=layer.alpha,
                rgba=layer_surface_rgba(layer, surface),
                albedo_rgba=layer.albedo_rgba,
                shading_rgba=layer.shading_rgba,
                visible_mask=layer.visible_mask,
                amodal_mask=layer.amodal_mask,
                source_segment_ids=layer.source_segment_ids,
                occludes=layer.occludes,
                occluded_by=layer.occluded_by,
                metadata=layer.metadata,
                hidden_mask=layer.hidden_mask,
                completed_rgba=layer.completed_rgba,
            )
        )
    return save_layer_contact_sheet(path, proxy_layers, thumb=thumb)


def save_depth_crop_contact_sheet(path: str | Path, layers: list[Layer], depth: np.ndarray, thumb: int = 160) -> Path:
    depth_rgb = heatmap_rgb(depth)
    proxy_layers: list[Layer] = []
    for layer in layers:
        alpha = np.clip(layer.alpha.astype(np.float32), 0.0, 1.0)
        rgba = np.zeros((*depth.shape, 4), dtype=np.uint8)
        rgba[..., :3] = depth_rgb
        rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        metadata = dict(layer.metadata)
        metadata["depth_crop_rgba"] = rgba
        proxy_layers.append(
            Layer(
                id=layer.id,
                name=layer.name,
                label=layer.label,
                group=layer.group,
                rank=layer.rank,
                depth_median=layer.depth_median,
                depth_p10=layer.depth_p10,
                depth_p90=layer.depth_p90,
                area=layer.area,
                bbox=layer.bbox,
                alpha=layer.alpha,
                rgba=rgba,
                albedo_rgba=layer.albedo_rgba,
                shading_rgba=layer.shading_rgba,
                visible_mask=layer.visible_mask,
                amodal_mask=layer.amodal_mask,
                source_segment_ids=layer.source_segment_ids,
                occludes=layer.occludes,
                occluded_by=layer.occluded_by,
                metadata=metadata,
                hidden_mask=layer.hidden_mask,
                completed_rgba=layer.completed_rgba,
            )
        )
    return save_layer_contact_sheet(path, proxy_layers, thumb=thumb)
