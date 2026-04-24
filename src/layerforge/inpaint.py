from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .utils import optional_import


def _opencv_inpaint(rgb: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    cv2 = optional_import("cv2")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _load_simple_lama():
    from simple_lama_inpainting import SimpleLama

    return SimpleLama


def _external_inpaint(rgb: np.ndarray, mask: np.ndarray, command: str, prompt: str = "") -> np.ndarray:
    if not command.strip():
        raise RuntimeError("external inpainting requires inpainting.external_command")
    with tempfile.TemporaryDirectory(prefix="layerforge_inpaint_") as tmp:
        tmp_dir = Path(tmp)
        image_path = tmp_dir / "image.png"
        mask_path = tmp_dir / "mask.png"
        output_path = tmp_dir / "output.png"
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        formatted = command.format(
            image=image_path,
            mask=mask_path,
            output=output_path,
            output_dir=tmp_dir,
            prompt=prompt,
        )
        subprocess.run(shlex.split(formatted), check=True)
        if not output_path.exists():
            raise RuntimeError("external inpainting command did not write output image")
        output = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
        if output.shape[:2] != rgb.shape[:2]:
            output = np.asarray(Image.fromarray(output).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR), dtype=np.uint8)
        return output


def inpaint_background(rgb: np.ndarray, mask: np.ndarray, cfg: dict[str, Any], device: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    method = str(cfg.get("method", "opencv_telea")).lower()
    m = mask.astype(bool)
    if not m.any():
        return rgb.copy(), m, "none"
    if method in {"auto", "opencv", "opencv_telea", "telea"}:
        out = _opencv_inpaint(rgb, m, float(cfg.get("radius", 5)))
        return out, m, "opencv_telea" if method != "auto" else "opencv_telea_auto"
    if method in {"lama", "simple_lama"}:
        try:
            SimpleLama = _load_simple_lama()
        except Exception:
            out = _opencv_inpaint(rgb, m, float(cfg.get("radius", 5)))
            return out, m, "opencv_telea_fallback"
        lama = SimpleLama()
        out = lama(Image.fromarray(rgb), Image.fromarray((m.astype(np.uint8) * 255), mode="L"))
        return np.asarray(out.convert("RGB"), dtype=np.uint8), m, "lama"
    if method == "external":
        try:
            out = _external_inpaint(rgb, m, str(cfg.get("external_command", "")), str(cfg.get("prompt", "")))
            return out, m, "external"
        except Exception:
            out = _opencv_inpaint(rgb, m, float(cfg.get("radius", 5)))
            return out, m, "opencv_telea_fallback"
    if method == "diffusion":
        out = _opencv_inpaint(rgb, m, float(cfg.get("radius", 5)))
        return out, m, "opencv_telea_fallback"
    raise ValueError(f"Unknown inpainting method: {method}")


def inpainting_quality_metrics(original_rgb: np.ndarray, completed_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    m = mask.astype(bool)
    if not m.any():
        return {
            "boundary_consistency": 1.0,
            "masked_region_texture_continuity": 1.0,
            "recomposition_residual_outside_mask": 0.0,
        }
    cv2 = optional_import("cv2")
    dilated = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    boundary = dilated & ~m
    if boundary.any():
        boundary_delta = np.mean(np.abs(original_rgb[boundary].astype(np.float32) - completed_rgb[boundary].astype(np.float32))) / 255.0
    else:
        boundary_delta = 0.0
    outside = ~m
    outside_residual = float(np.mean(np.abs(original_rgb[outside].astype(np.float32) - completed_rgb[outside].astype(np.float32))) / 255.0) if outside.any() else 0.0
    masked_vals = completed_rgb[m].astype(np.float32)
    boundary_vals = completed_rgb[boundary].astype(np.float32) if boundary.any() else completed_rgb[outside].astype(np.float32)
    if masked_vals.size and boundary_vals.size:
        texture_delta = float(np.linalg.norm(masked_vals.mean(axis=0) - boundary_vals.mean(axis=0)) / (255.0 * np.sqrt(3)))
    else:
        texture_delta = 0.0
    return {
        "boundary_consistency": float(np.clip(1.0 - boundary_delta, 0.0, 1.0)),
        "masked_region_texture_continuity": float(np.clip(1.0 - texture_delta, 0.0, 1.0)),
        "recomposition_residual_outside_mask": outside_residual,
    }


def complete_hidden_layer(
    rgb: np.ndarray,
    alpha: np.ndarray,
    visible_mask: np.ndarray,
    hidden_mask: np.ndarray | None,
    cfg: dict[str, Any] | None,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    hidden = np.zeros(alpha.shape, dtype=bool) if hidden_mask is None else hidden_mask.astype(bool)
    completed_alpha = np.maximum(alpha.astype(np.float32), hidden.astype(np.float32))
    if not hidden.any():
        return np.dstack([rgb, (completed_alpha * 255).clip(0, 255).astype(np.uint8)]), {
            "method": "none",
            "hidden_pixels": 0,
            **inpainting_quality_metrics(rgb, rgb, hidden),
        }

    layer_rgb = np.zeros_like(rgb)
    visible = visible_mask.astype(bool)
    layer_rgb[visible] = rgb[visible]
    if visible.any():
        fill_rgb = np.median(rgb[visible], axis=0).clip(0, 255).astype(np.uint8)
        layer_rgb[hidden] = fill_rgb
    completed_rgb, _, method = inpaint_background(layer_rgb, hidden, cfg or {"method": "auto"}, device=device)
    completed_rgb[visible] = rgb[visible]
    metrics = inpainting_quality_metrics(layer_rgb, completed_rgb, hidden)
    return np.dstack([completed_rgb, (completed_alpha * 255).clip(0, 255).astype(np.uint8)]), {
        "method": method,
        "hidden_pixels": int(hidden.sum()),
        **metrics,
    }
