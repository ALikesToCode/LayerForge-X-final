from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .compose import rgba_from_rgb_alpha
from .utils import image_to_float, normalize01, optional_import


def retinex_decompose(rgb: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cv2 = optional_import("cv2")
    img = np.clip(image_to_float(rgb), 1e-4, 1.0)
    sigma = float(cfg.get("sigma", 28.0))
    log_img = np.log(img)
    smooth = cv2.GaussianBlur(log_img, (0, 0), sigma)
    reflectance = log_img - smooth
    albedo = np.zeros_like(img)
    for c in range(3):
        albedo[..., c] = normalize01(reflectance[..., c], robust=True)
    shading_scalar = np.exp(np.mean(smooth, axis=2))
    shading_scalar = normalize01(shading_scalar, robust=True)
    shading = np.repeat(shading_scalar[..., None], 3, axis=2)
    return (albedo * 255).astype(np.uint8), (shading * 255).astype(np.uint8)


def external_decompose(rgb: np.ndarray, command: str) -> tuple[np.ndarray, np.ndarray]:
    if not command.strip():
        raise RuntimeError("external intrinsic decomposition requires intrinsics.external_command")
    with tempfile.TemporaryDirectory(prefix="layerforge_iid_") as tmp:
        d = Path(tmp)
        inp, alb, shd = d / "input.png", d / "albedo.png", d / "shading.png"
        Image.fromarray(rgb).save(inp)
        subprocess.run(shlex.split(command.format(input=inp, albedo=alb, shading=shd)), check=True)
        if not alb.exists() or not shd.exists():
            raise RuntimeError("external intrinsic command did not write albedo and shading")
        return np.asarray(Image.open(alb).convert("RGB")), np.asarray(Image.open(shd).convert("RGB"))


def identity_decompose(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rgb.copy(), np.full_like(rgb, 255, dtype=np.uint8)


def intrinsic_residual(rgb: np.ndarray, albedo_rgb: np.ndarray, shading_rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    target = image_to_float(rgb)
    reconstructed = image_to_float(albedo_rgb) * np.clip(image_to_float(shading_rgb), 0.0, 1.0)
    if mask is not None:
        m = mask.astype(bool)
        if not m.any():
            return 0.0
        target = target[m]
        reconstructed = reconstructed[m]
    return float(np.mean(np.abs(target - reconstructed)))


def decompose_intrinsics(rgb: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    method = str(cfg.get("method", "retinex")).lower()
    if method in {"none", "off", "disabled"}:
        a, s = identity_decompose(rgb)
        return a, s, "none"
    if method in {"auto", "retinex"}:
        a, s = retinex_decompose(rgb, cfg)
        return a, s, "retinex" if method == "retinex" else "retinex_auto"
    if method in {"ordinal", "intrinsic_model"}:
        a, s = retinex_decompose(rgb, cfg)
        return a, s, "retinex_fallback"
    if method in {"external", "marigold", "marigold_iid"}:
        try:
            a, s = external_decompose(rgb, str(cfg.get("external_command", "")))
            return a, s, "external"
        except Exception:
            a, s = retinex_decompose(rgb, cfg)
            return a, s, "retinex_fallback"
    raise ValueError(f"Unknown intrinsic method: {method}")


def intrinsic_rgba(albedo_rgb: np.ndarray, shading_rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rgba_from_rgb_alpha(albedo_rgb, alpha), rgba_from_rgb_alpha(shading_rgb, alpha)
