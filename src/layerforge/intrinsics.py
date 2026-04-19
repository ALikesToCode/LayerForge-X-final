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


def decompose_intrinsics(rgb: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    method = str(cfg.get("method", "retinex")).lower()
    if method == "retinex":
        a, s = retinex_decompose(rgb, cfg)
        return a, s, "retinex"
    if method in {"external", "marigold", "marigold_iid"}:
        a, s = external_decompose(rgb, str(cfg.get("external_command", "")))
        return a, s, "external"
    raise ValueError(f"Unknown intrinsic method: {method}")


def intrinsic_rgba(albedo_rgb: np.ndarray, shading_rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rgba_from_rgb_alpha(albedo_rgb, alpha), rgba_from_rgb_alpha(shading_rgb, alpha)
