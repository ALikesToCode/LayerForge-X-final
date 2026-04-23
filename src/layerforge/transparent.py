from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .editability import (
    _alpha_mask,
    _composite_rgba_layers,
    _load_rgb,
    load_ordered_layers,
    select_editable_layer,
)
from .image_io import save_gray, save_rgb, save_rgba
from .inpaint import inpaint_background
from .matting import predict_alpha_matte
from .utils import ensure_dir, write_json


def recover_transparent_foreground(
    input_rgb: np.ndarray,
    clean_background_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    input_f = input_rgb.astype(np.float32) / 255.0
    background_f = clean_background_rgb.astype(np.float32) / 255.0
    alpha_f = np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]
    safe_alpha = np.maximum(alpha_f, float(eps))
    foreground = (input_f - (1.0 - alpha_f) * background_f) / safe_alpha
    foreground = np.clip(foreground, 0.0, 1.0)
    return np.dstack([np.clip(foreground * 255.0, 0, 255).astype(np.uint8), np.clip(alpha_f[..., 0] * 255.0, 0, 255).astype(np.uint8)])


def _estimate_transparent_alpha(
    input_rgb: np.ndarray,
    background_rgb: np.ndarray,
    base_alpha: np.ndarray,
    support_mask: np.ndarray,
    *,
    residual_scale: float = 0.18,
    base_weight: float = 0.35,
    smooth_radius: int = 2,
) -> np.ndarray:
    residual = np.mean(np.abs(input_rgb.astype(np.float32) - background_rgb.astype(np.float32)), axis=2) / 255.0
    residual_alpha = np.clip(residual / max(1e-4, float(residual_scale)), 0.0, 1.0)
    alpha = np.maximum(base_alpha.astype(np.float32) * float(base_weight), residual_alpha)
    alpha *= support_mask.astype(np.float32)
    if smooth_radius > 0:
        im = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
        alpha = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=float(smooth_radius))), dtype=np.uint8).astype(np.float32) / 255.0
        alpha *= support_mask.astype(np.float32)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _refine_transparent_alpha(
    input_rgb: np.ndarray,
    background_rgb: np.ndarray,
    base_alpha: np.ndarray,
    support_mask: np.ndarray,
    *,
    cfg: dict[str, Any],
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    alpha = _estimate_transparent_alpha(
        input_rgb,
        background_rgb,
        base_alpha,
        support_mask,
        residual_scale=float(cfg["residual_alpha_scale"]),
        base_weight=float(cfg["base_alpha_weight"]),
        smooth_radius=int(cfg["alpha_blur_radius"]),
    )
    backend_alpha, backend_meta = predict_alpha_matte(
        input_rgb,
        support_mask,
        {
            "backend": cfg.get("backend", "auto"),
            "model": cfg.get("model", "ZhengPeng7/BiRefNet-matting"),
            "max_side": cfg.get("backend_max_side", 1024),
            "crop_expand_px": cfg.get("backend_crop_expand_px", 48),
            "support_expand_px": cfg.get("backend_support_expand_px", 12),
            "respect_support_mask": cfg.get("backend_respect_support_mask", True),
            "prefer_half": cfg.get("backend_prefer_half", True),
        },
        device=device,
    )
    if backend_alpha is not None:
        blend_weight = float(cfg.get("backend_blend_weight", 0.75))
        alpha = np.clip(np.maximum(alpha, backend_alpha.astype(np.float32) * blend_weight), 0.0, 1.0)
        alpha *= support_mask.astype(np.float32)
    return alpha.astype(np.float32), backend_meta


def export_transparent_assets(
    path_or_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    prompt: str | None = None,
    point: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    target_name: str | None = None,
    cfg: dict[str, Any] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    manifest, layers, input_rgb = load_ordered_layers(path_or_dir)
    run_dir = Path(path_or_dir)
    if run_dir.is_file():
        run_dir = run_dir.parent
    output_root = ensure_dir(Path(output_dir) if output_dir is not None else run_dir / "transparent_extract")
    config = {
        "support_expand_px": 12,
        "residual_alpha_scale": 0.18,
        "base_alpha_weight": 0.35,
        "alpha_blur_radius": 2,
        "eps": 1e-3,
        "inpaint_radius": 5.0,
        "backend": "auto",
        "model": "ZhengPeng7/BiRefNet-matting",
        "backend_max_side": 1024,
        "backend_crop_expand_px": 48,
        "backend_support_expand_px": 12,
        "backend_respect_support_mask": True,
        "backend_prefer_half": True,
        "backend_blend_weight": 0.75,
    }
    run_config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    if isinstance(run_config.get("transparent"), dict):
        config.update(run_config["transparent"])
    if cfg:
        config.update(cfg)

    selection_cfg = run_config.get("target_selection", {}) if isinstance(run_config.get("target_selection"), dict) else None
    target = select_editable_layer(
        layers,
        prompt=prompt,
        point=point,
        box=box,
        target_name=target_name,
        input_rgb=input_rgb,
        cfg=selection_cfg,
    )
    if target is None:
        raise ValueError("No editable target layer could be selected for transparent decomposition")

    target_mask = target["mask"]
    if int(target_mask.sum()) == 0:
        raise ValueError("Selected transparent target has an empty mask")

    support_expand_px = max(0, int(config["support_expand_px"]))
    support_mask = target_mask.copy()
    if support_expand_px > 0:
        from scipy import ndimage as ndi

        support_mask = ndi.binary_dilation(target_mask, iterations=support_expand_px)

    remaining = [layer for layer in layers if layer["name"] != target["name"]]
    composited_background = _composite_rgba_layers(remaining)[..., :3]
    inpainted_background, _, inpaint_method = inpaint_background(
        input_rgb,
        support_mask,
        {"method": "opencv_telea", "radius": float(config["inpaint_radius"])},
    )
    background_rgb = inpainted_background
    base_alpha = target["alpha"]
    alpha, backend_meta = _refine_transparent_alpha(
        input_rgb,
        background_rgb,
        base_alpha,
        support_mask,
        cfg=config,
        device=device,
    )
    foreground_rgba = recover_transparent_foreground(input_rgb, background_rgb, alpha, eps=float(config["eps"]))
    recomposition = _composite_rgba_layers(
        [
            {"rgba": foreground_rgba, "rank": int(target["rank"])},
            {"rgba": np.dstack([background_rgb, np.full(background_rgb.shape[:2], 255, dtype=np.uint8)]), "rank": int(target["rank"]) + 1},
        ]
    )[..., :3]

    save_rgba(output_root / "transparent_foreground_rgba.png", foreground_rgba)
    save_rgb(output_root / "estimated_clean_background.png", background_rgb)
    save_rgb(output_root / "background_from_remaining_layers.png", composited_background)
    save_gray(output_root / "alpha_map.png", alpha)
    save_rgb(output_root / "recomposition.png", recomposition)

    metrics = {
        "input": manifest.get("input"),
        "run_dir": str(run_dir),
        "selected_target": {
            "name": target["name"],
            "label": target["label"],
            "group": target["group"],
            "rank": int(target["rank"]),
            "bbox": [int(v) for v in target["bbox"]],
        },
        "background_method": "opencv_telea_inpaint",
        "inpaint_method": inpaint_method,
        "alpha_backend": backend_meta,
        "alpha_nonzero_ratio": float(np.mean(alpha > 0.05)),
        "alpha_mean": float(alpha[target_mask].mean()) if np.any(target_mask) else 0.0,
        "transparent_residual_mean": float(np.mean(np.abs(input_rgb.astype(np.float32) - background_rgb.astype(np.float32))) / 255.0),
        "recompose_psnr": float(peak_signal_noise_ratio(input_rgb, recomposition, data_range=255)),
        "recompose_ssim": float(structural_similarity(input_rgb, recomposition, channel_axis=2, data_range=255)),
        "prompt": prompt,
        "point": list(point) if point is not None else None,
        "box": list(box) if box is not None else None,
        "exports": {
            "transparent_foreground_rgba": str((output_root / "transparent_foreground_rgba.png").relative_to(output_root)),
            "estimated_clean_background": str((output_root / "estimated_clean_background.png").relative_to(output_root)),
            "background_from_remaining_layers": str((output_root / "background_from_remaining_layers.png").relative_to(output_root)),
            "alpha_map": str((output_root / "alpha_map.png").relative_to(output_root)),
            "recomposition": str((output_root / "recomposition.png").relative_to(output_root)),
        },
    }
    write_json(output_root / "transparent_metrics.json", metrics)
    return metrics
