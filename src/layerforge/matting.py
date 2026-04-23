from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from .utils import bbox_from_mask, optional_import


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _resolve_device_name(device: str) -> str:
    torch = optional_import("torch")
    choice = str(device).lower()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _crop_box(mask: np.ndarray, *, expand_px: int) -> tuple[int, int, int, int]:
    h, w = mask.shape
    if not np.any(mask):
        return (0, 0, w, h)
    x1, y1, x2, y2 = bbox_from_mask(mask)
    return (
        max(0, int(x1) - int(expand_px)),
        max(0, int(y1) - int(expand_px)),
        min(w, int(x2) + int(expand_px)),
        min(h, int(y2) + int(expand_px)),
    )


def _preprocess_crop(crop_rgb: np.ndarray, *, max_side: int, torch: Any) -> tuple[Any, tuple[int, int]]:
    crop = Image.fromarray(crop_rgb, mode="RGB")
    size = max(256, int(max_side))
    resized = crop.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN[None, None, :]) / _IMAGENET_STD[None, None, :]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, crop.size[::-1]


def _extract_logits(output: Any) -> Any:
    if isinstance(output, (list, tuple)):
        return output[-1]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        for key in ("logits", "preds", "pred"):
            if key in output:
                return output[key]
    return output


@lru_cache(maxsize=4)
def _load_birefnet_model(model_name: str, device_name: str, prefer_half: bool):
    torch = optional_import("torch")
    transformers = optional_import("transformers")
    model = transformers.AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device_name).eval()
    if prefer_half and device_name.startswith("cuda"):
        model = model.half()
    return model


def _predict_birefnet_alpha(
    image_rgb: np.ndarray,
    support_mask: np.ndarray,
    *,
    model_name: str,
    max_side: int,
    crop_expand_px: int,
    support_expand_px: int,
    respect_support_mask: bool,
    prefer_half: bool,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = optional_import("torch")

    device_name = _resolve_device_name(device)
    model = _load_birefnet_model(model_name, device_name, bool(prefer_half))

    support = support_mask.astype(bool)
    x1, y1, x2, y2 = _crop_box(support, expand_px=int(crop_expand_px))
    crop_rgb = np.asarray(image_rgb[y1:y2, x1:x2], dtype=np.uint8)
    input_tensor, original_size = _preprocess_crop(crop_rgb, max_side=int(max_side), torch=torch)
    input_tensor = input_tensor.to(device_name)
    if prefer_half and device_name.startswith("cuda"):
        input_tensor = input_tensor.to(dtype=torch.float16)

    with torch.no_grad():
        if device_name.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16 if prefer_half else torch.float32):
                output = model(input_tensor)
        else:
            output = model(input_tensor)

    logits = _extract_logits(output)
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    alpha = torch.sigmoid(logits[:, :1]).to(torch.float32)
    alpha = torch.nn.functional.interpolate(alpha, size=original_size, mode="bilinear", align_corners=False)
    alpha_crop = alpha[0, 0].detach().cpu().numpy().astype(np.float32)

    full_alpha = np.zeros(support.shape, dtype=np.float32)
    full_alpha[y1:y2, x1:x2] = alpha_crop

    if respect_support_mask:
        from scipy import ndimage as ndi

        allowed = ndi.binary_dilation(support, iterations=max(0, int(support_expand_px)))
        full_alpha *= allowed.astype(np.float32)

    metadata = {
        "backend": "birefnet",
        "model": model_name,
        "device": device_name,
        "crop_box": [int(x1), int(y1), int(x2), int(y2)],
    }
    return np.clip(full_alpha, 0.0, 1.0), metadata


def predict_alpha_matte(
    image_rgb: np.ndarray,
    support_mask: np.ndarray,
    cfg: dict[str, Any] | None,
    *,
    device: str = "auto",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    config = {
        "backend": "auto",
        "model": "ZhengPeng7/BiRefNet-matting",
        "max_side": 1024,
        "crop_expand_px": 48,
        "support_expand_px": 12,
        "respect_support_mask": True,
        "prefer_half": True,
    }
    if cfg:
        config.update(cfg)

    backend = str(config.get("backend", "auto")).lower()
    if backend in {"heuristic", "none", "off", "disabled"}:
        return None, {"backend": backend, "used": False}
    if backend not in {"auto", "birefnet"}:
        raise ValueError(f"Unsupported matting backend: {backend}")

    try:
        alpha, metadata = _predict_birefnet_alpha(
            image_rgb,
            support_mask,
            model_name=str(config.get("model", "ZhengPeng7/BiRefNet-matting")),
            max_side=int(config.get("max_side", 1024)),
            crop_expand_px=int(config.get("crop_expand_px", 48)),
            support_expand_px=int(config.get("support_expand_px", 12)),
            respect_support_mask=bool(config.get("respect_support_mask", True)),
            prefer_half=bool(config.get("prefer_half", True)),
            device=device,
        )
        metadata["used"] = True
        return alpha, metadata
    except Exception as exc:
        if backend == "birefnet":
            raise
        return None, {
            "backend": "heuristic_fallback",
            "requested_backend": backend,
            "used": False,
            "error": str(exc),
        }
