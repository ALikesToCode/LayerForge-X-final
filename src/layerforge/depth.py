from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from .types import DepthPrediction
from .utils import image_to_float, normalize01, optional_import, rank_normalize, transformers_pipeline_device_index


@lru_cache(maxsize=8)
def _load_hf_depth_pipeline(model_name: str, device_index: int):
    transformers = optional_import("transformers")
    logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)
    return transformers.pipeline("depth-estimation", model=model_name, device=device_index)


@lru_cache(maxsize=8)
def _load_depth_pro_components(model_name: str, device_name: str):
    transformers = optional_import("transformers")
    processor_cls = getattr(transformers, "DepthProImageProcessor", None) or transformers.DepthProImageProcessorFast
    proc = processor_cls.from_pretrained(model_name)
    model = transformers.DepthProForDepthEstimation.from_pretrained(model_name).to(device_name).eval()
    return proc, model


@lru_cache(maxsize=4)
def _load_marigold_pipeline(model_name: str, device_name: str, dtype_name: str):
    torch = optional_import("torch")
    diffusers = optional_import("diffusers")
    dtype = getattr(torch, dtype_name)
    return diffusers.MarigoldDepthPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device_name)


def geometric_luminance_depth(rgb: np.ndarray) -> DepthPrediction:
    cv2 = optional_import("cv2")
    img = image_to_float(rgb)
    h, w = img.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    blur = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 7)
    edges = normalize01(np.abs(lum - blur), robust=True)
    far_top_prior = 1.0 - y
    depth = 0.74 * np.broadcast_to(far_top_prior, (h, w)) + 0.18 * lum - 0.08 * edges
    depth = normalize01(depth, robust=True)
    return DepthPrediction(depth, "geometric_luminance", False, raw_depth=depth.copy())


def edge_aware_smooth(depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    cv2 = optional_import("cv2")
    d = depth.astype(np.float32)
    try:
        guide = image_to_float(rgb).astype(np.float32)
        return np.clip(cv2.ximgproc.guidedFilter(guide=guide, src=d, radius=16, eps=1e-4), 0, 1).astype(np.float32)
    except Exception:
        return np.clip(cv2.bilateralFilter(d, d=7, sigmaColor=0.08, sigmaSpace=11), 0, 1).astype(np.float32)


def hf_depth_estimation(pil: Image.Image, cfg: dict[str, Any], model_key: str, source: str, device: str) -> DepthPrediction:
    model_name = cfg.get("model", {}).get(model_key)
    dev = transformers_pipeline_device_index(device)
    pipe = _load_hf_depth_pipeline(model_name, dev)
    out = pipe(pil)
    obj = out.get("predicted_depth")
    if obj is None:
        obj = out.get("depth")
    if obj is None:
        raise RuntimeError(f"Depth pipeline for {source} did not return 'predicted_depth' or 'depth'")
    if hasattr(obj, "detach"):
        depth = obj.detach().cpu().numpy().squeeze()
    else:
        depth = np.asarray(obj)
    depth = np.asarray(Image.fromarray(depth.astype(np.float32)).resize(pil.size, Image.Resampling.BILINEAR), dtype=np.float32)
    return DepthPrediction(normalize01(depth, robust=True), source, metric=(source == "depth_pro"), metadata={"model": model_name}, raw_depth=depth.copy())


def depth_pro(pil: Image.Image, cfg: dict[str, Any], device: str) -> DepthPrediction:
    # Prefer the official Transformers class when available; otherwise use the generic pipeline.
    try:
        torch = optional_import("torch")
        model_name = cfg.get("model", {}).get("depth_pro", "apple/DepthPro-hf")
        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        proc, model = _load_depth_pro_components(model_name, str(dev))
        inputs = proc(images=pil, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        post = proc.post_process_depth_estimation(outputs, target_sizes=[(pil.height, pil.width)])[0]
        depth = post["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        return DepthPrediction(normalize01(depth, robust=True), "depth_pro", True, {"model": model_name}, raw_depth=depth.copy())
    except Exception:
        return hf_depth_estimation(pil, cfg, "depth_pro", "depth_pro", device)


def marigold_depth(pil: Image.Image, cfg: dict[str, Any], device: str) -> DepthPrediction:
    torch = optional_import("torch")
    diffusers = optional_import("diffusers")
    model_name = cfg.get("model", {}).get("marigold", "prs-eth/marigold-depth-v1-1")
    dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    dtype_name = "float16" if dtype == torch.float16 else "float32"
    pipe = _load_marigold_pipeline(model_name, dev, dtype_name)
    out = pipe(pil, output_type="np")
    if hasattr(out, "depth_np"):
        depth = np.asarray(out.depth_np).squeeze()
    elif hasattr(out, "prediction"):
        depth = np.asarray(out.prediction).squeeze()
    elif hasattr(out, "depth"):
        depth = np.asarray(out.depth).squeeze()
    else:
        depth = np.asarray(out[0]).squeeze()
    depth = np.asarray(Image.fromarray(depth.astype(np.float32)).resize(pil.size, Image.Resampling.BILINEAR), dtype=np.float32)
    return DepthPrediction(normalize01(depth, robust=True), "marigold", False, {"model": model_name}, raw_depth=depth.copy())


def ensemble_depth(pil: Image.Image, rgb: np.ndarray, cfg: dict[str, Any], device: str) -> DepthPrediction:
    maps: list[np.ndarray] = []
    sources: list[str] = []
    errors: dict[str, str] = {}
    for method in cfg.get("ensemble", ["depth_pro", "depth_anything_v2", "marigold"]):
        try:
            pred = estimate_depth(pil, rgb, {**cfg, "method": method, "edge_smooth": False}, device)
            maps.append(rank_normalize(pred.depth))
            sources.append(pred.source)
        except Exception as exc:
            errors[str(method)] = str(exc)
    if not maps:
        pred = geometric_luminance_depth(rgb)
        pred.metadata["ensemble_errors"] = errors
        return pred
    fused = np.median(np.stack(maps, axis=0), axis=0).astype(np.float32)
    fused = normalize01(edge_aware_smooth(fused, rgb), robust=True)
    return DepthPrediction(fused, "ensemble(" + "+".join(sources) + ")", False, {"errors": errors}, raw_depth=fused.copy())


def estimate_depth(pil: Image.Image, rgb: np.ndarray, cfg: dict[str, Any], device: str = "auto") -> DepthPrediction:
    method = str(cfg.get("method", "geometric_luminance")).lower()
    if method in {"geometric_luminance", "luminance", "fast"}:
        pred = geometric_luminance_depth(rgb)
    elif method in {"depth_anything_v2", "depth-anything-v2", "depth_anything"}:
        pred = hf_depth_estimation(pil, cfg, "depth_anything_v2", "depth_anything_v2", device)
    elif method in {"depth_pro", "depthpro", "depth-pro"}:
        pred = depth_pro(pil, cfg, device)
    elif method == "marigold":
        pred = marigold_depth(pil, cfg, device)
    elif method == "ensemble":
        pred = ensemble_depth(pil, rgb, cfg, device)
    else:
        raise ValueError(f"Unknown depth method: {method}")
    raw_depth = pred.raw_depth.copy() if pred.raw_depth is not None else pred.depth.copy()
    depth = normalize01(pred.depth, robust=True)
    if not bool(cfg.get("near_is_smaller", True)):
        depth = 1.0 - depth
    if bool(cfg.get("flip", False)):
        depth = 1.0 - depth
    if bool(cfg.get("edge_smooth", True)) and method != "ensemble":
        depth = edge_aware_smooth(depth, rgb)
    pred.depth = normalize01(depth, robust=True)
    pred.raw_depth = raw_depth.astype(np.float32, copy=False)
    return pred
