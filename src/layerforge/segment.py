from __future__ import annotations

import inspect
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .semantic import BACKGROUND_GROUPS, label_to_group
from .types import Segment
from .utils import bbox_from_mask, mask_iou, normalize01, optional_import, touches_border, transformers_pipeline_device_index, write_json


DEFAULT_GROUNDED_SAM2_PROMPTS = [
    "person",
    "animal",
    "vehicle",
    "furniture",
    "plant",
    "building",
    "sky",
    "road",
]


def _gemini_suggest_labels(pil: Image.Image, cfg: dict[str, Any]) -> list[str]:
    from .gemini_io import gemini_suggest_labels

    return gemini_suggest_labels(pil, cfg)


def _gemini_segment_items(pil: Image.Image, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from .gemini_io import gemini_segment_items

    return gemini_segment_items(pil, cfg)


def _decode_segmentation_item(item: dict[str, Any], size: tuple[int, int]) -> tuple[str, np.ndarray]:
    from .gemini_io import decode_segmentation_item

    return decode_segmentation_item(item, size)


def merge_prompt_labels(base: list[str], extra: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for label in [*base, *extra]:
        clean = str(label).strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        merged.append(clean)
    return merged


def _normalize_prompt_list(prompts: list[str] | None) -> list[str]:
    return [str(x).strip().lower() for x in (prompts or []) if str(x).strip()]


def _resolve_grounded_sam2_prompts(pil: Image.Image, cfg: dict[str, Any]) -> list[str]:
    prompts = _normalize_prompt_list(cfg.get("prompts"))
    prompt_source = str(cfg.get("prompt_source", "manual")).lower()
    gemini_modes = {"gemini", "augment", "hybrid", "manual+gemini"}
    if prompt_source in gemini_modes or (prompt_source == "auto" and not prompts):
        try:
            suggested = _gemini_suggest_labels(pil, cfg)
            if prompt_source in {"augment", "hybrid", "manual+gemini"}:
                prompts = merge_prompt_labels(prompts, suggested)
            else:
                prompts = suggested
        except Exception:
            if prompt_source == "gemini" and not prompts:
                raise
    if not prompts:
        prompts = list(DEFAULT_GROUNDED_SAM2_PROMPTS)
    return prompts


@lru_cache(maxsize=8)
def _load_mask2former_pipeline(model_name: str, device_index: int):
    transformers = optional_import("transformers")
    # These pipeline warnings fire for every image and swamp long benchmark runs.
    for logger_name in [
        "transformers.pipelines.base",
        "transformers.models.mask2former.image_processing_mask2former",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    return transformers.pipeline("image-segmentation", model=model_name, device=device_index)


@lru_cache(maxsize=8)
def _load_grounding_dino(model_name: str, device_name: str):
    transformers = optional_import("transformers")
    proc = transformers.AutoProcessor.from_pretrained(model_name)
    model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device_name).eval()
    return proc, model


@lru_cache(maxsize=8)
def _load_sam2(model_name: str, device_name: str):
    transformers = optional_import("transformers")
    proc = transformers.Sam2Processor.from_pretrained(model_name)
    model = transformers.Sam2Model.from_pretrained(model_name).to(device_name).eval()
    return proc, model


def make_segment(seg_id: int, label: str, mask: np.ndarray, score: float, source: str, metadata: dict[str, Any] | None = None) -> Segment:
    m = np.asarray(mask).astype(bool)
    return Segment(seg_id, str(label), label_to_group(str(label)), m, float(score), bbox_from_mask(m), source, metadata or {})


def filter_segments(segments: list[Segment], shape: tuple[int, int], min_area_ratio: float, nms_iou: float) -> list[Segment]:
    h, w = shape
    min_area = max(8, int(h * w * min_area_ratio))
    kept: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.score, s.area), reverse=True):
        if seg.area < min_area:
            continue
        if any(mask_iou(seg.mask, old.mask) >= nms_iou for old in kept):
            continue
        kept.append(seg)
    for idx, seg in enumerate(kept):
        seg.id = idx
    return kept


def add_background_segment(segments: list[Segment], shape: tuple[int, int]) -> list[Segment]:
    union = np.zeros(shape, dtype=bool)
    for seg in segments:
        union |= seg.mask
    bg = ~union
    if bg.mean() > 0.001:
        segments = list(segments) + [make_segment(len(segments), "background visible", bg, 1.0, "background-complement")]
    return segments


def _semantic_family(group: str) -> str:
    if group == "person":
        return "people"
    if group == "animal":
        return "animals"
    if group == "vehicle":
        return "vehicles"
    if group == "furniture":
        return "furniture"
    if group in BACKGROUND_GROUPS:
        return "background/stuff"
    if group in {"effect", "effects", "transparent"}:
        return "effects/transparent"
    return "unknown"


def _mask_containment(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    denom = max(1, min(int(aa.sum()), int(bb.sum())))
    return float((aa & bb).sum() / denom)


def _boundary_quality(mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not m.any():
        return 0.0
    labels, count = ndi.label(m)
    if count <= 0:
        return 0.0
    largest = max(int((labels == idx).sum()) for idx in range(1, count + 1))
    component_score = largest / max(1, int(m.sum()))
    eroded = ndi.binary_erosion(m)
    boundary = m ^ eroded
    compactness = float(m.sum() / max(1, int(boundary.sum()) ** 2))
    return float(np.clip(0.75 * component_score + 0.25 * compactness, 0.0, 1.0))


def _semantic_agreement(a: Segment, b: Segment, prompt_labels: set[str]) -> bool:
    if a.group == b.group:
        return True
    labels = {str(a.label).lower(), str(b.label).lower()}
    if labels & prompt_labels:
        return True
    return _semantic_family(a.group) == _semantic_family(b.group) and _semantic_family(a.group) != "unknown"


def _merge_segment_pair(base: Segment, other: Segment, *, reason: str) -> Segment:
    base_quality = float(base.metadata.get("boundary_quality", _boundary_quality(base.mask)))
    other_quality = float(other.metadata.get("boundary_quality", _boundary_quality(other.mask)))
    winner = other if (other.score, other_quality, other.area) > (base.score, base_quality, base.area) else base
    merged_mask = base.mask | other.mask
    metadata = {
        **winner.metadata,
        "fusion": {
            "reason": reason,
            "members": [base.label, other.label],
            "sources": sorted({base.source, other.source}),
            "semantic_family": _semantic_family(winner.group),
        },
        "boundary_quality": max(base_quality, other_quality),
    }
    return make_segment(base.id, winner.label, merged_mask, max(base.score, other.score), f"fusion({winner.source})", metadata)


def fuse_proposals(
    proposals: list[Segment],
    *,
    shape: tuple[int, int] | None = None,
    prompts: list[str] | None = None,
    iou_threshold: float = 0.82,
    containment_threshold: float = 0.94,
    stuff_overlap_threshold: float = 0.05,
    diagnostics_path: str | Path | None = None,
) -> tuple[list[Segment], dict[str, Any]]:
    if not proposals:
        diagnostics = {"input_count": 0, "output_count": 0, "merges": [], "stuff_object_splits": []}
        if diagnostics_path:
            write_json(diagnostics_path, diagnostics)
        return [], diagnostics

    prompt_labels = {str(label).strip().lower() for label in (prompts or []) if str(label).strip()}
    canvas_shape = shape or proposals[0].mask.shape
    working: list[Segment] = []
    diagnostics: dict[str, Any] = {
        "input_count": len(proposals),
        "output_count": 0,
        "merges": [],
        "stuff_object_splits": [],
        "semantic_families": {},
    }

    for proposal in sorted(proposals, key=lambda seg: (seg.score, _boundary_quality(seg.mask), seg.area), reverse=True):
        candidate = make_segment(
            proposal.id,
            proposal.label,
            proposal.mask,
            proposal.score,
            proposal.source,
            {**proposal.metadata, "boundary_quality": _boundary_quality(proposal.mask), "semantic_family": _semantic_family(proposal.group)},
        )
        merged = False
        for idx, existing in enumerate(working):
            iou = mask_iou(candidate.mask, existing.mask)
            containment = _mask_containment(candidate.mask, existing.mask)
            if _semantic_agreement(candidate, existing, prompt_labels) and (iou >= iou_threshold or containment >= containment_threshold):
                reason = "iou" if iou >= iou_threshold else "containment"
                diagnostics["merges"].append(
                    {
                        "kept": existing.label,
                        "merged": candidate.label,
                        "reason": reason,
                        "iou": iou,
                        "containment": containment,
                    }
                )
                working[idx] = _merge_segment_pair(existing, candidate, reason=reason)
                merged = True
                break
        if not merged:
            working.append(candidate)

    thing_union = np.zeros(canvas_shape, dtype=bool)
    for seg in working:
        if seg.group not in BACKGROUND_GROUPS:
            thing_union |= seg.mask

    resolved: list[Segment] = []
    for seg in working:
        mask = seg.mask
        if seg.group in BACKGROUND_GROUPS and thing_union.any():
            overlap = mask & thing_union
            overlap_ratio = float(overlap.sum() / max(1, int(mask.sum())))
            if overlap_ratio > stuff_overlap_threshold:
                mask = mask & ~thing_union
                diagnostics["stuff_object_splits"].append(
                    {
                        "label": seg.label,
                        "overlap_ratio": overlap_ratio,
                        "remaining_area": int(mask.sum()),
                    }
                )
        if mask.any():
            resolved.append(make_segment(len(resolved), seg.label, mask, seg.score, seg.source, seg.metadata))

    diagnostics["output_count"] = len(resolved)
    diagnostics["semantic_families"] = {
        family: sum(1 for seg in resolved if _semantic_family(seg.group) == family)
        for family in ["people", "animals", "vehicles", "furniture", "background/stuff", "effects/transparent", "unknown"]
    }
    if diagnostics_path:
        write_json(diagnostics_path, diagnostics)
    return resolved, diagnostics


def classical_segments(rgb: np.ndarray, cfg: dict[str, Any]) -> list[Segment]:
    h, w = rgb.shape[:2]
    min_area_ratio = float(cfg.get("min_area_ratio", 0.0018))
    n_segments = max(8, int(cfg.get("slic_segments", 96)))
    cell = max(12, int(round((h * w / n_segments) ** 0.5)))
    gray = normalize01(np.mean(rgb.astype(np.float32), axis=2), robust=True)
    ygrid = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    yfull = np.broadcast_to(ygrid, (h, w))
    segments: list[Segment] = []
    sid = 0
    for y0 in range(0, h, cell):
        for x0 in range(0, w, cell):
            y1, x1 = min(h, y0 + cell), min(w, x0 + cell)
            mask = np.zeros((h, w), dtype=bool)
            mask[y0:y1, x0:x1] = True
            if mask.mean() < min_area_ratio:
                continue
            y = float(yfull[mask].mean())
            lum = float(gray[mask].mean())
            border = touches_border(mask, margin=3)
            if border and y < 0.55 and lum > 0.45:
                label = "sky background stuff"
            elif border and y >= 0.55:
                label = "ground background stuff"
            elif y > 0.63:
                label = "foreground object"
            else:
                label = "object region"
            score = 0.50 + 0.25 * abs(lum - 0.5)
            segments.append(make_segment(sid, label, mask, score, "classical-grid", {"cell": cell}))
            sid += 1
    return filter_segments(segments, (h, w), min_area_ratio, float(cfg.get("nms_iou", 0.86)))


def mask2former_segments(pil: Image.Image, cfg: dict[str, Any], device: str) -> list[Segment]:
    model_name = cfg.get("model", {}).get("mask2former", "facebook/mask2former-swin-large-coco-panoptic")
    dev = transformers_pipeline_device_index(device)
    pipe = _load_mask2former_pipeline(model_name, dev)
    outputs = pipe(pil)
    h, w = pil.height, pil.width
    segs: list[Segment] = []
    for idx, item in enumerate(outputs):
        mask_img = item.get("mask")
        if mask_img is None:
            continue
        mask = np.asarray(mask_img.convert("L") if hasattr(mask_img, "convert") else mask_img) > 127
        if mask.shape != (h, w):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
        segs.append(make_segment(idx, item.get("label", f"region_{idx}"), mask, float(item.get("score", 0.8) or 0.8), "mask2former"))
    return filter_segments(segs, (h, w), float(cfg.get("min_area_ratio", 0.0018)), float(cfg.get("nms_iou", 0.86)))


def _post_process_grounding_dino(
    processor: Any,
    *,
    outputs: Any,
    input_ids: Any,
    box_threshold: float,
    text_threshold: float,
    target_sizes: list[tuple[int, int]],
) -> dict[str, Any]:
    fn = processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {
        "outputs": outputs,
        "input_ids": input_ids,
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "threshold" in params:
        kwargs["threshold"] = box_threshold
    else:
        kwargs["box_threshold"] = box_threshold
    return fn(**kwargs)[0]


def grounded_sam2_segments(pil: Image.Image, cfg: dict[str, Any], device: str) -> list[Segment]:
    torch = optional_import("torch")
    dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    model_cfg = cfg.get("model", {})
    dino_name = model_cfg.get("grounding_dino", "IDEA-Research/grounding-dino-base")
    sam2_name = model_cfg.get("sam2", "facebook/sam2.1-hiera-large")
    prompts = _resolve_grounded_sam2_prompts(pil, cfg)
    text = ". ".join([str(x).strip().rstrip(".") for x in prompts if str(x).strip()]) + "."
    proc, det_model = _load_grounding_dino(dino_name, dev)
    inputs = proc(images=pil, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = det_model(**inputs)
    target_sizes = [(pil.height, pil.width)]
    det = _post_process_grounding_dino(
        proc,
        outputs=outputs,
        input_ids=inputs.input_ids,
        box_threshold=float(model_cfg.get("box_threshold", 0.28)),
        text_threshold=float(model_cfg.get("text_threshold", 0.25)),
        target_sizes=target_sizes,
    )
    boxes = det.get("boxes", [])
    labels = det.get("labels", ["object"] * len(boxes))
    scores = det.get("scores", [0.7] * len(boxes))
    boxes_list = [b.detach().cpu().tolist() if hasattr(b, "detach") else list(b) for b in boxes]
    labels_list = [str(x) for x in labels]
    scores_list = [float(x.detach().cpu()) if hasattr(x, "detach") else float(x) for x in scores]
    if not boxes_list:
        return []
    sam_proc, sam = _load_sam2(sam2_name, dev)
    segs: list[Segment] = []
    for start in range(0, len(boxes_list), 24):
        sub_boxes = boxes_list[start:start + 24]
        sam_inputs = sam_proc(images=pil, input_boxes=[sub_boxes], return_tensors="pt").to(dev)
        with torch.no_grad():
            sam_outputs = sam(**sam_inputs, multimask_output=True)
        masks = sam_proc.post_process_masks(sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"])[0]
        masks_np = masks.detach().cpu().numpy()
        iou = getattr(sam_outputs, "iou_scores", None)
        iou_np = iou.detach().cpu().numpy()[0] if iou is not None else None
        for j in range(len(sub_boxes)):
            if masks_np.ndim == 4:
                best = int(np.argmax(iou_np[j])) if iou_np is not None and iou_np.ndim == 2 else 0
                mask = masks_np[j, best] > 0
            else:
                mask = masks_np[j] > 0
            gi = start + j
            segs.append(make_segment(len(segs), labels_list[gi] if gi < len(labels_list) else "grounded object", mask, scores_list[gi] if gi < len(scores_list) else 0.7, "grounding-dino+sam2", {"box": boxes_list[gi]}))
    return filter_segments(segs, (pil.height, pil.width), float(cfg.get("min_area_ratio", 0.001)), float(cfg.get("nms_iou", 0.80)))


def gemini_segments(pil: Image.Image, cfg: dict[str, Any]) -> list[Segment]:
    items = _gemini_segment_items(pil, cfg)
    if not items:
        raise RuntimeError("Gemini segmentation returned no items")
    segs: list[Segment] = []
    decode_errors: list[str] = []
    for idx, item in enumerate(items):
        try:
            label, mask = _decode_segmentation_item(item, pil.size)
        except Exception as exc:
            decode_errors.append(f"{idx}: {type(exc).__name__}: {exc}")
            continue
        if mask.any():
            segs.append(make_segment(len(segs), label, mask, 1.0, "gemini-segmentation"))
    if not segs:
        detail = "; ".join(decode_errors[:3]) if decode_errors else "all decoded masks were empty"
        raise RuntimeError(f"Gemini segmentation returned no usable masks ({detail})")
    filtered = filter_segments(segs, (pil.height, pil.width), float(cfg.get("min_area_ratio", 0.001)), float(cfg.get("nms_iou", 0.80)))
    if not filtered:
        raise RuntimeError("Gemini segmentation masks were all removed by filtering")
    return filtered


def resolve_disjoint_masks(segments: list[Segment], depth: np.ndarray) -> list[Segment]:
    if not segments:
        return []
    owner = np.full(segments[0].mask.shape, -1, dtype=np.int32)
    ordered = sorted(segments, key=lambda s: (float(np.median(depth[s.mask])) if s.mask.any() else 1.0, s.area))
    out: list[Segment] = []
    for seg in ordered:
        mask = seg.mask & (owner < 0)
        if mask.any():
            owner[mask] = seg.id
            out.append(make_segment(len(out), seg.label, mask, seg.score, seg.source, seg.metadata))
    return out


def segment_image(rgb: np.ndarray, pil: Image.Image, cfg: dict[str, Any], device: str = "auto") -> list[Segment]:
    method = str(cfg.get("method", "classical")).lower()
    fallback_on_error = bool(cfg.get("fallback_on_error", True))
    try:
        if method in {"classical", "slic", "fast"}:
            segs = classical_segments(rgb, cfg)
        elif method in {"mask2former", "panoptic"}:
            segs = mask2former_segments(pil, cfg, device)
        elif method in {"grounded_sam2", "grounded-sam2", "open_vocab", "open-vocab-sam2"}:
            segs = grounded_sam2_segments(pil, cfg, device)
        elif method in {"gemini", "gemini_segmentation", "gemini-segmentation"}:
            segs = gemini_segments(pil, cfg)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    except Exception as exc:
        if method in {"classical", "slic", "fast"} or not fallback_on_error:
            raise
        warnings.warn(
            f"Segmentation backend '{method}' failed ({type(exc).__name__}: {exc}); falling back to classical segmentation.",
            RuntimeWarning,
            stacklevel=2,
        )
        segs = classical_segments(rgb, {**cfg, "method": "classical"})
        for seg in segs:
            seg.metadata["fallback_from"] = method
            seg.metadata["fallback_error"] = f"{type(exc).__name__}: {exc}"
    fusion_cfg = cfg.get("fusion", {}) if isinstance(cfg.get("fusion", {}), dict) else {}
    if bool(fusion_cfg.get("enabled", False)):
        segs, _ = fuse_proposals(
            segs,
            shape=rgb.shape[:2],
            prompts=_normalize_prompt_list(cfg.get("prompts")),
            iou_threshold=float(fusion_cfg.get("iou_threshold", 0.82)),
            containment_threshold=float(fusion_cfg.get("containment_threshold", 0.94)),
            stuff_overlap_threshold=float(fusion_cfg.get("stuff_overlap_threshold", 0.05)),
            diagnostics_path=fusion_cfg.get("diagnostics_path"),
        )
    if bool(cfg.get("add_background_segment", True)):
        segs = add_background_segment(segs, rgb.shape[:2])
    return segs


def summarize_segments(segments: list[Segment]) -> list[dict[str, Any]]:
    return [{"id": s.id, "label": s.label, "group": s.group, "area": s.area, "bbox": s.bbox, "score": s.score, "source": s.source, "metadata": s.metadata} for s in segments]
