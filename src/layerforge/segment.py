from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .semantic import label_to_group
from .types import Segment
from .utils import bbox_from_mask, mask_iou, normalize01, optional_import, touches_border


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
    transformers = optional_import("transformers")
    model_name = cfg.get("model", {}).get("mask2former", "facebook/mask2former-swin-large-coco-panoptic")
    dev = 0 if device in {"auto", "cuda"} else -1
    pipe = transformers.pipeline("image-segmentation", model=model_name, device=dev)
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


def grounded_sam2_segments(pil: Image.Image, cfg: dict[str, Any], device: str) -> list[Segment]:
    torch = optional_import("torch")
    transformers = optional_import("transformers")
    dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    model_cfg = cfg.get("model", {})
    dino_name = model_cfg.get("grounding_dino", "IDEA-Research/grounding-dino-base")
    sam2_name = model_cfg.get("sam2", "facebook/sam2.1-hiera-large")
    prompts = cfg.get("prompts") or ["person", "animal", "vehicle", "furniture", "plant", "building", "sky", "road"]
    text = ". ".join([str(x).strip().rstrip(".") for x in prompts if str(x).strip()]) + "."
    proc = transformers.AutoProcessor.from_pretrained(dino_name)
    det_model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(dino_name).to(dev).eval()
    inputs = proc(images=pil, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = det_model(**inputs)
    target_sizes = [(pil.height, pil.width)]
    try:
        det = proc.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=float(model_cfg.get("box_threshold", 0.28)), text_threshold=float(model_cfg.get("text_threshold", 0.25)), target_sizes=target_sizes)[0]
    except TypeError:
        det = proc.post_process_object_detection(outputs, threshold=float(model_cfg.get("box_threshold", 0.28)), target_sizes=target_sizes)[0]
    boxes = det.get("boxes", [])
    labels = det.get("labels", ["object"] * len(boxes))
    scores = det.get("scores", [0.7] * len(boxes))
    boxes_list = [b.detach().cpu().tolist() if hasattr(b, "detach") else list(b) for b in boxes]
    labels_list = [str(x) for x in labels]
    scores_list = [float(x.detach().cpu()) if hasattr(x, "detach") else float(x) for x in scores]
    if not boxes_list:
        return []
    sam_proc = transformers.Sam2Processor.from_pretrained(sam2_name)
    sam = transformers.Sam2Model.from_pretrained(sam2_name).to(dev).eval()
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
    if method in {"classical", "slic", "fast"}:
        segs = classical_segments(rgb, cfg)
    elif method in {"mask2former", "panoptic"}:
        segs = mask2former_segments(pil, cfg, device)
    elif method in {"grounded_sam2", "grounded-sam2", "open_vocab", "open-vocab-sam2"}:
        segs = grounded_sam2_segments(pil, cfg, device)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    if bool(cfg.get("add_background_segment", True)):
        segs = add_background_segment(segs, rgb.shape[:2])
    return segs


def summarize_segments(segments: list[Segment]) -> list[dict[str, Any]]:
    return [{"id": s.id, "label": s.label, "group": s.group, "area": s.area, "bbox": s.bbox, "score": s.score, "source": s.source, "metadata": s.metadata} for s in segments]
