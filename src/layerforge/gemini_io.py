from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from typing import Any

import numpy as np
from PIL import Image


GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def prepare_gemini_image(image: Image.Image, max_side: int) -> Image.Image:
    max_side = max(64, int(max_side))
    prepared = image.copy()
    prepared.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return prepared


def _pil_to_inline_part(image: Image.Image) -> dict[str, Any]:
    fmt = "PNG" if image.mode == "RGBA" else "JPEG"
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        }
    }


def _extract_text_payload(response_json: dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response_json!r}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    if not text.strip():
        raise RuntimeError(f"Gemini returned no text payload: {response_json!r}")
    return text


def parse_jsonish_text(text: str) -> Any:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)


def gemini_generate_content(
    image: Image.Image,
    prompt: str,
    *,
    model: str,
    api_key: str | None = None,
    thinking_budget: int = 0,
    temperature: float = 0.0,
    timeout_sec: int = 300,
    response_mime_type: str = "application/json",
) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini segmentation requires GEMINI_API_KEY in the environment")

    payload = {
        "contents": [
            {
                "parts": [
                    _pil_to_inline_part(image),
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": str(response_mime_type),
            "thinkingConfig": {"thinkingBudget": int(thinking_budget)},
        },
    }
    req = urllib.request.Request(
        f"{GEMINI_API_ROOT}/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=max(30, int(timeout_sec))) as resp:
        body = resp.read().decode("utf-8")
    return _extract_text_payload(json.loads(body))


def gemini_suggest_labels(image: Image.Image, cfg: dict[str, Any]) -> list[str]:
    model_cfg = cfg.get("model", {})
    max_layers = int(model_cfg.get("gemini_max_layers", 12))
    prepared = prepare_gemini_image(image, int(model_cfg.get("gemini_max_side", 1024)))
    manual = [str(x).strip() for x in cfg.get("prompts", []) if str(x).strip()]
    prompt = (
        "You are helping a layered image decomposition system. "
        f"Return a JSON array of at most {max_layers} short noun labels for the most important editable layers in this image. "
        "Prefer semantically meaningful objects and large background surfaces that matter for editing, such as sky, road, ground, water, wall, building, window, person, vehicle, furniture, animal, plant, sign, text. "
        "Avoid tiny fragments, duplicate synonyms, camera artifacts, and over-segmentation. "
    )
    if manual:
        prompt += (
            "Bias toward these user-provided categories when they are visible: "
            + ", ".join(manual)
            + ". "
        )
    prompt += 'Output JSON only, for example ["person", "car", "road", "building", "sky"].'
    text = gemini_generate_content(
        prepared,
        prompt,
        model=str(model_cfg.get("gemini", "gemini-3-flash-preview")),
        timeout_sec=int(model_cfg.get("gemini_timeout_sec", 300)),
    )
    items = parse_jsonish_text(text)
    if not isinstance(items, list):
        raise RuntimeError(f"Gemini labels response was not a list: {items!r}")
    labels: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = str(item).strip().lower()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if not labels:
        raise RuntimeError("Gemini returned an empty label list")
    return labels[:max_layers]


def gemini_segment_items(image: Image.Image, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    model_cfg = cfg.get("model", {})
    max_layers = int(model_cfg.get("gemini_max_layers", 12))
    prepared = prepare_gemini_image(image, int(model_cfg.get("gemini_max_side", 1024)))
    manual = [str(x).strip() for x in cfg.get("prompts", []) if str(x).strip()]
    prompt = (
        "Decompose this image into a concise set of semantically meaningful RGBA editing layers. "
        f"Return a JSON list with at most {max_layers} entries. "
        'Each entry must contain keys "label", "box_2d", and "mask". '
        '"box_2d" must be [y0, x0, y1, x1] with coordinates normalized to 0..1000. '
        '"mask" must be a data:image/png;base64 probability map for the pixels inside the bounding box. '
        "Use descriptive labels. Avoid tiny repeated fragments and merge visually connected regions that should edit together. "
        "Include important background surfaces only when they are significant editable layers."
    )
    if manual:
        prompt += " Focus on these categories when visible: " + ", ".join(manual) + "."
    text = gemini_generate_content(
        prepared,
        prompt,
        model=str(model_cfg.get("gemini", "gemini-3-flash-preview")),
        timeout_sec=int(model_cfg.get("gemini_timeout_sec", 300)),
    )
    items = parse_jsonish_text(text)
    if not isinstance(items, list):
        raise RuntimeError(f"Gemini segmentation response was not a list: {items!r}")
    return items


def decode_segmentation_item(item: dict[str, Any], size: tuple[int, int], threshold: int = 127) -> tuple[str, np.ndarray]:
    width, height = size
    box = item["box_2d"]
    y0 = int(float(box[0]) / 1000.0 * height)
    x0 = int(float(box[1]) / 1000.0 * width)
    y1 = int(float(box[2]) / 1000.0 * height)
    x1 = int(float(box[3]) / 1000.0 * width)
    if y0 >= y1 or x0 >= x1:
        raise ValueError("Invalid Gemini segmentation bounding box")
    png_str = str(item["mask"])
    prefix = "data:image/png;base64,"
    if png_str.startswith(prefix):
        png_str = png_str[len(prefix):]
    mask = Image.open(io.BytesIO(base64.b64decode(png_str)))
    mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    full = np.zeros((height, width), dtype=bool)
    full[y0:y1, x0:x1] = mask_arr >= threshold
    return str(item.get("label", "gemini-object")), full
