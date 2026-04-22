from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any


DEFAULT_SELF_EVAL_WEIGHTS = {
    "fidelity": 0.40,
    "structure": 0.25,
    "editability": 0.20,
    "runtime": 0.15,
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize(values: list[float], value: float | None, *, invert: bool = False) -> float:
    if value is None:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return 1.0
    score = (value - lo) / (hi - lo)
    return float(1.0 - score if invert else score)


def _layer_balance_score(num_layers: float | None) -> float:
    if num_layers is None:
        return 0.0
    target = 6.0
    spread = 6.0
    return max(0.0, 1.0 - abs(float(num_layers) - target) / spread)


def _structure_score(row: dict[str, Any]) -> float:
    graph_score = 1.0 if row.get("has_graph") else 0.0
    ordered_score = 1.0 if row.get("has_ordered_layers") else 0.0
    effect_count = _as_float(row.get("effect_layer_count")) or 0.0
    effect_score = min(1.0, effect_count)
    return 0.55 * graph_score + 0.25 * ordered_score + 0.20 * effect_score


def _editability_score(row: dict[str, Any]) -> float:
    layer_balance = _layer_balance_score(_as_float(row.get("num_layers")))
    graph_score = 1.0 if row.get("has_graph") else 0.0
    return 0.65 * layer_balance + 0.35 * graph_score


def _reason_for_components(components: dict[str, float]) -> str:
    lead = sorted(components.items(), key=lambda item: item[1], reverse=True)[:2]
    if not lead:
        return "No self-evaluation signals were available."
    labels = ", ".join(f"{name}={value:.3f}" for name, value in lead)
    return f"Selected by heuristic self-evaluation with strongest components: {labels}."


def choose_best_candidates(
    rows: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    applied_weights = deepcopy(DEFAULT_SELF_EVAL_WEIGHTS)
    if weights:
        applied_weights.update({key: float(value) for key, value in weights.items()})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[str(row["image"])].append(row)

    scored_rows: list[dict[str, Any]] = []
    best_by_image: list[dict[str, Any]] = []
    for image, items in grouped.items():
        psnr_values = [_as_float(item.get("recompose_psnr")) for item in items]
        psnr_values = [value for value in psnr_values if value is not None]
        ssim_values = [_as_float(item.get("recompose_ssim")) for item in items]
        ssim_values = [value for value in ssim_values if value is not None]
        runtime_values = [_as_float(item.get("duration_sec")) for item in items]
        runtime_values = [value for value in runtime_values if value is not None]

        for original in items:
            row = deepcopy(original)
            psnr_score = _normalize(psnr_values or [0.0], _as_float(row.get("recompose_psnr")))
            ssim_score = _normalize(ssim_values or [0.0], _as_float(row.get("recompose_ssim")))
            fidelity_score = 0.55 * psnr_score + 0.45 * ssim_score
            runtime_score = _normalize(runtime_values or [0.0], _as_float(row.get("duration_sec")), invert=True)
            structure_score = _structure_score(row)
            editability_score = _editability_score(row)
            components = {
                "fidelity": fidelity_score,
                "structure": structure_score,
                "editability": editability_score,
                "runtime": runtime_score,
            }
            total = sum(applied_weights[key] * components[key] for key in applied_weights)
            row["self_eval_components"] = {key: round(value, 6) for key, value in components.items()}
            row["self_eval_score"] = round(float(total), 6)
            row["self_eval_reason"] = _reason_for_components(components)
            scored_rows.append(row)

        image_rows = [row for row in scored_rows if row["image"] == image]
        if image_rows:
            best = max(
                image_rows,
                key=lambda item: (
                    float(item["self_eval_score"]),
                    _as_float(item.get("recompose_ssim")) or 0.0,
                    _as_float(item.get("recompose_psnr")) or 0.0,
                ),
            )
            best_by_image.append(best)

    return scored_rows, sorted(best_by_image, key=lambda item: item["image"])
