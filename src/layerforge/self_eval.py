from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any


DEFAULT_SELF_EVAL_WEIGHTS = {
    "recomposition_fidelity": 0.20,
    "edit_preservation": 0.25,
    "semantic_separation": 0.20,
    "alpha_quality": 0.10,
    "graph_confidence": 0.15,
    "runtime": 0.0,
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


def _semantic_separation_score(row: dict[str, Any]) -> float:
    purity = _as_float(row.get("semantic_purity"))
    if purity is not None:
        return _clip01(purity)
    layer_balance = _layer_balance_score(_as_float(row.get("num_layers")))
    return 0.60 * layer_balance + 0.40 * (1.0 if row.get("has_ordered_layers") else 0.0)


def _alpha_quality_score(row: dict[str, Any]) -> float:
    alpha_quality = _as_float(row.get("alpha_quality_score"))
    if alpha_quality is not None:
        return _clip01(alpha_quality)
    return 0.0


def _graph_confidence_score(row: dict[str, Any]) -> float:
    graph_score = 1.0 if row.get("has_graph") else 0.0
    ordered_score = 1.0 if row.get("has_ordered_layers") else 0.0
    effect_count = _as_float(row.get("effect_layer_count")) or 0.0
    occlusion_edges = _as_float(row.get("occlusion_edge_count")) or 0.0
    edge_score = min(1.0, occlusion_edges / 12.0)
    effect_score = min(1.0, effect_count)
    return _clip01(0.40 * graph_score + 0.20 * ordered_score + 0.25 * edge_score + 0.15 * effect_score)


def _edit_preservation_score(row: dict[str, Any]) -> float:
    explicit = _as_float(row.get("edit_success_score"))
    if explicit is not None:
        return _clip01(explicit)
    layer_balance = _layer_balance_score(_as_float(row.get("num_layers")))
    return 0.50 * layer_balance + 0.50 * (1.0 if row.get("has_graph") else 0.0)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    applied = deepcopy(DEFAULT_SELF_EVAL_WEIGHTS)
    if not weights:
        total = sum(applied.values())
        return {key: value / total for key, value in applied.items()}

    normalized_input = {key: float(value) for key, value in weights.items()}
    if any(key in normalized_input for key in DEFAULT_SELF_EVAL_WEIGHTS):
        applied.update({key: normalized_input[key] for key in normalized_input if key in applied})
    elif {"fidelity", "structure", "editability", "runtime"} & normalized_input.keys():
        fidelity = normalized_input.get("fidelity", 0.0)
        structure = normalized_input.get("structure", 0.0)
        editability = normalized_input.get("editability", 0.0)
        runtime = normalized_input.get("runtime", 0.0)
        applied = {
            "recomposition_fidelity": fidelity,
            "edit_preservation": editability * 0.70,
            "semantic_separation": structure * 0.60,
            "alpha_quality": editability * 0.30,
            "graph_confidence": structure * 0.40,
            "runtime": runtime,
        }
    total = sum(applied.values()) or 1.0
    return {key: value / total for key, value in applied.items()}


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
    applied_weights = _normalize_weights(weights)

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
            components = {
                "recomposition_fidelity": fidelity_score,
                "edit_preservation": _edit_preservation_score(row),
                "semantic_separation": _semantic_separation_score(row),
                "alpha_quality": _alpha_quality_score(row),
                "graph_confidence": _graph_confidence_score(row),
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
