from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .dalg import build_dalg_manifest, export_dalg_manifest


DESIGN_MANIFEST_FORMAT = "layerforge.design_manifest"


def _resolve_run_path(run_dir: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return run_dir / path


def _load_or_build_dalg(run_dir: Path) -> dict[str, Any]:
    dalg_path = run_dir / "dalg_manifest.json"
    if dalg_path.exists():
        return json.loads(dalg_path.read_text(encoding="utf-8"))
    return build_dalg_manifest(run_dir)


def build_design_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    dalg = _load_or_build_dalg(root)
    design_export = dalg.get("design_export", {}) if isinstance(dalg.get("design_export"), dict) else {}
    layers = list(design_export.get("layers") or dalg.get("layers") or [])
    semantic_groups = dict(design_export.get("semantic_groups") or {})
    for layer in layers:
        group = str(layer.get("semantic_group") or layer.get("group") or "unknown")
        semantic_groups.setdefault(group, [])
        name = str(layer.get("name") or "")
        if name and name not in semantic_groups[group]:
            semantic_groups[group].append(name)
    return {
        "format": DESIGN_MANIFEST_FORMAT,
        "source": {
            "run_dir": ".",
            "dalg_manifest": "dalg_manifest.json" if (root / "dalg_manifest.json").exists() else None,
            "input": dalg.get("asset", {}).get("input") if isinstance(dalg.get("asset"), dict) else None,
        },
        "canvas": dalg.get("canvas"),
        "layer_order": "near_to_far",
        "alpha_mode": dalg.get("alpha_mode", "straight"),
        "color_space": dalg.get("color_space", "sRGB"),
        "semantic_groups": semantic_groups,
        "layers": sorted(layers, key=lambda row: int(row.get("rank", 0))),
        "includes": design_export.get(
            "includes",
            [
                "ordered_rgba",
                "semantic_groups",
                "alpha_masks",
                "albedo_layers",
                "shading_layers",
                "hidden_masks",
                "completed_layers",
            ],
        ),
        "exports": {
            "dalg": "dalg_manifest.json",
            "psd": "layers.psd",
        },
    }


def export_design_manifest(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(run_dir)
    out_path = Path(output_path) if output_path is not None else root / "design_manifest.json"
    export_dalg_manifest(root, root / "dalg_manifest.json")
    manifest = build_design_manifest(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _image_layer_from_path(
    *,
    image_path: Path,
    parent: Any,
    name: str,
    canvas_size: tuple[int, int],
    visible: bool = True,
) -> Any:
    from psd_tools.api.layers import PixelLayer

    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        if rgba.size != canvas_size:
            rgba = rgba.resize(canvas_size, Image.Resampling.LANCZOS)
        layer = PixelLayer.frompil(rgba, parent, name=name)
    layer.visible = visible
    return layer


def _mask_layer_from_path(*, mask_path: Path, parent: Any, name: str, canvas_size: tuple[int, int]) -> Any:
    with Image.open(mask_path) as image:
        alpha = image.convert("L")
        if alpha.size != canvas_size:
            alpha = alpha.resize(canvas_size, Image.Resampling.LANCZOS)
        rgba = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
        from psd_tools.api.layers import PixelLayer

        layer = PixelLayer.frompil(rgba, parent, name=name)
    layer.visible = False
    return layer


def _support_layer_specs(layer: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        ("alpha mask", "alpha_mask", "mask"),
        ("completed RGBA", "completed_path", "rgba"),
        ("albedo", "albedo_path", "rgba"),
        ("shading", "shading_path", "rgba"),
        ("hidden mask", "hidden_mask", "mask"),
    ]


def export_psd(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    try:
        from psd_tools import PSDImage
        from psd_tools.api.layers import Group
    except ImportError as exc:
        raise RuntimeError(
            "PSD export requires the optional psd-tools package. Install it with "
            "`python -m pip install 'layerforge-x[export]'` or `python -m pip install psd-tools`."
        ) from exc

    root = Path(run_dir)
    design_manifest = build_design_manifest(root)
    canvas = design_manifest.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Cannot export PSD without a known DALG canvas size")
    canvas_size = (width, height)

    out_path = Path(output_path) if output_path is not None else root / "layers.psd"
    psd = PSDImage.new(mode="RGB", size=canvas_size, color=(0, 0, 0))
    group_map: dict[str, Any] = {}
    layers = sorted(design_manifest.get("layers", []), key=lambda row: int(row.get("rank", 0)))
    groups_by_first_rank: dict[str, int] = {}
    for layer in layers:
        group = str(layer.get("semantic_group") or layer.get("group") or "unknown")
        groups_by_first_rank[group] = min(groups_by_first_rank.get(group, 10**9), int(layer.get("rank", 0)))
    for group_name in sorted(groups_by_first_rank, key=lambda value: groups_by_first_rank[value]):
        group = Group.new(psd, name=group_name)
        psd.append(group)
        group_map[group_name] = group

    for layer in layers:
        group_name = str(layer.get("semantic_group") or layer.get("group") or "unknown")
        group = group_map[group_name]
        rank = int(layer.get("rank", 0))
        layer_name = str(layer.get("name") or f"layer_{rank:03d}")
        rgba_path = _resolve_run_path(root, layer.get("path"))
        if rgba_path is not None and rgba_path.exists():
            group.append(
                _image_layer_from_path(
                    image_path=rgba_path,
                    parent=group,
                    name=f"{rank:03d} {layer_name}",
                    canvas_size=canvas_size,
                    visible=True,
                )
            )
        for label, key, kind in _support_layer_specs(layer):
            support_path = _resolve_run_path(root, layer.get(key))
            if support_path is None or not support_path.exists():
                continue
            support_name = f"{rank:03d} {layer_name} / {label}"
            if kind == "mask":
                group.append(_mask_layer_from_path(mask_path=support_path, parent=group, name=support_name, canvas_size=canvas_size))
            else:
                group.append(
                    _image_layer_from_path(
                        image_path=support_path,
                        parent=group,
                        name=support_name,
                        canvas_size=canvas_size,
                        visible=False,
                    )
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(out_path)
    return out_path


def export_design_assets(
    run_dir: str | Path,
    *,
    dalg_output: str | Path | None = None,
    design_output: str | Path | None = None,
    psd_output: str | Path | None = None,
    include_design_json: bool = False,
    include_psd: bool = False,
) -> dict[str, Path]:
    root = Path(run_dir)
    outputs: dict[str, Path] = {
        "dalg": export_dalg_manifest(root, dalg_output),
    }
    if include_design_json:
        outputs["design_json"] = export_design_manifest(root, design_output)
    if include_psd:
        outputs["psd"] = export_psd(root, psd_output)
    return outputs
