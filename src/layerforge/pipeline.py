from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .compose import composite_layers_near_to_far
from .config import deep_update, load_config
from .dalg import export_dalg_manifest
from .depth import estimate_depth
from .graph import build_completed_background_layer, build_layers, graph_json, grouped_layers, renumber_layers_in_place
from .image_io import load_rgb, save_depth16, save_gray, save_rgb, save_rgba
from .inpaint import inpaint_background
from .intrinsics import decompose_intrinsics
from .metrics import compute_run_metrics
from .segment import resolve_disjoint_masks, segment_image, summarize_segments
from .types import PipelineOutputs
from .utils import ensure_dir, seed_everything, write_json
from .visualize import depth_to_rgb, draw_segment_labels, save_layer_contact_sheet, segmentation_overlay


class LayerForgePipeline:
    def __init__(self, config: dict[str, Any] | str | Path | None = None, device: str = "auto") -> None:
        self.cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else load_config(None, config)
        self.device = device
        seed_everything(int(self.cfg.get("project", {}).get("seed", 7)))

    def run(self, input_path: str | Path, output_dir: str | Path, *, segmenter: str | None = None, depth_method: str | None = None, prompts: list[str] | None = None, prompt_source: str | None = None, flip_depth: bool | None = None, save_parallax: bool | None = None, ordering_method: str | None = None, ranker_model_path: str | Path | None = None, config_overrides: dict[str, Any] | None = None) -> PipelineOutputs:
        cfg = load_config(None, self.cfg)
        if config_overrides:
            cfg = deep_update(cfg, config_overrides)
        seed_everything(int(cfg.get("project", {}).get("seed", 7)))
        if segmenter:
            cfg["segmentation"]["method"] = segmenter
        if depth_method:
            cfg["depth"]["method"] = depth_method
        if prompts:
            cfg["segmentation"]["prompts"] = prompts
        if prompt_source:
            cfg["segmentation"]["prompt_source"] = prompt_source
        if flip_depth is not None:
            cfg["depth"]["flip"] = bool(flip_depth)
        if ordering_method:
            cfg["layering"]["ordering_method"] = ordering_method
        if ranker_model_path:
            cfg["layering"]["ranker_model_path"] = str(ranker_model_path)

        out = ensure_dir(output_dir)
        dirs = {k: ensure_dir(out / k) for k in ["layers_ordered_rgba", "layers_alpha", "layers_completed_rgba", "layers_albedo_rgba", "layers_shading_rgba", "layers_amodal_masks", "layers_hidden_masks", "layers_grouped_rgba", "debug"]}
        rgb, pil = load_rgb(input_path, cfg.get("io", {}).get("max_side"))
        save_rgb(dirs["debug"] / "input_rgb.png", rgb)

        raw_segments = segment_image(rgb, pil, cfg["segmentation"], self.device)
        depth_pred = estimate_depth(pil, rgb, cfg["depth"], self.device)
        depth = depth_pred.depth.astype(np.float32)
        segments = resolve_disjoint_masks(raw_segments, depth)

        save_rgb(dirs["debug"] / "segmentation_overlay.png", segmentation_overlay(rgb, segments))
        save_rgb(dirs["debug"] / "segments_labeled.png", draw_segment_labels(rgb, segments))
        save_rgb(dirs["debug"] / "depth_gray.png", depth_to_rgb(depth))
        save_depth16(dirs["debug"] / "depth_16bit.png", depth)
        albedo, shading, iid_method = decompose_intrinsics(rgb, cfg["intrinsics"])
        save_rgb(dirs["debug"] / "intrinsic_albedo.png", albedo)
        save_rgb(dirs["debug"] / "intrinsic_shading.png", shading)

        semantic_layers, nodes = build_layers(
            rgb,
            segments,
            depth,
            albedo,
            shading,
            cfg["layering"],
            cfg["matting"],
            device=self.device,
            amodal_cfg=cfg.get("amodal", {}),
            inpainting_cfg=cfg.get("inpainting", {}),
        )
        premerge_semantic_layer_count = len(nodes)
        remove_mask = np.zeros(depth.shape, dtype=bool)
        for l in semantic_layers:
            if l.group not in {"sky", "road", "ground", "building", "water", "stuff", "background"}:
                remove_mask |= l.alpha > 0.05
        bg_rgb, bg_mask, inpaint_method = inpaint_background(rgb, remove_mask, cfg["inpainting"], self.device)
        bg_albedo, bg_shading, _ = decompose_intrinsics(bg_rgb, cfg["intrinsics"])
        background = build_completed_background_layer(bg_rgb, bg_albedo, bg_shading, len(semantic_layers), inpaint_method)
        layers = renumber_layers_in_place(semantic_layers + [background])
        ordered_layers = list(layers)

        ordered_paths = []
        alpha_paths = []
        completed_paths = []
        hidden_paths: list[str | None] = []
        for l in ordered_layers:
            ordered_paths.append(save_rgba(dirs["layers_ordered_rgba"] / f"{l.name}.png", l.rgba))
            alpha_paths.append(save_gray(dirs["layers_alpha"] / f"{l.name}_alpha.png", l.alpha))
            completed_paths.append(save_rgba(dirs["layers_completed_rgba"] / f"{l.name}_completed.png", l.completed_rgba if l.completed_rgba is not None else l.rgba))
            save_rgba(dirs["layers_albedo_rgba"] / f"{l.name}_albedo.png", l.albedo_rgba)
            save_rgba(dirs["layers_shading_rgba"] / f"{l.name}_shading.png", l.shading_rgba)
            save_gray(dirs["debug"] / f"{l.name}_alpha.png", l.alpha)
            if l.amodal_mask is not None:
                save_gray(dirs["layers_amodal_masks"] / f"{l.name}_amodal.png", l.amodal_mask.astype(np.float32))
            if l.hidden_mask is not None:
                hidden_paths.append(str(save_gray(dirs["layers_hidden_masks"] / f"{l.name}_hidden.png", l.hidden_mask.astype(np.float32))))
            else:
                hidden_paths.append(None)

        grouped = grouped_layers(ordered_layers, bins=3)
        grouped_paths = [save_rgba(dirs["layers_grouped_rgba"] / f"{g.name}.png", g.rgba) for g in grouped]
        recomposed = composite_layers_near_to_far(ordered_layers)
        save_rgba(dirs["debug"] / "recomposed_rgba.png", recomposed)
        save_rgb(dirs["debug"] / "recomposed_rgb.png", recomposed[..., :3])
        save_rgb(dirs["debug"] / "background_completion.png", bg_rgb)
        save_gray(dirs["debug"] / "background_inpaint_mask.png", bg_mask.astype(np.float32))
        if cfg["io"].get("save_contact_sheet", True):
            save_layer_contact_sheet(dirs["debug"] / "ordered_layer_contact_sheet.png", ordered_layers)
            save_layer_contact_sheet(dirs["debug"] / "grouped_layer_contact_sheet.png", grouped)

        parallax_path = None
        if cfg["io"].get("save_parallax_gif", True) if save_parallax is None else save_parallax:
            from .render import save_parallax_gif
            parallax_path = save_parallax_gif(dirs["debug"] / "parallax_preview.gif", ordered_layers, int(cfg["render"].get("parallax_frames", 24)), float(cfg["render"].get("parallax_pixels", 28)))

        metrics = compute_run_metrics(rgb, ordered_layers, cfg)
        metrics.update({
            "segmentation_method": cfg["segmentation"]["method"],
            "depth_method": cfg["depth"]["method"],
            "depth_source": depth_pred.source,
            "intrinsic_method": iid_method,
            "inpaint_method": inpaint_method,
            "ordering_method": cfg["layering"].get("ordering_method", "boundary"),
            "premerge_semantic_layers": float(premerge_semantic_layer_count),
            "merge_reduction": float(max(0, premerge_semantic_layer_count - len(semantic_layers))),
        })
        metrics_path = write_json(out / "metrics.json", metrics)
        graph_path = write_json(dirs["debug"] / "layer_graph.json", graph_json(ordered_layers, nodes))
        segments_path = write_json(dirs["debug"] / "segments.json", summarize_segments(segments))
        manifest = {
            "input": str(input_path), "output_dir": str(out), "config": cfg,
            "ordering_method": cfg["layering"].get("ordering_method", "boundary"),
            "metrics": str(metrics_path), "segments": str(segments_path), "layer_graph": str(graph_path),
            "ordered_layers_near_to_far": [
                {
                    "path": str(path),
                    "alpha_path": str(alpha_path),
                    "completed_path": str(completed_path),
                    "hidden_mask_path": hidden_path,
                    "name": layer.name,
                    "rank": layer.rank,
                    "label": layer.label,
                    "group": layer.group,
                    "depth_median": layer.depth_median,
                    "alpha_quality_score": layer.metadata.get("alpha_quality_score"),
                }
                for path, alpha_path, completed_path, hidden_path, layer in zip(ordered_paths, alpha_paths, completed_paths, hidden_paths, ordered_layers)
            ],
            "grouped_layers": [str(p) for p in grouped_paths],
            "debug": {"input_rgb": str(dirs["debug"] / "input_rgb.png"), "depth_gray": str(dirs["debug"] / "depth_gray.png"), "segmentation_overlay": str(dirs["debug"] / "segmentation_overlay.png"), "background_completion": str(dirs["debug"] / "background_completion.png"), "recomposed_rgb": str(dirs["debug"] / "recomposed_rgb.png"), "parallax_preview": str(parallax_path) if parallax_path else None}
        }
        manifest_path = write_json(out / "manifest.json", manifest)
        canonical_dalg_path = export_dalg_manifest(out)
        manifest["canonical_dalg"] = str(canonical_dalg_path)
        manifest_path = write_json(out / "manifest.json", manifest)
        return PipelineOutputs(out, manifest_path, metrics_path, ordered_paths, grouped_paths, {k: Path(v) for k, v in manifest["debug"].items() if v})

    def enrich_rgba_layers(
        self,
        input_path: str | Path,
        layers_dir: str | Path,
        output_dir: str | Path,
        *,
        depth_method: str | None = None,
        flip_depth: bool | None = None,
        ordering_method: str | None = None,
        ranker_model_path: str | Path | None = None,
        preserve_external_order: bool | None = None,
        merge_external_layers: bool | None = None,
    ) -> PipelineOutputs:
        cfg = load_config(None, self.cfg)
        seed_everything(int(cfg.get("project", {}).get("seed", 7)))
        if ordering_method:
            cfg["layering"]["ordering_method"] = ordering_method
        if ranker_model_path:
            cfg["layering"]["ranker_model_path"] = str(ranker_model_path)
        if preserve_external_order is not None:
            cfg.setdefault("qwen", {})["preserve_external_order"] = bool(preserve_external_order)
        if merge_external_layers is not None:
            cfg.setdefault("qwen", {})["merge_external_layers"] = bool(merge_external_layers)
        from .qwen_io import enrich_rgba_layers as enrich
        return enrich(input_path, layers_dir, output_dir, cfg, self.device, depth_method=depth_method, flip_depth=flip_depth)

    def peel(self, input_path: str | Path, output_dir: str | Path, *, segmenter: str | None = None, depth_method: str | None = None, prompts: list[str] | None = None, prompt_source: str | None = None, flip_depth: bool | None = None, max_layers: int | None = None, config_overrides: dict[str, Any] | None = None) -> PipelineOutputs:
        cfg = load_config(None, self.cfg)
        if config_overrides:
            cfg = deep_update(cfg, config_overrides)
        seed_everything(int(cfg.get("project", {}).get("seed", 7)))
        if segmenter:
            cfg["segmentation"]["method"] = segmenter
        if depth_method:
            cfg["depth"]["method"] = depth_method
        if prompts:
            cfg["segmentation"]["prompts"] = prompts
        if prompt_source:
            cfg["segmentation"]["prompt_source"] = prompt_source
        if flip_depth is not None:
            cfg["depth"]["flip"] = bool(flip_depth)
        if max_layers is not None:
            cfg["peeling"]["max_layers"] = int(max_layers)
        from .peeling import run_recursive_peeling

        return run_recursive_peeling(input_path, output_dir, cfg, self.device)
