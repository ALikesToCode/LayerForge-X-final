from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .compose import composite_layers_near_to_far
from .config import load_config
from .depth import estimate_depth
from .graph import build_completed_background_layer, build_layers, graph_json, grouped_layers
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

    def run(self, input_path: str | Path, output_dir: str | Path, *, segmenter: str | None = None, depth_method: str | None = None, prompts: list[str] | None = None, flip_depth: bool | None = None, save_parallax: bool | None = None) -> PipelineOutputs:
        cfg = load_config(None, self.cfg)
        if segmenter:
            cfg["segmentation"]["method"] = segmenter
        if depth_method:
            cfg["depth"]["method"] = depth_method
        if prompts:
            cfg["segmentation"]["prompts"] = prompts
        if flip_depth is not None:
            cfg["depth"]["flip"] = bool(flip_depth)

        out = ensure_dir(output_dir)
        dirs = {k: ensure_dir(out / k) for k in ["layers_ordered_rgba", "layers_albedo_rgba", "layers_shading_rgba", "layers_amodal_masks", "layers_grouped_rgba", "debug"]}
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

        semantic_layers, nodes = build_layers(rgb, segments, depth, albedo, shading, cfg["layering"], cfg["matting"])
        remove_mask = np.zeros(depth.shape, dtype=bool)
        for l in semantic_layers:
            if l.group not in {"sky", "road", "ground", "building", "water", "stuff", "background"}:
                remove_mask |= l.alpha > 0.05
        bg_rgb, bg_mask, inpaint_method = inpaint_background(rgb, remove_mask, cfg["inpainting"], self.device)
        bg_albedo, bg_shading, _ = decompose_intrinsics(bg_rgb, cfg["intrinsics"])
        background = build_completed_background_layer(bg_rgb, bg_albedo, bg_shading, len(semantic_layers), inpaint_method)
        layers = semantic_layers + [background]
        for idx, l in enumerate(sorted(layers, key=lambda x: x.rank)):
            l.id = idx
            l.rank = idx

        ordered_paths = []
        for l in sorted(layers, key=lambda x: x.rank):
            ordered_paths.append(save_rgba(dirs["layers_ordered_rgba"] / f"{l.name}.png", l.rgba))
            save_rgba(dirs["layers_albedo_rgba"] / f"{l.name}_albedo.png", l.albedo_rgba)
            save_rgba(dirs["layers_shading_rgba"] / f"{l.name}_shading.png", l.shading_rgba)
            save_gray(dirs["debug"] / f"{l.name}_alpha.png", l.alpha)
            if l.amodal_mask is not None:
                save_gray(dirs["layers_amodal_masks"] / f"{l.name}_amodal.png", l.amodal_mask.astype(np.float32))

        grouped = grouped_layers(layers, bins=3)
        grouped_paths = [save_rgba(dirs["layers_grouped_rgba"] / f"{g.name}.png", g.rgba) for g in sorted(grouped, key=lambda x: x.rank)]
        recomposed = composite_layers_near_to_far(layers)
        save_rgba(dirs["debug"] / "recomposed_rgba.png", recomposed)
        save_rgb(dirs["debug"] / "recomposed_rgb.png", recomposed[..., :3])
        save_rgb(dirs["debug"] / "background_completion.png", bg_rgb)
        save_gray(dirs["debug"] / "background_inpaint_mask.png", bg_mask.astype(np.float32))
        if cfg["io"].get("save_contact_sheet", True):
            save_layer_contact_sheet(dirs["debug"] / "ordered_layer_contact_sheet.png", layers)
            save_layer_contact_sheet(dirs["debug"] / "grouped_layer_contact_sheet.png", grouped)

        parallax_path = None
        if cfg["io"].get("save_parallax_gif", True) if save_parallax is None else save_parallax:
            from .render import save_parallax_gif
            parallax_path = save_parallax_gif(dirs["debug"] / "parallax_preview.gif", layers, int(cfg["render"].get("parallax_frames", 24)), float(cfg["render"].get("parallax_pixels", 28)))

        metrics = compute_run_metrics(rgb, layers, cfg)
        metrics.update({"segmentation_method": cfg["segmentation"]["method"], "depth_method": cfg["depth"]["method"], "depth_source": depth_pred.source, "intrinsic_method": iid_method, "inpaint_method": inpaint_method})
        metrics_path = write_json(out / "metrics.json", metrics)
        graph_path = write_json(dirs["debug"] / "layer_graph.json", graph_json(layers, nodes))
        segments_path = write_json(dirs["debug"] / "segments.json", summarize_segments(segments))
        manifest = {
            "input": str(input_path), "output_dir": str(out), "config": cfg,
            "metrics": str(metrics_path), "segments": str(segments_path), "layer_graph": str(graph_path),
            "ordered_layers_near_to_far": [{"path": str(p), "name": l.name, "rank": l.rank, "label": l.label, "group": l.group, "depth_median": l.depth_median} for p, l in zip(ordered_paths, sorted(layers, key=lambda x: x.rank))],
            "grouped_layers": [str(p) for p in grouped_paths],
            "debug": {"input_rgb": str(dirs["debug"] / "input_rgb.png"), "depth_gray": str(dirs["debug"] / "depth_gray.png"), "segmentation_overlay": str(dirs["debug"] / "segmentation_overlay.png"), "background_completion": str(dirs["debug"] / "background_completion.png"), "recomposed_rgb": str(dirs["debug"] / "recomposed_rgb.png"), "parallax_preview": str(parallax_path) if parallax_path else None}
        }
        manifest_path = write_json(out / "manifest.json", manifest)
        return PipelineOutputs(out, manifest_path, metrics_path, ordered_paths, grouped_paths, {k: Path(v) for k, v in manifest["debug"].items() if v})

    def enrich_rgba_layers(self, input_path: str | Path, layers_dir: str | Path, output_dir: str | Path, *, depth_method: str | None = None, flip_depth: bool | None = None) -> PipelineOutputs:
        cfg = load_config(None, self.cfg)
        from .qwen_io import enrich_rgba_layers as enrich
        return enrich(input_path, layers_dir, output_dir, cfg, self.device, depth_method=depth_method, flip_depth=flip_depth)

