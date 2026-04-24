from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .dalg import export_dalg_manifest
from .editability import export_target_assets
from .editability import target_geometry_is_confident
from .frontier import materialize_frontier_selection
from .frontier import resolve_frontier_native_config
from .frontier import run_single_image_frontier_selection
from .pipeline import LayerForgePipeline
from .transparent import export_transparent_assets
from .webui import serve_webui


def parse_prompts(text: str | None) -> list[str] | None:
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def parse_point(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    x, y = [int(part.strip()) for part in text.split(",", maxsplit=1)]
    return (x, y)


def parse_box(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--box expects x1,y1,x2,y2")
    return (parts[0], parts[1], parts[2], parts[3])


def default_frontier_output_root(output: str | Path) -> Path:
    output_path = Path(output)
    name = output_path.name or "output"
    return output_path.parent / f"{name}_frontier"


def build_frontier_base_kwargs(args: argparse.Namespace, *, output_root: Path) -> dict[str, Any]:
    return {
        "input_path": args.input,
        "output_root": output_root,
        "native_config": resolve_frontier_native_config(args.config),
        "native_segmenter": args.segmenter or "grounded_sam2",
        "native_depth": args.depth or "ensemble",
        "skip_native": bool(getattr(args, "frontier_skip_native", False)),
        "peeling_config": getattr(args, "frontier_peeling_config", "configs/recursive_peeling.yaml"),
        "peeling_segmenter": getattr(args, "frontier_peeling_segmenter", None) or args.segmenter or "grounded_sam2",
        "peeling_depth": getattr(args, "frontier_peeling_depth", None) or args.depth or "ensemble",
        "skip_peeling": bool(getattr(args, "frontier_skip_peeling", False)),
        "qwen_model": getattr(args, "frontier_qwen_model", "Qwen/Qwen-Image-Layered"),
        "qwen_layers": getattr(args, "frontier_qwen_layers", "3,4,6,8"),
        "qwen_resolution": int(getattr(args, "frontier_qwen_resolution", 640)),
        "qwen_steps": int(getattr(args, "frontier_qwen_steps", 10)),
        "qwen_device": getattr(args, "frontier_qwen_device", "cuda"),
        "qwen_dtype": getattr(args, "frontier_qwen_dtype", "bfloat16"),
        "qwen_offload": getattr(args, "frontier_qwen_offload", "sequential"),
        "qwen_hybrid_modes": getattr(args, "frontier_qwen_hybrid_modes", "preserve,reorder"),
        "qwen_merge_external_layers": bool(getattr(args, "frontier_qwen_merge_external_layers", False)),
        "skip_existing": False,
    }


def _should_rerun_with_geometry_prompt(
    metadata: dict[str, Any],
    *,
    prompt_values: list[str] | None,
    point: tuple[int, int] | None,
    box: tuple[int, int, int, int] | None,
    target_name: str | None,
) -> bool:
    return (
        not prompt_values
        and not target_name
        and bool(point or box)
        and bool(metadata.get("resolved_prompt"))
        and not target_geometry_is_confident(metadata)
    )


def add_frontier_base_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frontier", action="store_true", help="Select the strongest measured base decomposition from the native, peeling, and Qwen candidate bank before running this targeted export")
    parser.add_argument("--frontier-output-root", default=None, help="Optional frontier working directory; defaults to <output>/frontier")
    parser.add_argument("--frontier-skip-native", action="store_true", help="Exclude native LayerForge from the frontier base selection")
    parser.add_argument("--frontier-skip-peeling", action="store_true", help="Exclude recursive peeling from the frontier base selection")
    parser.add_argument("--frontier-peeling-config", default="configs/recursive_peeling.yaml")
    parser.add_argument("--frontier-peeling-segmenter", default=None)
    parser.add_argument("--frontier-peeling-depth", default=None)
    parser.add_argument("--frontier-qwen-model", default="Qwen/Qwen-Image-Layered")
    parser.add_argument("--frontier-qwen-layers", default="3,4,6,8")
    parser.add_argument("--frontier-qwen-resolution", type=int, default=640)
    parser.add_argument("--frontier-qwen-steps", type=int, default=10)
    parser.add_argument("--frontier-qwen-device", default="cuda")
    parser.add_argument("--frontier-qwen-dtype", default="bfloat16")
    parser.add_argument("--frontier-qwen-offload", default="sequential", choices=["none", "model", "sequential"])
    parser.add_argument("--frontier-qwen-hybrid-modes", default="preserve,reorder")
    parser.add_argument("--frontier-qwen-merge-external-layers", action="store_true")


def cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "frontier", False):
        frontier_root = Path(args.frontier_output_root) if args.frontier_output_root else default_frontier_output_root(args.output)
        selection = run_single_image_frontier_selection(**build_frontier_base_kwargs(args, output_root=frontier_root))
        materialized = materialize_frontier_selection(selection, Path(args.output), frontier_root=frontier_root)
        print(f"winner:   {selection['selected_label']}")
        print(f"frontier: {selection['summary_path']}")
        print(f"manifest: {materialized['manifest_path']}")
        print(f"metrics:  {materialized['metrics_path']}")
        print(f"layers:   {(Path(args.output) / 'layers_ordered_rgba')}")
        return 0
    cfg = load_config(args.config)
    pipe = LayerForgePipeline(cfg, device=args.device)
    out = pipe.run(
        args.input,
        args.output,
        segmenter=args.segmenter,
        depth_method=args.depth,
        prompts=parse_prompts(args.prompts),
        prompt_source=args.prompt_source,
        flip_depth=args.flip_depth,
        save_parallax=not args.no_parallax,
        ordering_method=args.ordering,
        ranker_model_path=args.ranker_model,
    )
    print(f"manifest: {out.manifest_path}")
    print(f"metrics:  {out.metrics_path}")
    print(f"layers:   {len(out.ordered_layer_paths)} RGBA files")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pipe = LayerForgePipeline(cfg, device=args.device)
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(sorted(Path(args.input_dir).glob(ext)))
    for p in paths:
        pipe.run(
            p,
            Path(args.output_dir) / p.stem,
            segmenter=args.segmenter,
            depth_method=args.depth,
            prompts=parse_prompts(args.prompts),
            prompt_source=args.prompt_source,
            flip_depth=args.flip_depth,
            save_parallax=not args.no_parallax,
            ordering_method=args.ordering,
            ranker_model_path=args.ranker_model,
        )
        print(f"{p.name} -> {Path(args.output_dir) / p.stem}")
    return 0


def cmd_autotune(args: argparse.Namespace) -> int:
    from .autotune import run_autotune

    cfg = load_config(args.config)
    pipe = LayerForgePipeline(cfg, device=args.device)
    summary = run_autotune(
        pipe,
        input_path=args.input,
        output_dir=args.output,
        segmenter=args.segmenter or cfg["segmentation"]["method"],
        depth_method=args.depth,
        prompts=parse_prompts(args.prompts),
        flip_depth=args.flip_depth,
        save_parallax=not args.no_parallax,
        ordering_method=args.ordering,
        ranker_model_path=args.ranker_model,
    )
    print(f"summary: {Path(args.output) / 'search_summary.json'}")
    print(f"best:    {summary['best']['copied_best_dir']}")
    print(f"winner:  {summary['best']['name']}")
    metrics = summary["best"]["metrics"]
    print(f"score:   PSNR={float(metrics.get('recompose_psnr', 0.0)):.4f} SSIM={float(metrics.get('recompose_ssim', 0.0)):.4f} layers={float(metrics.get('num_layers', 0.0)):.1f}")
    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    from .frontier import run_frontier_comparison

    return run_frontier_comparison(args)


def cmd_enrich_qwen(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pipe = LayerForgePipeline(cfg, device=args.device)
    out = pipe.enrich_rgba_layers(
        args.input,
        args.layers_dir,
        args.output,
        depth_method=args.depth,
        flip_depth=args.flip_depth,
        ordering_method=args.ordering,
        ranker_model_path=args.ranker_model,
        preserve_external_order=args.preserve_external_order,
        merge_external_layers=args.merge_external_layers,
    )
    print(f"manifest: {out.manifest_path}")
    print(f"metrics:  {out.metrics_path}")
    print(f"ordered enriched layers: {len(out.ordered_layer_paths)}")
    return 0


def cmd_peel(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pipe = LayerForgePipeline(cfg, device=args.device)
    out = pipe.peel(
        args.input,
        args.output,
        segmenter=args.segmenter,
        depth_method=args.depth,
        prompts=parse_prompts(args.prompts),
        prompt_source=args.prompt_source,
        flip_depth=args.flip_depth,
        max_layers=args.max_layers,
    )
    print(f"manifest: {out.manifest_path}")
    print(f"metrics:  {out.metrics_path}")
    print(f"layers:   {len(out.ordered_layer_paths)} RGBA files")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    prompt_values = parse_prompts(args.prompt)
    point = parse_point(args.point)
    box = parse_box(args.box)
    pipe: LayerForgePipeline | None = None
    if args.frontier:
        frontier_root = Path(args.frontier_output_root) if args.frontier_output_root else Path(args.output) / "frontier"
        selection = run_single_image_frontier_selection(**build_frontier_base_kwargs(args, output_root=frontier_root))
        manifest_path = selection["manifest_path"]
        metrics_path = selection["metrics_path"]
        base_run_dir = selection["run_dir"]
    else:
        cfg = load_config(args.config)
        pipe = LayerForgePipeline(cfg, device=args.device)
        out = pipe.run(
            args.input,
            args.output,
            segmenter=args.segmenter,
            depth_method=args.depth,
            prompts=prompt_values,
            prompt_source=args.prompt_source,
            flip_depth=args.flip_depth,
            save_parallax=not args.no_parallax,
            ordering_method=args.ordering,
            ranker_model_path=args.ranker_model,
        )
        manifest_path = out.manifest_path
        metrics_path = out.metrics_path
        base_run_dir = Path(out.output_dir)
    metadata = export_target_assets(
        base_run_dir,
        output_dir=Path(args.output) / "target_extract",
        prompt=args.prompt,
        point=point,
        box=box,
        target_name=args.target_name,
    )
    if pipe is not None and _should_rerun_with_geometry_prompt(
        metadata,
        prompt_values=prompt_values,
        point=point,
        box=box,
        target_name=args.target_name,
    ):
        fallback_prompt = str(metadata["resolved_prompt"])
        out = pipe.run(
            args.input,
            Path(args.output) / "geometry_prompted_base",
            segmenter=args.segmenter,
            depth_method=args.depth,
            prompts=[fallback_prompt],
            prompt_source="manual",
            flip_depth=args.flip_depth,
            save_parallax=not args.no_parallax,
            ordering_method=args.ordering,
            ranker_model_path=args.ranker_model,
        )
        manifest_path = out.manifest_path
        metrics_path = out.metrics_path
        base_run_dir = Path(out.output_dir)
        metadata = export_target_assets(
            base_run_dir,
            output_dir=Path(args.output) / "target_extract",
            prompt=fallback_prompt,
            point=point,
            box=box,
            target_name=args.target_name,
        )
    if args.frontier:
        print(f"winner:   {selection['selected_label']}")
        print(f"frontier: {selection['summary_path']}")
    print(f"manifest: {manifest_path}")
    print(f"metrics:  {metrics_path}")
    print(f"target:   {(Path(args.output) / 'target_extract' / 'target_metadata.json')}")
    print(f"selected: {metadata['selected_target']['name']}")
    return 0


def cmd_export_design(args: argparse.Namespace) -> int:
    output_path = export_dalg_manifest(args.run_dir, args.output)
    print(f"dalg:     {output_path}")
    return 0


def cmd_transparent(args: argparse.Namespace) -> int:
    prompt_values = parse_prompts(args.prompt)
    point = parse_point(args.point)
    box = parse_box(args.box)
    pipe: LayerForgePipeline | None = None
    if args.frontier:
        frontier_root = Path(args.frontier_output_root) if args.frontier_output_root else Path(args.output) / "frontier"
        selection = run_single_image_frontier_selection(**build_frontier_base_kwargs(args, output_root=frontier_root))
        manifest_path = selection["manifest_path"]
        base_run_dir = selection["run_dir"]
    else:
        cfg = load_config(args.config)
        pipe = LayerForgePipeline(cfg, device=args.device)
        run_dir = Path(args.output) / "base_run"
        out = pipe.run(
            args.input,
            run_dir,
            segmenter=args.segmenter,
            depth_method=args.depth,
            prompts=prompt_values,
            prompt_source=args.prompt_source,
            flip_depth=args.flip_depth,
            save_parallax=False,
            ordering_method=args.ordering,
            ranker_model_path=args.ranker_model,
        )
        manifest_path = out.manifest_path
        base_run_dir = Path(out.output_dir)
    metadata = export_transparent_assets(
        base_run_dir,
        output_dir=Path(args.output) / "transparent_extract",
        prompt=args.prompt,
        point=point,
        box=box,
        target_name=args.target_name,
        device=args.device,
    )
    if pipe is not None and _should_rerun_with_geometry_prompt(
        metadata,
        prompt_values=prompt_values,
        point=point,
        box=box,
        target_name=args.target_name,
    ):
        fallback_prompt = str(metadata["resolved_prompt"])
        out = pipe.run(
            args.input,
            Path(args.output) / "geometry_prompted_base",
            segmenter=args.segmenter,
            depth_method=args.depth,
            prompts=[fallback_prompt],
            prompt_source="manual",
            flip_depth=args.flip_depth,
            save_parallax=False,
            ordering_method=args.ordering,
            ranker_model_path=args.ranker_model,
        )
        manifest_path = out.manifest_path
        base_run_dir = Path(out.output_dir)
        metadata = export_transparent_assets(
            base_run_dir,
            output_dir=Path(args.output) / "transparent_extract",
            prompt=fallback_prompt,
            point=point,
            box=box,
            target_name=args.target_name,
            device=args.device,
        )
    if args.frontier:
        print(f"winner:   {selection['selected_label']}")
        print(f"frontier: {selection['summary_path']}")
    print(f"manifest: {manifest_path}")
    print(f"metrics:  {(Path(args.output) / 'transparent_extract' / 'transparent_metrics.json')}")
    print(f"selected: {metadata['selected_target']['name']}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_synthetic_benchmark

    report_path = run_synthetic_benchmark(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        config_path=args.config,
        segmenter=args.segmenter,
        depth=args.depth,
        device=args.device,
        max_scenes=args.max_scenes,
        ordering_method=args.ordering,
        ranker_model_path=args.ranker_model,
    )
    print(f"benchmark report: {report_path}")
    return 0


def cmd_benchmark_coco_panoptic(args: argparse.Namespace) -> int:
    from .coco_benchmark import run_coco_panoptic_group_benchmark

    report_path = run_coco_panoptic_group_benchmark(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        config_path=args.config,
        segmenter=args.segmenter,
        prompts=parse_prompts(args.prompts),
        prompt_source=args.prompt_source,
        device=args.device,
        max_images=args.max_images,
        seed=args.seed,
    )
    print(f"benchmark report: {report_path}")
    return 0


def cmd_benchmark_ade20k(args: argparse.Namespace) -> int:
    from .ade20k_benchmark import run_ade20k_group_benchmark

    report_path = run_ade20k_group_benchmark(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        config_path=args.config,
        segmenter=args.segmenter,
        prompts=parse_prompts(args.prompts),
        prompt_source=args.prompt_source,
        device=args.device,
        max_images=args.max_images,
        seed=args.seed,
    )
    print(f"benchmark report: {report_path}")
    return 0


def cmd_benchmark_diode(args: argparse.Namespace) -> int:
    from .diode_benchmark import run_diode_depth_benchmark

    report_path = run_diode_depth_benchmark(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        config_path=args.config,
        depth_method=args.depth,
        device=args.device,
        max_images=args.max_images,
        seed=args.seed,
        alignment=args.alignment,
    )
    print(f"benchmark report: {report_path}")
    return 0


def cmd_train_ranker(args: argparse.Namespace) -> int:
    from .ranker import train_synthetic_order_ranker

    report = train_synthetic_order_ranker(
        dataset_dir=Path(args.dataset_dir),
        output_path=Path(args.output),
        config_path=args.config,
        segmenter=args.segmenter,
        depth=args.depth,
        device=args.device,
        max_scenes=args.max_scenes,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_webui(args: argparse.Namespace) -> int:
    return serve_webui(host=args.host, port=args.port, open_browser=args.open_browser)


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import build_doctor_report, doctor_exit_code, doctor_json, render_doctor_text

    report = build_doctor_report(
        config_path=args.config,
        device=args.device,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    if args.json:
        print(doctor_json(report))
    else:
        print(render_doctor_text(report))
    return doctor_exit_code(report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="layerforge", description="LayerForge-X: depth-aware amodal layer graph generator")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run LayerForge-X on one RGB image")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config", default="configs/fast.yaml")
    run.add_argument("--segmenter", default=None, help="classical | mask2former | grounded_sam2 | gemini")
    run.add_argument("--depth", default=None, help="geometric_luminance | depth_pro | depth_anything_v2 | marigold | ensemble")
    run.add_argument("--prompts", default=None, help="Comma-separated open-vocabulary prompts")
    run.add_argument("--prompt-source", default=None, help="manual | gemini | augment | hybrid | auto (for promptable segmenters)")
    run.add_argument("--device", default="auto")
    run.add_argument("--flip-depth", action="store_true")
    run.add_argument("--ordering", default=None, help="boundary | learned")
    run.add_argument("--ranker-model", default=None, help="Path to a trained order-ranker JSON file")
    run.add_argument("--no-parallax", action="store_true")
    add_frontier_base_arguments(run)
    run.set_defaults(func=cmd_run)

    batch = sub.add_parser("batch", help="Run LayerForge-X on all images in a folder")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--config", default="configs/fast.yaml")
    batch.add_argument("--segmenter", default=None)
    batch.add_argument("--depth", default=None)
    batch.add_argument("--prompts", default=None)
    batch.add_argument("--prompt-source", default=None, help="manual | gemini | augment | hybrid | auto")
    batch.add_argument("--device", default="auto")
    batch.add_argument("--flip-depth", action="store_true")
    batch.add_argument("--ordering", default=None, help="boundary | learned")
    batch.add_argument("--ranker-model", default=None, help="Path to a trained order-ranker JSON file")
    batch.add_argument("--no-parallax", action="store_true")
    batch.set_defaults(func=cmd_batch)

    autotune = sub.add_parser("autotune", help="Run multiple strong candidate settings for one image and keep the best measured output")
    autotune.add_argument("--input", required=True)
    autotune.add_argument("--output", required=True)
    autotune.add_argument("--config", default="configs/best_score.yaml")
    autotune.add_argument("--segmenter", default=None, help="Default: use config segmenter; grounded_sam2 gets the full search ladder")
    autotune.add_argument("--depth", default=None, help="Optional override depth method")
    autotune.add_argument("--prompts", default=None, help="Comma-separated seed prompts; if omitted, search falls back to Gemini-only prompting")
    autotune.add_argument("--device", default="auto")
    autotune.add_argument("--flip-depth", action="store_true")
    autotune.add_argument("--ordering", default=None, help="boundary | learned")
    autotune.add_argument("--ranker-model", default=None, help="Path to a trained order-ranker JSON file")
    autotune.add_argument("--no-parallax", action="store_true")
    autotune.set_defaults(func=cmd_autotune)

    frontier = sub.add_parser("frontier", help="Run the full measured native/Qwen/peeling candidate bank for one or more images and keep the best scored decomposition per image")
    from .frontier import add_frontier_arguments

    add_frontier_arguments(frontier)
    frontier.set_defaults(func=cmd_frontier)

    enrich = sub.add_parser("enrich-qwen", help="Import RGBA layers from Qwen-Image-Layered or any external decomposer and add LayerForge graph/depth/intrinsic metadata")
    enrich.add_argument("--input", required=True, help="Original RGB image")
    enrich.add_argument("--layers-dir", required=True, help="Folder containing external RGBA layer PNGs")
    enrich.add_argument("--output", required=True)
    enrich.add_argument("--config", default="configs/fast.yaml")
    enrich.add_argument("--depth", default=None)
    enrich.add_argument("--device", default="auto")
    enrich.add_argument("--flip-depth", action="store_true")
    enrich.add_argument("--ordering", default=None, help="boundary | learned")
    enrich.add_argument("--ranker-model", default=None, help="Path to a trained order-ranker JSON file")
    enrich.add_argument("--preserve-external-order", action="store_true", help="Keep the best manifest-derived visual order and add LayerForge metadata without graph reordering")
    enrich.add_argument("--merge-external-layers", action="store_true", help="Allow LayerForge to merge compatible external layers; disabled by default for fair Qwen comparisons")
    enrich.set_defaults(func=cmd_enrich_qwen)

    peel = sub.add_parser("peel", help="Run graph-guided recursive layer peeling with residual inpainting and optional effect layers")
    peel.add_argument("--input", required=True)
    peel.add_argument("--output", required=True)
    peel.add_argument("--config", default="configs/fast.yaml")
    peel.add_argument("--segmenter", default=None, help="classical | mask2former | grounded_sam2 | gemini")
    peel.add_argument("--depth", default=None, help="geometric_luminance | depth_pro | depth_anything_v2 | marigold | ensemble")
    peel.add_argument("--prompts", default=None, help="Comma-separated open-vocabulary prompts")
    peel.add_argument("--prompt-source", default=None, help="manual | gemini | augment | auto")
    peel.add_argument("--device", default="auto")
    peel.add_argument("--flip-depth", action="store_true")
    peel.add_argument("--max-layers", type=int, default=None)
    peel.set_defaults(func=cmd_peel)

    extract = sub.add_parser("extract", help="Run LayerForge-X and export one prompt-selected editable target layer plus background-completed previews")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--config", default="configs/fast.yaml")
    extract.add_argument("--segmenter", default=None, help="classical | mask2former | grounded_sam2 | gemini")
    extract.add_argument("--depth", default=None, help="geometric_luminance | depth_pro | depth_anything_v2 | marigold | ensemble")
    extract.add_argument("--prompt", default=None, help="Prompt describing the target layer, e.g. 'the red car'")
    extract.add_argument("--point", default=None, help="Optional x,y point hint in image pixels")
    extract.add_argument("--box", default=None, help="Optional x1,y1,x2,y2 box hint in image pixels")
    extract.add_argument("--target-name", default=None, help="Optional exact layer name override after decomposition")
    extract.add_argument("--prompt-source", default=None, help="manual | gemini | augment | hybrid | auto")
    extract.add_argument("--device", default="auto")
    extract.add_argument("--flip-depth", action="store_true")
    extract.add_argument("--ordering", default=None, help="boundary | learned")
    extract.add_argument("--ranker-model", default=None, help="Path to a trained order-ranker JSON file")
    extract.add_argument("--no-parallax", action="store_true")
    add_frontier_base_arguments(extract)
    extract.set_defaults(func=cmd_extract)

    export_design = sub.add_parser("export-design", help="Normalize an existing LayerForge run into the canonical DALG design-manifest JSON")
    export_design.add_argument("--run-dir", required=True, help="Run directory containing manifest.json")
    export_design.add_argument("--output", default=None, help="Optional output JSON path; defaults to <run-dir>/dalg_manifest.json")
    export_design.set_defaults(func=cmd_export_design)

    transparent = sub.add_parser("transparent", help="Approximate transparent or semi-transparent foreground decomposition using clean-background estimation and alpha blending")
    transparent.add_argument("--input", required=True)
    transparent.add_argument("--output", required=True)
    transparent.add_argument("--config", default="configs/fast.yaml")
    transparent.add_argument("--segmenter", default=None, help="classical | mask2former | grounded_sam2 | gemini")
    transparent.add_argument("--depth", default=None, help="geometric_luminance | depth_pro | depth_anything_v2 | marigold | ensemble")
    transparent.add_argument("--prompt", default=None, help="Prompt describing the transparent target, e.g. 'glass cup' or 'watermark text'")
    transparent.add_argument("--point", default=None, help="Optional x,y point hint in image pixels")
    transparent.add_argument("--box", default=None, help="Optional x1,y1,x2,y2 box hint in image pixels")
    transparent.add_argument("--target-name", default=None, help="Optional exact layer name override after decomposition")
    transparent.add_argument("--prompt-source", default=None, help="manual | gemini | augment | hybrid | auto")
    transparent.add_argument("--device", default="auto")
    transparent.add_argument("--flip-depth", action="store_true")
    transparent.add_argument("--ordering", default=None, help="boundary | learned")
    transparent.add_argument("--ranker-model", default=None, help="Path to a trained order-ranker JSON file")
    add_frontier_base_arguments(transparent)
    transparent.set_defaults(func=cmd_transparent)

    bench = sub.add_parser("benchmark", help="Run a lightweight synthetic benchmark and write a CSV/JSON report")
    bench.add_argument("--dataset-dir", required=True)
    bench.add_argument("--output-dir", required=True)
    bench.add_argument("--config", default="configs/fast.yaml")
    bench.add_argument("--segmenter", default="classical")
    bench.add_argument("--depth", default="geometric_luminance")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--ordering", default=None, help="boundary | learned")
    bench.add_argument("--ranker-model", default=None, help="Path to a trained order-ranker JSON file")
    bench.add_argument("--max-scenes", type=int, default=None)
    bench.set_defaults(func=cmd_benchmark)

    coco = sub.add_parser("benchmark-coco-panoptic", help="Benchmark visible semantic grouping on COCO Panoptic val2017 using coarse LayerForge groups")
    coco.add_argument("--dataset-dir", required=True, help="Directory containing val2017/ and annotations/panoptic_val2017*")
    coco.add_argument("--output-dir", required=True)
    coco.add_argument("--config", default="configs/fast.yaml")
    coco.add_argument("--segmenter", default="mask2former")
    coco.add_argument("--prompts", default=None)
    coco.add_argument("--prompt-source", default=None)
    coco.add_argument("--device", default="auto")
    coco.add_argument("--max-images", type=int, default=None)
    coco.add_argument("--seed", type=int, default=7)
    coco.set_defaults(func=cmd_benchmark_coco_panoptic)

    ade = sub.add_parser("benchmark-ade20k", help="Benchmark visible semantic grouping on ADE20K SceneParse150 validation using coarse LayerForge groups")
    ade.add_argument("--dataset-dir", required=True, help="Directory containing ADEChallengeData2016/")
    ade.add_argument("--output-dir", required=True)
    ade.add_argument("--config", default="configs/fast.yaml")
    ade.add_argument("--segmenter", default="mask2former")
    ade.add_argument("--prompts", default=None)
    ade.add_argument("--prompt-source", default=None)
    ade.add_argument("--device", default="auto")
    ade.add_argument("--max-images", type=int, default=None)
    ade.add_argument("--seed", type=int, default=7)
    ade.set_defaults(func=cmd_benchmark_ade20k)

    diode = sub.add_parser("benchmark-diode", help="Benchmark monocular depth on the public DIODE validation split")
    diode.add_argument("--dataset-dir", required=True, help="Directory containing val/indoors and val/outdoor, or the val directory itself")
    diode.add_argument("--output-dir", required=True)
    diode.add_argument("--config", default="configs/diode_depthpro.yaml")
    diode.add_argument("--depth", default="depth_pro", help="geometric_luminance | depth_pro | depth_anything_v2 | marigold | ensemble")
    diode.add_argument("--alignment", default="auto", help="auto | none | scale | scale_shift")
    diode.add_argument("--device", default="auto")
    diode.add_argument("--max-images", type=int, default=None)
    diode.add_argument("--seed", type=int, default=7)
    diode.set_defaults(func=cmd_benchmark_diode)

    train = sub.add_parser("train-ranker", help="Train a lightweight pairwise layer-order ranker on the synthetic dataset")
    train.add_argument("--dataset-dir", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config", default="configs/fast.yaml")
    train.add_argument("--segmenter", default="classical")
    train.add_argument("--depth", default="geometric_luminance")
    train.add_argument("--device", default="auto")
    train.add_argument("--max-scenes", type=int, default=None)
    train.set_defaults(func=cmd_train_ranker)

    webui = sub.add_parser("webui", help="Serve the local browser interface for editors, reviewers, and non-CLI users")
    webui.add_argument("--host", default="127.0.0.1")
    webui.add_argument("--port", type=int, default=8765)
    webui.add_argument("--open-browser", action="store_true")
    webui.set_defaults(func=cmd_webui)

    doctor = sub.add_parser("doctor", help="Report LayerForge runtime, optional backend, GPU, and path readiness")
    doctor.add_argument("--config", default="configs/fast.yaml")
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--cache-dir", default=None)
    doctor.add_argument("--output-dir", default=None)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
