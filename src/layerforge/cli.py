from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import LayerForgePipeline


def parse_prompts(text: str | None) -> list[str] | None:
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def cmd_run(args: argparse.Namespace) -> int:
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
