from __future__ import annotations

import argparse
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
        flip_depth=args.flip_depth,
        save_parallax=not args.no_parallax,
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
            flip_depth=args.flip_depth,
            save_parallax=not args.no_parallax,
        )
        print(f"{p.name} -> {Path(args.output_dir) / p.stem}")
    return 0


def cmd_enrich_qwen(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pipe = LayerForgePipeline(cfg, device=args.device)
    out = pipe.enrich_rgba_layers(args.input, args.layers_dir, args.output, depth_method=args.depth, flip_depth=args.flip_depth)
    print(f"manifest: {out.manifest_path}")
    print(f"metrics:  {out.metrics_path}")
    print(f"ordered enriched layers: {len(out.ordered_layer_paths)}")
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
    )
    print(f"benchmark report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="layerforge", description="LayerForge-X: depth-aware amodal layer graph generator")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run LayerForge-X on one RGB image")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config", default="configs/fast.yaml")
    run.add_argument("--segmenter", default=None, help="classical | mask2former | grounded_sam2")
    run.add_argument("--depth", default=None, help="geometric_luminance | depth_pro | depth_anything_v2 | marigold | ensemble")
    run.add_argument("--prompts", default=None, help="Comma-separated open-vocabulary prompts")
    run.add_argument("--device", default="auto")
    run.add_argument("--flip-depth", action="store_true")
    run.add_argument("--no-parallax", action="store_true")
    run.set_defaults(func=cmd_run)

    batch = sub.add_parser("batch", help="Run LayerForge-X on all images in a folder")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--config", default="configs/fast.yaml")
    batch.add_argument("--segmenter", default=None)
    batch.add_argument("--depth", default=None)
    batch.add_argument("--prompts", default=None)
    batch.add_argument("--device", default="auto")
    batch.add_argument("--flip-depth", action="store_true")
    batch.add_argument("--no-parallax", action="store_true")
    batch.set_defaults(func=cmd_batch)

    enrich = sub.add_parser("enrich-qwen", help="Import RGBA layers from Qwen-Image-Layered or any external decomposer and add LayerForge graph/depth/intrinsic metadata")
    enrich.add_argument("--input", required=True, help="Original RGB image")
    enrich.add_argument("--layers-dir", required=True, help="Folder containing external RGBA layer PNGs")
    enrich.add_argument("--output", required=True)
    enrich.add_argument("--config", default="configs/fast.yaml")
    enrich.add_argument("--depth", default=None)
    enrich.add_argument("--device", default="auto")
    enrich.add_argument("--flip-depth", action="store_true")
    enrich.set_defaults(func=cmd_enrich_qwen)

    bench = sub.add_parser("benchmark", help="Run a lightweight synthetic benchmark and write a CSV/JSON report")
    bench.add_argument("--dataset-dir", required=True)
    bench.add_argument("--output-dir", required=True)
    bench.add_argument("--config", default="configs/fast.yaml")
    bench.add_argument("--segmenter", default="classical")
    bench.add_argument("--depth", default="geometric_luminance")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--max-scenes", type=int, default=None)
    bench.set_defaults(func=cmd_benchmark)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
