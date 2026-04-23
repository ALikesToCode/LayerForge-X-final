from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config
from .editability import evaluate_run_editability
from .proposals import build_frontier_candidate_specs
from .self_eval import choose_best_candidates
from .utils import write_json

ROOT = Path(__file__).resolve().parents[2]


def add_frontier_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--input-dir", help="Directory containing input images")
    parser.add_argument("--inputs", nargs="*", help="Explicit input image paths")
    parser.add_argument("--output-root", default="runs/frontier_comparison")
    parser.add_argument("--native-config", default="configs/best_score.yaml")
    parser.add_argument("--native-segmenter", default="grounded_sam2")
    parser.add_argument("--native-depth", default="ensemble")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--peeling-config", default="configs/recursive_peeling.yaml")
    parser.add_argument("--peeling-segmenter", default="grounded_sam2")
    parser.add_argument("--peeling-depth", default="ensemble")
    parser.add_argument("--skip-peeling", action="store_true")
    parser.add_argument("--qwen-model", default="Qwen/Qwen-Image-Layered")
    parser.add_argument("--qwen-layers", default="4,6,8")
    parser.add_argument("--qwen-resolution", type=int, default=640)
    parser.add_argument("--qwen-steps", type=int, default=20)
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--qwen-offload", default="sequential", choices=["none", "model", "sequential"])
    parser.add_argument("--qwen-hybrid-modes", default="preserve,reorder")
    parser.add_argument("--qwen-merge-external-layers", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LayerForge-X++ frontier comparison bank and select the best decomposition per image."
    )
    return add_frontier_arguments(parser)


def build_single_image_frontier_args(
    *,
    input_path: str | Path,
    output_root: str | Path,
    native_config: str = "configs/best_score.yaml",
    native_segmenter: str = "grounded_sam2",
    native_depth: str = "ensemble",
    skip_native: bool = False,
    peeling_config: str = "configs/recursive_peeling.yaml",
    peeling_segmenter: str = "grounded_sam2",
    peeling_depth: str = "ensemble",
    skip_peeling: bool = False,
    qwen_model: str = "Qwen/Qwen-Image-Layered",
    qwen_layers: str = "3,4,6,8",
    qwen_resolution: int = 640,
    qwen_steps: int = 10,
    qwen_device: str = "cuda",
    qwen_dtype: str = "bfloat16",
    qwen_offload: str = "sequential",
    qwen_hybrid_modes: str = "preserve,reorder",
    qwen_merge_external_layers: bool = False,
    skip_existing: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=None,
        inputs=[str(input_path)],
        output_root=str(output_root),
        native_config=native_config,
        native_segmenter=native_segmenter,
        native_depth=native_depth,
        skip_native=skip_native,
        peeling_config=peeling_config,
        peeling_segmenter=peeling_segmenter,
        peeling_depth=peeling_depth,
        skip_peeling=skip_peeling,
        qwen_model=qwen_model,
        qwen_layers=qwen_layers,
        qwen_resolution=qwen_resolution,
        qwen_steps=qwen_steps,
        qwen_device=qwen_device,
        qwen_dtype=qwen_dtype,
        qwen_offload=qwen_offload,
        qwen_hybrid_modes=qwen_hybrid_modes,
        qwen_merge_external_layers=qwen_merge_external_layers,
        limit=1,
        skip_existing=skip_existing,
        dry_run=False,
    )


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    inputs: list[Path] = []
    if args.inputs:
        inputs.extend(Path(p) for p in args.inputs)
    if args.input_dir:
        root = Path(args.input_dir)
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            inputs.extend(sorted(root.glob(pattern)))
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        absolute = path if path.is_absolute() else ROOT / path
        absolute = absolute.resolve()
        if absolute in seen:
            continue
        seen.add(absolute)
        resolved.append(absolute)
    if args.limit is not None:
        resolved = resolved[: max(0, int(args.limit))]
    if not resolved:
        raise SystemExit("No input images found. Use --input-dir or --inputs.")
    return resolved


def find_layerforge_bin() -> str:
    candidates = [
        ROOT / ".venv" / "bin" / "layerforge",
        Path(sys.executable).resolve().parent / "layerforge",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "layerforge"


def to_repo_relative(path: str | Path) -> str:
    value = Path(path)
    try:
        resolved = value.resolve()
    except FileNotFoundError:
        resolved = value if value.is_absolute() else (ROOT / value).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(value)


def parse_hybrid_modes(raw: str) -> list[str]:
    allowed = {"preserve", "reorder"}
    seen: list[str] = []
    for item in raw.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in allowed:
            raise SystemExit(f"Unsupported hybrid mode '{mode}'. Expected a comma-separated subset of {sorted(allowed)}.")
        if mode not in seen:
            seen.append(mode)
    if not seen:
        raise SystemExit("At least one hybrid mode must be requested.")
    return seen


def resolve_frontier_native_config(config_path: str | Path | None) -> str:
    raw = str(config_path or "configs/best_score.yaml")
    return "configs/best_score.yaml" if raw == "configs/fast.yaml" else raw


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> dict:
    printable = " ".join(shlex.quote(part) for part in cmd)
    if dry_run:
        return {"command": printable, "returncode": None, "status": "dry-run", "duration_sec": None}
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    duration_sec = time.monotonic() - started
    return {
        "command": printable,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "ok" if proc.returncode == 0 else "failed",
        "duration_sec": round(float(duration_sec), 4),
    }


def should_skip(run_dir: Path, marker_name: str, *, skip_existing: bool) -> bool:
    return bool(skip_existing and (run_dir / marker_name).exists())


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_row(label: str, run_dir: Path, *, image: Path, command_results: list[dict], extra: dict | None = None) -> dict:
    metrics = load_json_if_exists(run_dir / "metrics.json") or {}
    manifest = load_json_if_exists(run_dir / "manifest.json") or {}
    editability = load_json_if_exists(run_dir / "editability_metrics.json")
    if editability is None and (run_dir / "manifest.json").exists():
        editability = evaluate_run_editability(run_dir)
    graph_payload = load_json_if_exists(run_dir / "debug" / "layer_graph.json") or load_json_if_exists(run_dir / "debug" / "peeling_graph.json") or {}
    occlusion_edges = graph_payload.get("occlusion_edges", [])
    duration_values = [float(item["duration_sec"]) for item in command_results if item.get("duration_sec") is not None]
    row = {
        "image": to_repo_relative(image),
        "label": label,
        "run_dir": to_repo_relative(run_dir),
        "mode": metrics.get("mode", "layerforge"),
        "num_layers": metrics.get("num_layers", manifest.get("layers_requested")),
        "recompose_psnr": metrics.get("recompose_psnr"),
        "recompose_ssim": metrics.get("recompose_ssim"),
        "mean_amodal_extra_ratio": metrics.get("mean_amodal_extra_ratio"),
        "effect_layer_count": metrics.get("effect_layer_count"),
        "ordering_method": metrics.get("ordering_method"),
        "visual_order_mode": metrics.get("visual_order_mode"),
        "selected_visual_order": metrics.get("selected_visual_order", metrics.get("selected_external_visual_order")),
        "graph_order_psnr": metrics.get("graph_order_psnr"),
        "graph_order_ssim": metrics.get("graph_order_ssim"),
        "graph_order_applied": metrics.get("graph_order_applied"),
        "graph_order_guardrail_triggered": metrics.get("graph_order_guardrail_triggered"),
        "graph_order_guardrail_reason": metrics.get("graph_order_guardrail_reason"),
        "selected_external_order_psnr": metrics.get("selected_external_order_psnr"),
        "selected_external_order_ssim": metrics.get("selected_external_order_ssim"),
        "preserve_external_order": metrics.get("preserve_external_order"),
        "merge_external_layers": metrics.get("merge_external_layers"),
        "segmentation_method": metrics.get("segmentation_method"),
        "depth_method": metrics.get("depth_method"),
        "occlusion_edge_count": len(occlusion_edges),
        "duration_sec": round(sum(duration_values), 4) if duration_values else None,
        "has_graph": (run_dir / "debug" / "layer_graph.json").exists() or (run_dir / "debug" / "peeling_graph.json").exists(),
        "has_ordered_layers": (run_dir / "layers_ordered_rgba").exists(),
        "status": "ok",
    }
    if editability:
        row.update({key: value for key, value in editability.items() if key != "preview_paths"})
    if extra:
        row.update(extra)
    return row


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(str(row["label"]), []).append(row)
    summaries: list[dict] = []
    for label, items in grouped.items():
        psnr = [float(item["recompose_psnr"]) for item in items if item.get("recompose_psnr") is not None]
        ssim = [float(item["recompose_ssim"]) for item in items if item.get("recompose_ssim") is not None]
        scores = [float(item["self_eval_score"]) for item in items if item.get("self_eval_score") is not None]
        summaries.append(
            {
                "label": label,
                "images": len(items),
                "mean_psnr": sum(psnr) / len(psnr) if psnr else None,
                "mean_ssim": sum(ssim) / len(ssim) if ssim else None,
                "mean_self_eval_score": sum(scores) / len(scores) if scores else None,
            }
        )
    return sorted(summaries, key=lambda item: item["label"])


def to_markdown(rows: list[dict], best_by_image: list[dict]) -> str:
    lines = [
        "| Image | Method | Mode | Layers | PSNR | SSIM | Self-eval | Best | Status |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    best_lookup = {str(item["image"]): str(item["label"]) for item in best_by_image}
    for row in rows:
        best = "yes" if best_lookup.get(str(row["image"])) == str(row["label"]) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(str(row["image"])).stem,
                    str(row["label"]),
                    str(row.get("mode", "")),
                    "" if row.get("num_layers") is None else str(int(float(row["num_layers"]))),
                    "" if row.get("recompose_psnr") is None else f"{float(row['recompose_psnr']):.4f}",
                    "" if row.get("recompose_ssim") is None else f"{float(row['recompose_ssim']):.4f}",
                    "" if row.get("self_eval_score") is None else f"{float(row['self_eval_score']):.4f}",
                    best,
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_best_selection_files(output_root: Path, best_by_image: list[dict]) -> None:
    for row in best_by_image:
        scene_root = output_root / Path(str(row["image"])).stem
        scene_root.mkdir(parents=True, exist_ok=True)
        selection = {
            "image": row["image"],
            "selected_label": row["label"],
            "run_dir": row["run_dir"],
            "self_eval_score": row["self_eval_score"],
            "self_eval_components": row.get("self_eval_components"),
            "why_selected": row.get("self_eval_reason"),
        }
        (scene_root / "best_decomposition.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        (scene_root / "why_selected.md").write_text(
            f"# {row['label']}\n\n{row.get('self_eval_reason', '')}\n",
            encoding="utf-8",
        )


def run_frontier_comparison(args: argparse.Namespace) -> int:
    inputs = resolve_inputs(args)
    hybrid_modes = parse_hybrid_modes(args.qwen_hybrid_modes)
    output_root = (ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    layerforge_bin = find_layerforge_bin()
    qwen_score_script = ROOT / "scripts" / "score_qwen_raw_layers.py"
    qwen_layers = [int(part.strip()) for part in args.qwen_layers.split(",") if part.strip()]
    frontier_cfg = load_config(args.native_config)
    self_eval_weights = frontier_cfg.get("self_eval", {}).get("weights")

    rows: list[dict] = []
    commands: list[dict] = []
    for image in inputs:
        specs = build_frontier_candidate_specs(
            image=image,
            output_root=output_root,
            qwen_layers=qwen_layers,
            hybrid_modes=hybrid_modes,
            layerforge_bin=layerforge_bin,
            python_bin=python_bin,
            qwen_score_script=qwen_score_script,
            native_config=args.native_config,
            native_segmenter=args.native_segmenter,
            native_depth=args.native_depth,
            peeling_config=args.peeling_config,
            peeling_segmenter=args.peeling_segmenter,
            peeling_depth=args.peeling_depth,
            qwen_resolution=args.qwen_resolution,
            qwen_steps=args.qwen_steps,
            qwen_device=args.qwen_device,
            qwen_dtype=args.qwen_dtype,
            qwen_offload=args.qwen_offload,
            qwen_model=args.qwen_model,
            merge_external_layers=args.qwen_merge_external_layers,
            include_native=not args.skip_native,
            include_peeling=not args.skip_peeling,
        )

        for spec in specs:
            command_results: list[dict] = []
            if should_skip(spec.run_dir, spec.marker_name, skip_existing=args.skip_existing):
                result = {"command": f"skip existing {spec.run_dir}", "returncode": None, "status": "skipped", "duration_sec": None}
                commands.append(result)
                command_results.append(result)
            else:
                result = run_command(spec.command, cwd=ROOT, dry_run=args.dry_run)
                commands.append(result)
                command_results.append(result)

            if not args.dry_run and result.get("status") == "ok":
                for post_cmd in spec.post_commands:
                    if (spec.run_dir / "metrics.json").exists():
                        break
                    post_result = run_command(post_cmd, cwd=ROOT, dry_run=False)
                    commands.append(post_result)
                    command_results.append(post_result)
                    if post_result.get("status") != "ok":
                        break

            if not args.dry_run and (spec.run_dir / "metrics.json").exists():
                rows.append(collect_row(spec.label, spec.run_dir, image=image, command_results=command_results, extra=spec.extra))
            elif not args.dry_run and any(item.get("status") == "failed" for item in command_results):
                failed = next(item for item in command_results if item.get("status") == "failed")
                rows.append(
                    {
                        "image": to_repo_relative(image),
                        "label": spec.label,
                        "run_dir": to_repo_relative(spec.run_dir),
                        "status": "failed",
                        "error": failed.get("stderr", ""),
                        **spec.extra,
                    }
                )

    scored_rows, best_by_image = choose_best_candidates(rows, weights=self_eval_weights)
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    all_rows = failed_rows + scored_rows
    if not args.dry_run:
        write_best_selection_files(output_root, best_by_image)

    summary = {
        "inputs": [to_repo_relative(path) for path in inputs],
        "output_root": to_repo_relative(output_root),
        "qwen_hybrid_modes": hybrid_modes,
        "self_eval_weights": self_eval_weights,
        "rows": all_rows,
        "best_by_image": best_by_image,
        "aggregates": summarize_rows(scored_rows),
        "commands": commands,
    }
    (output_root / "frontier_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "frontier_summary.md").write_text(to_markdown(all_rows, best_by_image), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": to_repo_relative(output_root / "frontier_summary.json"),
                "summary_md": to_repo_relative(output_root / "frontier_summary.md"),
                "rows": len(all_rows),
            },
            indent=2,
        )
    )
    return 0


def run_single_image_frontier_selection(
    *,
    input_path: str | Path,
    output_root: str | Path,
    native_config: str = "configs/best_score.yaml",
    native_segmenter: str = "grounded_sam2",
    native_depth: str = "ensemble",
    skip_native: bool = False,
    peeling_config: str = "configs/recursive_peeling.yaml",
    peeling_segmenter: str = "grounded_sam2",
    peeling_depth: str = "ensemble",
    skip_peeling: bool = False,
    qwen_model: str = "Qwen/Qwen-Image-Layered",
    qwen_layers: str = "3,4,6,8",
    qwen_resolution: int = 640,
    qwen_steps: int = 10,
    qwen_device: str = "cuda",
    qwen_dtype: str = "bfloat16",
    qwen_offload: str = "sequential",
    qwen_hybrid_modes: str = "preserve,reorder",
    qwen_merge_external_layers: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    args = build_single_image_frontier_args(
        input_path=input_path,
        output_root=output_root,
        native_config=native_config,
        native_segmenter=native_segmenter,
        native_depth=native_depth,
        skip_native=skip_native,
        peeling_config=peeling_config,
        peeling_segmenter=peeling_segmenter,
        peeling_depth=peeling_depth,
        skip_peeling=skip_peeling,
        qwen_model=qwen_model,
        qwen_layers=qwen_layers,
        qwen_resolution=qwen_resolution,
        qwen_steps=qwen_steps,
        qwen_device=qwen_device,
        qwen_dtype=qwen_dtype,
        qwen_offload=qwen_offload,
        qwen_hybrid_modes=qwen_hybrid_modes,
        qwen_merge_external_layers=qwen_merge_external_layers,
        skip_existing=skip_existing,
    )
    run_frontier_comparison(args)
    summary_path = Path(output_root) / "frontier_summary.json"
    summary = _read_json(summary_path)
    best_by_image = summary.get("best_by_image") or []
    if not best_by_image:
        raise RuntimeError("Frontier comparison produced no successful candidate selection")
    selection = dict(best_by_image[0])
    run_dir_value = selection.get("run_dir")
    if not run_dir_value:
        raise RuntimeError("Frontier comparison selection did not include a run directory")
    run_dir = Path(run_dir_value)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    return {
        "selected_label": selection.get("label"),
        "run_dir": run_dir,
        "manifest_path": run_dir / "manifest.json",
        "metrics_path": run_dir / "metrics.json",
        "summary_path": summary_path,
        "selection": selection,
    }


def materialize_frontier_selection(
    selection_result: dict[str, Any],
    target_dir: str | Path,
    *,
    frontier_root: str | Path | None = None,
) -> dict[str, Path]:
    run_dir = Path(selection_result["run_dir"])
    output_dir = Path(target_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(run_dir, output_dir)
    selection_path = write_json(
        output_dir / "frontier_selection.json",
        {
            "selected_label": selection_result.get("selected_label"),
            "summary_path": to_repo_relative(selection_result.get("summary_path", "")),
            "source_run_dir": to_repo_relative(run_dir),
            "selection": selection_result.get("selection", {}),
        },
    )
    why_selected = selection_result.get("selection", {}).get("self_eval_reason")
    if why_selected:
        (output_dir / "why_selected.md").write_text(
            f"# Selected frontier winner\n\n{why_selected}\n",
            encoding="utf-8",
        )
    if frontier_root is not None:
        write_json(
            output_dir / "frontier_workspace.json",
            {
                "frontier_root": to_repo_relative(frontier_root),
                "summary_path": to_repo_relative(selection_result.get("summary_path", "")),
            },
        )
    return {
        "output_dir": output_dir,
        "manifest_path": output_dir / "manifest.json",
        "metrics_path": output_dir / "metrics.json",
        "selection_path": selection_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_frontier_comparison(args)
