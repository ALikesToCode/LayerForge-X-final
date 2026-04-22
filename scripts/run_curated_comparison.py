#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a reproducible native/Qwen/hybrid comparison over a curated image set.")
    p.add_argument("--input-dir", help="Directory containing input images")
    p.add_argument("--inputs", nargs="*", help="Explicit input image paths")
    p.add_argument("--output-root", default="runs/curated_comparison")
    p.add_argument("--native-config", default="configs/cutting_edge.yaml")
    p.add_argument("--native-segmenter", default="grounded_sam2")
    p.add_argument("--native-depth", default="depth_pro")
    p.add_argument("--skip-native", action="store_true", help="Skip native LayerForge runs and only execute Qwen raw plus Qwen+graph")
    p.add_argument("--qwen-model", default="Qwen/Qwen-Image-Layered")
    p.add_argument("--qwen-layers", default="3,4,6,8", help="Comma-separated layer counts")
    p.add_argument("--qwen-resolution", type=int, default=640)
    p.add_argument("--qwen-steps", type=int, default=20)
    p.add_argument("--qwen-device", default="cuda")
    p.add_argument("--qwen-dtype", default="bfloat16")
    p.add_argument("--qwen-offload", default="sequential", choices=["none", "model", "sequential"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true", help="Reuse existing output directories when the expected manifest/metrics file already exists")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


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


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> dict:
    printable = " ".join(shlex.quote(part) for part in cmd)
    if dry_run:
        return {"command": printable, "returncode": None, "status": "dry-run"}
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "command": printable,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "ok" if proc.returncode == 0 else "failed",
    }


def should_skip(run_dir: Path, marker_name: str, *, skip_existing: bool) -> bool:
    return bool(skip_existing and (run_dir / marker_name).exists())


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_row(label: str, run_dir: Path, *, image: Path, extra: dict | None = None) -> dict:
    metrics = load_json_if_exists(run_dir / "metrics.json") or {}
    manifest = load_json_if_exists(run_dir / "manifest.json") or {}
    row = {
        "image": str(image),
        "label": label,
        "run_dir": str(run_dir),
        "mode": metrics.get("mode", "layerforge"),
        "num_layers": metrics.get("num_layers", manifest.get("layers_requested")),
        "recompose_psnr": metrics.get("recompose_psnr"),
        "recompose_ssim": metrics.get("recompose_ssim"),
        "mean_amodal_extra_ratio": metrics.get("mean_amodal_extra_ratio"),
        "effect_layer_count": metrics.get("effect_layer_count"),
        "ordering_method": metrics.get("ordering_method"),
        "segmentation_method": metrics.get("segmentation_method"),
        "depth_method": metrics.get("depth_method"),
        "has_graph": (run_dir / "debug" / "layer_graph.json").exists() or (run_dir / "debug" / "peeling_graph.json").exists(),
        "has_ordered_layers": (run_dir / "layers_ordered_rgba").exists(),
        "status": "ok",
    }
    if extra:
        row.update(extra)
    return row


def to_markdown(rows: list[dict]) -> str:
    header = [
        "Image",
        "Method",
        "Mode",
        "Layers",
        "PSNR",
        "SSIM",
        "Amodal+",
        "Effects",
        "Graph",
        "Status",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(row["image"]).stem,
                    str(row["label"]),
                    str(row.get("mode", "")),
                    "" if row.get("num_layers") is None else str(int(row["num_layers"])),
                    "" if row.get("recompose_psnr") is None else f"{float(row['recompose_psnr']):.4f}",
                    "" if row.get("recompose_ssim") is None else f"{float(row['recompose_ssim']):.4f}",
                    "" if row.get("mean_amodal_extra_ratio") is None else f"{float(row['mean_amodal_extra_ratio']):.4f}",
                    "" if row.get("effect_layer_count") is None else str(int(row["effect_layer_count"])),
                    "yes" if row.get("has_graph") else "no",
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    inputs = resolve_inputs(args)
    output_root = (ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    layerforge_bin = find_layerforge_bin()
    qwen_score_script = ROOT / "scripts" / "score_qwen_raw_layers.py"
    qwen_layers = [int(part.strip()) for part in args.qwen_layers.split(",") if part.strip()]

    rows: list[dict] = []
    commands: list[dict] = []
    for image in inputs:
        slug = image.stem
        scene_root = output_root / slug
        scene_root.mkdir(parents=True, exist_ok=True)

        if not args.skip_native:
            native_dir = scene_root / "native"
            native_cmd = [
                layerforge_bin,
                "run",
                "--input",
                str(image),
                "--output",
                str(native_dir),
                "--config",
                args.native_config,
                "--segmenter",
                args.native_segmenter,
                "--depth",
                args.native_depth,
            ]
            if should_skip(native_dir, "metrics.json", skip_existing=args.skip_existing):
                commands.append({"command": f"skip existing {native_dir}", "returncode": None, "status": "skipped"})
            else:
                commands.append(run_command(native_cmd, cwd=ROOT, dry_run=args.dry_run))
            if not args.dry_run and (native_dir / "metrics.json").exists():
                rows.append(collect_row("LayerForge native", native_dir, image=image))
            elif not args.dry_run and commands[-1]["status"] == "failed":
                rows.append({"image": str(image), "label": "LayerForge native", "run_dir": str(native_dir), "status": "failed", "error": commands[-1].get("stderr", "")})

        for layer_count in qwen_layers:
            qwen_dir = scene_root / f"qwen_{layer_count}"
            qwen_cmd = [
                python_bin,
                str(ROOT / "scripts" / "run_qwen_image_layered.py"),
                "--input",
                str(image),
                "--output-dir",
                str(qwen_dir),
                "--model",
                args.qwen_model,
                "--layers",
                str(layer_count),
                "--resolution",
                str(args.qwen_resolution),
                "--steps",
                str(args.qwen_steps),
                "--device",
                args.qwen_device,
                "--dtype",
                args.qwen_dtype,
                "--offload",
                args.qwen_offload,
            ]
            if should_skip(qwen_dir, "manifest.json", skip_existing=args.skip_existing):
                commands.append({"command": f"skip existing {qwen_dir}", "returncode": None, "status": "skipped"})
            else:
                commands.append(run_command(qwen_cmd, cwd=ROOT, dry_run=args.dry_run))
            if not args.dry_run and (qwen_dir / "manifest.json").exists() and not (qwen_dir / "metrics.json").exists():
                score_cmd = [
                    python_bin,
                    str(qwen_score_script),
                    "--input",
                    str(image),
                    "--layers-dir",
                    str(qwen_dir),
                ]
                commands.append(run_command(score_cmd, cwd=ROOT, dry_run=False))
            if not args.dry_run and (qwen_dir / "manifest.json").exists():
                rows.append(
                    collect_row(
                        f"Qwen raw ({layer_count})",
                        qwen_dir,
                        image=image,
                        extra={"mode": "qwen_raw", "requested_layers": layer_count},
                    )
                )
            elif not args.dry_run and commands[-1]["status"] == "failed":
                rows.append({"image": str(image), "label": f"Qwen raw ({layer_count})", "run_dir": str(qwen_dir), "status": "failed", "error": commands[-1].get("stderr", "")})
                continue

            hybrid_dir = scene_root / f"qwen_{layer_count}_hybrid"
            hybrid_cmd = [
                layerforge_bin,
                "enrich-qwen",
                "--input",
                str(image),
                "--layers-dir",
                str(qwen_dir),
                "--output",
                str(hybrid_dir),
                "--config",
                args.native_config,
                "--depth",
                args.native_depth,
            ]
            if should_skip(hybrid_dir, "metrics.json", skip_existing=args.skip_existing):
                commands.append({"command": f"skip existing {hybrid_dir}", "returncode": None, "status": "skipped"})
            else:
                commands.append(run_command(hybrid_cmd, cwd=ROOT, dry_run=args.dry_run))
            if not args.dry_run and (hybrid_dir / "metrics.json").exists():
                rows.append(
                    collect_row(
                        f"Qwen + graph ({layer_count})",
                        hybrid_dir,
                        image=image,
                        extra={"requested_layers": layer_count},
                    )
                )
            elif not args.dry_run and commands[-1]["status"] == "failed":
                rows.append({"image": str(image), "label": f"Qwen + graph ({layer_count})", "run_dir": str(hybrid_dir), "status": "failed", "error": commands[-1].get("stderr", "")})

    summary = {
        "inputs": [str(path) for path in inputs],
        "output_root": str(output_root),
        "rows": rows,
        "commands": commands,
    }
    (output_root / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "comparison_summary.md").write_text(to_markdown(rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str(output_root / "comparison_summary.json"),
                "summary_md": str(output_root / "comparison_summary.md"),
                "rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
