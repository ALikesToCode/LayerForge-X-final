#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optional public LayerForge benchmarks only when the required datasets already exist locally."
    )
    parser.add_argument("--data-root", default="data", help="Root containing coco_panoptic_val/, ade20k/, and diode/ folders")
    parser.add_argument("--output-root", default="results/public_benchmarks")
    parser.add_argument("--config", default=None, help="Override config for COCO/ADE; defaults to world_best config when --preset world_best is used, otherwise fast.yaml")
    parser.add_argument("--preset", default=None, choices=["world_best"], help="Use a named preset config where public benchmark CLIs support config files")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=None, help="Optional image cap for COCO/ADE/DIODE smoke runs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layerforge-bin", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write the command plan without executing benchmark commands")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def find_layerforge_bin(raw: str | None = None) -> str:
    if raw:
        return raw
    candidates = [
        ROOT / ".venv" / "bin" / "layerforge",
        Path(sys.executable).resolve().parent / "layerforge",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "layerforge"


def default_config(args: argparse.Namespace) -> str:
    if args.config:
        return args.config
    if args.preset == "world_best":
        return "configs/world_best.yaml"
    return "configs/fast.yaml"


def dataset_specs(data_root: Path, output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    layerforge = find_layerforge_bin(args.layerforge_bin)
    config = default_config(args)
    max_images_args = ["--max-images", str(args.max_images)] if args.max_images is not None else []
    return [
        {
            "name": "coco_panoptic",
            "dataset_dir": data_root / "coco_panoptic_val",
            "required_paths": [data_root / "coco_panoptic_val" / "val2017", data_root / "coco_panoptic_val" / "annotations"],
            "summary_path": output_root / "coco_panoptic" / "coco_panoptic_group_benchmark_summary.json",
            "command": [
                layerforge,
                "benchmark-coco-panoptic",
                "--dataset-dir",
                str(data_root / "coco_panoptic_val"),
                "--output-dir",
                str(output_root / "coco_panoptic"),
                "--config",
                config,
                "--segmenter",
                "mask2former",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                *max_images_args,
            ],
        },
        {
            "name": "ade20k",
            "dataset_dir": data_root / "ade20k",
            "required_paths": [data_root / "ade20k" / "ADEChallengeData2016"],
            "summary_path": output_root / "ade20k" / "ade20k_group_benchmark_summary.json",
            "command": [
                layerforge,
                "benchmark-ade20k",
                "--dataset-dir",
                str(data_root / "ade20k"),
                "--output-dir",
                str(output_root / "ade20k"),
                "--config",
                config,
                "--segmenter",
                "mask2former",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                *max_images_args,
            ],
        },
        {
            "name": "diode",
            "dataset_dir": data_root / "diode",
            "required_paths": [data_root / "diode" / "val"],
            "summary_path": output_root / "diode" / "diode_depth_benchmark_summary.json",
            "command": [
                layerforge,
                "benchmark-diode",
                "--dataset-dir",
                str(data_root / "diode"),
                "--output-dir",
                str(output_root / "diode"),
                "--config",
                "configs/diode_depthpro.yaml",
                "--depth",
                "depth_pro",
                "--alignment",
                "auto",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                *max_images_args,
            ],
        },
    ]


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "dry-run", "returncode": None, "command": command_text(command)}
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": command_text(command),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def markdown_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Public Benchmark Run Report",
        "",
        "| Benchmark | Status | Dataset | Output | Command |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["status"]),
                    str(row["dataset_dir"]),
                    str(row.get("summary_path") or ""),
                    f"`{row.get('command', '')}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_public_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in dataset_specs(data_root, output_root, args):
        missing = [str(path) for path in spec["required_paths"] if not path.exists()]
        row = {
            "name": spec["name"],
            "dataset_dir": str(spec["dataset_dir"]),
            "summary_path": str(spec["summary_path"]),
            "missing_required_paths": missing,
        }
        if missing:
            row.update({"status": "skipped", "reason": "dataset not present", "command": command_text(spec["command"])})
        else:
            spec["summary_path"].parent.mkdir(parents=True, exist_ok=True)
            result = run_command(spec["command"], dry_run=args.dry_run)
            row.update(result)
            if spec["summary_path"].exists():
                row["result_summary"] = json.loads(spec["summary_path"].read_text(encoding="utf-8"))
        rows.append(row)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
        "benchmarks": rows,
    }
    (output_root / "public_benchmark_run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "public_benchmark_run_report.md").write_text(markdown_report(rows), encoding="utf-8")
    return report


def main() -> int:
    report = run_public_benchmarks(parse_args())
    print(output_summary(report))
    return 1 if any(row.get("status") == "failed" for row in report["benchmarks"]) else 0


def output_summary(report: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for row in report["benchmarks"]:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    return "public benchmarks: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
