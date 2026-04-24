from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_benchmark_runner_skips_missing_datasets(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_public_benchmarks_if_present.py"),
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads((output_root / "public_benchmark_run_report.json").read_text(encoding="utf-8"))

    assert "skipped=3" in proc.stdout
    assert {row["status"] for row in report["benchmarks"]} == {"skipped"}
    assert (output_root / "public_benchmark_run_report.md").exists()


def test_public_benchmark_runner_dry_runs_present_datasets(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "coco_panoptic_val" / "val2017").mkdir(parents=True)
    (data_root / "coco_panoptic_val" / "annotations").mkdir(parents=True)
    (data_root / "ade20k" / "ADEChallengeData2016").mkdir(parents=True)
    (data_root / "diode" / "val").mkdir(parents=True)
    output_root = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_public_benchmarks_if_present.py"),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--preset",
            "world_best",
            "--max-images",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads((output_root / "public_benchmark_run_report.json").read_text(encoding="utf-8"))
    rows = report["benchmarks"]

    assert {row["status"] for row in rows} == {"dry-run"}
    assert "configs/world_best.yaml" in rows[0]["command"]
    assert "--max-images 2" in rows[0]["command"]
    assert "benchmark-diode" in rows[2]["command"]
