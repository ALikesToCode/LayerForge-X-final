from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from layerforge.frontier import materialize_frontier_selection


ROOT = Path(__file__).resolve().parents[1]


def test_run_frontier_comparison_dry_run(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (110, 130, 170)).save(image_path)

    output_root = tmp_path / "frontier"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_frontier_comparison.py"),
        "--inputs",
        str(image_path),
        "--output-root",
        str(output_root),
        "--qwen-layers",
        "4",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["rows"] == 0

    summary_json = output_root / "frontier_summary.json"
    summary_md = output_root / "frontier_summary.md"
    assert summary_json.exists()
    assert summary_md.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(summary["inputs"]) == 1
    assert summary["qwen_hybrid_modes"] == ["preserve", "reorder"]
    assert summary["best_by_image"] == []
    assert len(summary["commands"]) == 5
    assert all(item["status"] == "dry-run" for item in summary["commands"])


def test_materialize_frontier_selection_refuses_source_inside_target(tmp_path: Path) -> None:
    target = tmp_path / "run_output"
    run_dir = target / "frontier" / "sample" / "native"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="selected run is inside"):
        materialize_frontier_selection(
            {"run_dir": run_dir, "selected_label": "LayerForge native", "summary_path": target / "frontier" / "frontier_summary.json", "selection": {}},
            target,
            frontier_root=target / "frontier",
        )

    assert run_dir.exists()


def test_materialize_frontier_selection_refuses_non_layerforge_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "winner"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "important"
    target.mkdir()
    (target / "notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-LayerForge output"):
        materialize_frontier_selection(
            {"run_dir": run_dir, "selected_label": "LayerForge native", "summary_path": tmp_path / "frontier_summary.json", "selection": {}},
            target,
        )

    assert (target / "notes.txt").read_text(encoding="utf-8") == "keep me"
