from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_run_curated_comparison_dry_run(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (120, 140, 180)).save(image_path)

    output_root = tmp_path / "comparison"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_curated_comparison.py"),
        "--inputs",
        str(image_path),
        "--output-root",
        str(output_root),
        "--qwen-layers",
        "3,4",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["rows"] == 0

    summary_json = output_root / "comparison_summary.json"
    summary_md = output_root / "comparison_summary.md"
    assert summary_json.exists()
    assert summary_md.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(summary["inputs"]) == 1
    assert len(summary["commands"]) == 5
    assert all(item["status"] == "dry-run" for item in summary["commands"])
