from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from layerforge.cli import main


def test_layerforge_frontier_dry_run_writes_summary(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (90, 120, 160)).save(image_path)

    output_root = tmp_path / "frontier"
    exit_code = main(
        [
            "frontier",
            "--inputs",
            str(image_path),
            "--output-root",
            str(output_root),
            "--qwen-layers",
            "4",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    summary_json = output_root / "frontier_summary.json"
    summary_md = output_root / "frontier_summary.md"
    assert summary_json.exists()
    assert summary_md.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["qwen_hybrid_modes"] == ["preserve", "reorder"]
    assert len(summary["inputs"]) == 1
    assert len(summary["commands"]) == 5
    assert summary["best_by_image"] == []
