from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from layerforge.cli import main


def test_extract_frontier_uses_selected_run(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (120, 140, 180)).save(image_path)

    winner_dir = tmp_path / "winner_extract"
    winner_dir.mkdir(parents=True)
    (winner_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (winner_dir / "metrics.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fail_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("native pipeline should not run when --frontier is selected")

    def fake_frontier(**kwargs):  # type: ignore[no-untyped-def]
        captured["frontier_kwargs"] = kwargs
        return {
            "selected_label": "LayerForge native",
            "run_dir": winner_dir,
            "manifest_path": winner_dir / "manifest.json",
            "metrics_path": winner_dir / "metrics.json",
            "summary_path": tmp_path / "frontier_summary.json",
        }

    def fake_export(path_or_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured["path_or_dir"] = Path(path_or_dir)
        captured["export_kwargs"] = kwargs
        return {"selected_target": {"name": "wheel"}}

    monkeypatch.setattr("layerforge.cli.LayerForgePipeline", fail_pipeline)
    monkeypatch.setattr("layerforge.cli.run_single_image_frontier_selection", fake_frontier, raising=False)
    monkeypatch.setattr("layerforge.cli.export_target_assets", fake_export)

    exit_code = main(
        [
            "extract",
            "--input",
            str(image_path),
            "--output",
            str(tmp_path / "extract_output"),
            "--config",
            "configs/frontier.yaml",
            "--frontier",
            "--prompt",
            "wheel",
        ]
    )

    assert exit_code == 0
    assert captured["path_or_dir"] == winner_dir
    assert captured["export_kwargs"]["prompt"] == "wheel"


def test_transparent_frontier_uses_selected_run(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (120, 140, 180)).save(image_path)

    winner_dir = tmp_path / "winner_transparent"
    winner_dir.mkdir(parents=True)
    (winner_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (winner_dir / "metrics.json").write_text(json.dumps({"recompose_psnr": 30.0}), encoding="utf-8")

    captured: dict[str, object] = {}

    def fail_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("native pipeline should not run when --frontier is selected")

    def fake_frontier(**kwargs):  # type: ignore[no-untyped-def]
        captured["frontier_kwargs"] = kwargs
        return {
            "selected_label": "Qwen + graph preserve (3)",
            "run_dir": winner_dir,
            "manifest_path": winner_dir / "manifest.json",
            "metrics_path": winner_dir / "metrics.json",
            "summary_path": tmp_path / "frontier_summary.json",
        }

    def fake_export(path_or_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured["path_or_dir"] = Path(path_or_dir)
        captured["export_kwargs"] = kwargs
        return {"selected_target": {"name": "glass"}}

    monkeypatch.setattr("layerforge.cli.LayerForgePipeline", fail_pipeline)
    monkeypatch.setattr("layerforge.cli.run_single_image_frontier_selection", fake_frontier, raising=False)
    monkeypatch.setattr("layerforge.cli.export_transparent_assets", fake_export)

    exit_code = main(
        [
            "transparent",
            "--input",
            str(image_path),
            "--output",
            str(tmp_path / "transparent_output"),
            "--config",
            "configs/frontier.yaml",
            "--frontier",
            "--prompt",
            "glass",
        ]
    )

    assert exit_code == 0
    assert captured["path_or_dir"] == winner_dir
    assert captured["export_kwargs"]["prompt"] == "glass"
