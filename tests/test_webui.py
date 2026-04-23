from __future__ import annotations

import base64
import json
from pathlib import Path

from layerforge.webui import run_webui_job


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_webui_run_job_creates_accessible_outputs(tmp_path: Path) -> None:
    image_path = REPO_ROOT / "examples" / "synth" / "scene_000" / "image.png"
    payload = {
        "mode": "run",
        "filename": image_path.name,
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "config": "configs/fast.yaml",
        "segmenter": "classical",
        "depth": "geometric_luminance",
        "device": "cpu",
        "no_parallax": True,
    }
    result = run_webui_job(REPO_ROOT, payload, work_root=tmp_path / "webui")
    assert result["status"] == "ok"
    assert result["mode"] == "run"
    assert result["summary_metrics"]["num_layers"] >= 1
    assert result["urls"]["manifest"]
    assert result["urls"]["metrics"]
    assert result["urls"]["dalg"]
    assert result["previews"]


def test_webui_frontier_mode_returns_selected_best_run(tmp_path: Path, monkeypatch) -> None:
    image_path = REPO_ROOT / "examples" / "synth" / "scene_000" / "image.png"
    work_root = tmp_path / "webui"

    def fake_run_frontier(args) -> int:
        output_root = Path(args.output_root)
        scene_root = output_root / image_path.stem
        best_run = scene_root / "native"
        (best_run / "debug").mkdir(parents=True, exist_ok=True)
        (best_run / "manifest.json").write_text("{}", encoding="utf-8")
        (best_run / "metrics.json").write_text(json.dumps({"num_layers": 3, "recompose_psnr": 31.0, "recompose_ssim": 0.91}), encoding="utf-8")
        (best_run / "dalg_manifest.json").write_text("{}", encoding="utf-8")
        summary = {
            "best_by_image": [
                    {
                        "image": f"runs/webui/uploads/{image_path.name}",
                        "label": "LayerForge native",
                        "run_dir": str(best_run),
                        "self_eval_score": 0.7,
                    }
                ]
        }
        (output_root / "frontier_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (scene_root / "best_decomposition.json").write_text(json.dumps(summary["best_by_image"][0]), encoding="utf-8")
        (scene_root / "why_selected.md").write_text("# LayerForge native\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("layerforge.webui.run_frontier_comparison", fake_run_frontier)

    payload = {
        "mode": "frontier",
        "filename": image_path.name,
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "config": "configs/frontier.yaml",
        "device": "cpu",
        "no_parallax": True,
    }
    result = run_webui_job(REPO_ROOT, payload, work_root=work_root)
    assert result["status"] == "ok"
    assert result["mode"] == "frontier"
    assert result["summary_metrics"]["num_layers"] == 3
    assert result["urls"]["summary"]
    assert result["urls"]["selected_best"]
    assert result["urls"]["manifest"]
    assert result["urls"]["metrics"]


def test_webui_run_mode_can_use_frontier_base(tmp_path: Path, monkeypatch) -> None:
    image_path = REPO_ROOT / "examples" / "synth" / "scene_000" / "image.png"
    work_root = tmp_path / "webui"
    winner_dir = tmp_path / "winner_run"
    winner_dir.mkdir(parents=True)
    (winner_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (winner_dir / "metrics.json").write_text(json.dumps({"num_layers": 4, "recompose_psnr": 30.0}), encoding="utf-8")
    (winner_dir / "dalg_manifest.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fail_get_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("native pipeline should not be used when use_frontier_base is enabled")

    def fake_selection(**kwargs):  # type: ignore[no-untyped-def]
        captured["frontier_kwargs"] = kwargs
        return {
            "selected_label": "Qwen raw (3)",
            "run_dir": winner_dir,
            "manifest_path": winner_dir / "manifest.json",
            "metrics_path": winner_dir / "metrics.json",
            "summary_path": tmp_path / "frontier_summary.json",
            "selection": {"label": "Qwen raw (3)", "self_eval_score": 0.62},
        }

    def fake_materialize(selection, target_dir, *, frontier_root=None):  # type: ignore[no-untyped-def]
        captured["selection"] = selection
        captured["target_dir"] = Path(target_dir)
        captured["frontier_root"] = Path(frontier_root) if frontier_root is not None else None
        out = Path(target_dir)
        (out / "debug").mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text("{}", encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps({"num_layers": 4, "recompose_psnr": 30.0}), encoding="utf-8")
        (out / "dalg_manifest.json").write_text("{}", encoding="utf-8")
        return {
            "manifest_path": out / "manifest.json",
            "metrics_path": out / "metrics.json",
            "output_dir": out,
            "selection_path": out / "frontier_selection.json",
        }

    monkeypatch.setattr("layerforge.webui._get_pipeline", fail_get_pipeline)
    monkeypatch.setattr("layerforge.webui.run_single_image_frontier_selection", fake_selection, raising=False)
    monkeypatch.setattr("layerforge.webui.materialize_frontier_selection", fake_materialize, raising=False)

    payload = {
        "mode": "run",
        "filename": image_path.name,
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "config": "configs/frontier.yaml",
        "device": "cpu",
        "use_frontier_base": True,
    }
    result = run_webui_job(REPO_ROOT, payload, work_root=work_root)
    assert result["status"] == "ok"
    assert result["mode"] == "run"
    assert captured["target_dir"].parent.name.startswith("jobs")


def test_webui_extract_mode_can_use_frontier_base(tmp_path: Path, monkeypatch) -> None:
    image_path = REPO_ROOT / "examples" / "synth" / "scene_000" / "image.png"
    work_root = tmp_path / "webui"
    winner_dir = tmp_path / "winner_extract"
    winner_dir.mkdir(parents=True)
    (winner_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (winner_dir / "metrics.json").write_text(json.dumps({"num_layers": 4}), encoding="utf-8")
    (winner_dir / "dalg_manifest.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fail_get_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("native pipeline should not be used when use_frontier_base is enabled")

    def fake_selection(**kwargs):  # type: ignore[no-untyped-def]
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
        (work_root / "jobs" / "extract" / "target_extract").mkdir(parents=True, exist_ok=True)
        return {"selected_target": {"name": "wheel"}}

    monkeypatch.setattr("layerforge.webui._get_pipeline", fail_get_pipeline)
    monkeypatch.setattr("layerforge.webui.run_single_image_frontier_selection", fake_selection, raising=False)
    monkeypatch.setattr("layerforge.webui.export_target_assets", fake_export)

    payload = {
        "mode": "extract",
        "filename": image_path.name,
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "config": "configs/frontier.yaml",
        "device": "cpu",
        "prompt": "wheel",
        "use_frontier_base": True,
    }
    result = run_webui_job(REPO_ROOT, payload, work_root=work_root)
    assert result["status"] == "ok"
    assert result["mode"] == "extract"
    assert captured["path_or_dir"] == winner_dir


def test_webui_transparent_mode_can_use_frontier_base(tmp_path: Path, monkeypatch) -> None:
    image_path = REPO_ROOT / "examples" / "synth" / "scene_000" / "image.png"
    work_root = tmp_path / "webui"
    winner_dir = tmp_path / "winner_transparent"
    winner_dir.mkdir(parents=True)
    (winner_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (winner_dir / "metrics.json").write_text(json.dumps({"num_layers": 4}), encoding="utf-8")
    (winner_dir / "dalg_manifest.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fail_get_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("native pipeline should not be used when use_frontier_base is enabled")

    def fake_selection(**kwargs):  # type: ignore[no-untyped-def]
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
        transparent_dir = work_root / "jobs" / "transparent" / "transparent_extract"
        transparent_dir.mkdir(parents=True, exist_ok=True)
        (transparent_dir / "transparent_metrics.json").write_text(json.dumps({"selected_target": {"name": "glass"}, "recompose_psnr": 33.0}), encoding="utf-8")
        return {"selected_target": {"name": "glass"}}

    monkeypatch.setattr("layerforge.webui._get_pipeline", fail_get_pipeline)
    monkeypatch.setattr("layerforge.webui.run_single_image_frontier_selection", fake_selection, raising=False)
    monkeypatch.setattr("layerforge.webui.export_transparent_assets", fake_export)

    payload = {
        "mode": "transparent",
        "filename": image_path.name,
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "config": "configs/frontier.yaml",
        "device": "cpu",
        "prompt": "glass",
        "use_frontier_base": True,
    }
    result = run_webui_job(REPO_ROOT, payload, work_root=work_root)
    assert result["status"] == "ok"
    assert result["mode"] == "transparent"
    assert captured["path_or_dir"] == winner_dir
