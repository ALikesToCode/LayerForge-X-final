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
