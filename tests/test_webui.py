from __future__ import annotations

import base64
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
