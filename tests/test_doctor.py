from __future__ import annotations

import json

from layerforge.backends import build_backend_registry
from layerforge.cli import main
from layerforge.config import load_config


def test_backend_registry_declares_core_fallbacks() -> None:
    registry = build_backend_registry(load_config("configs/fast.yaml"), device="cpu")

    segmentation = {backend.name: backend for backend in registry.segmentation}
    depth = {backend.name: backend for backend in registry.depth}
    matting = {backend.name: backend for backend in registry.matting}
    amodal = {backend.name: backend for backend in registry.amodal}

    assert segmentation["classical"].available
    assert segmentation["mask2former"].fallback == "classical"
    assert depth["geometric_luminance"].available
    assert depth["depth_pro"].fallback == "geometric_luminance"
    assert matting["heuristic"].available
    assert matting["birefnet"].fallback == "heuristic"
    assert amodal["heuristic"].available
    assert amodal["sameo"].fallback == "heuristic"


def test_doctor_json_reports_readiness_without_requiring_optional_backends(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "doctor",
            "--config",
            "configs/fast.yaml",
            "--device",
            "cpu",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
            "--json",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["paths"]["cache_dir"]["ok"] is True
    assert report["paths"]["output_dir"]["ok"] is True
    assert report["packages"]["psd-tools"]["required"] is False
    assert report["backend_registry"]["segmentation"][0]["name"] == "classical"
    assert any(item["name"] == "birefnet" for item in report["backend_registry"]["matting"])
