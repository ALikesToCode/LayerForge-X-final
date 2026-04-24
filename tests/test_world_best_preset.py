from __future__ import annotations

import json

import layerforge.cli as cli
from layerforge.cli import build_parser, resolve_config_arg
from layerforge.config import load_config


def test_world_best_preset_config_enables_strong_fallback_stack() -> None:
    cfg = load_config("configs/world_best.yaml")

    assert cfg["segmentation"]["method"] == "grounded_sam2"
    assert cfg["segmentation"]["fusion"]["enabled"] is True
    assert cfg["depth"]["method"] == "ensemble"
    assert cfg["depth"]["orientation"] == "auto"
    assert cfg["matting"]["method"] == "auto"
    assert cfg["amodal"]["method"] == "auto"
    assert cfg["inpainting"]["method"] == "auto"
    assert cfg["intrinsics"]["method"] == "auto"


def test_run_doctor_and_benchmark_accept_world_best_preset() -> None:
    parser = build_parser()

    run_args = parser.parse_args(["run", "--preset", "world_best", "input.jpg", "-o", "out"])
    doctor_args = parser.parse_args(["doctor", "--preset", "world_best"])
    benchmark_args = parser.parse_args(["benchmark", "--preset", "world_best", "--dataset-dir", "data", "--output-dir", "out"])

    assert resolve_config_arg(run_args) == "configs/world_best.yaml"
    assert resolve_config_arg(doctor_args) == "configs/world_best.yaml"
    assert resolve_config_arg(benchmark_args) == "configs/world_best.yaml"
    assert run_args.input_pos == "input.jpg"
    assert run_args.output == "out"


def test_doctor_command_uses_world_best_preset_config(tmp_path, capsys, monkeypatch) -> None:
    preset_path = tmp_path / "world_best_test.yaml"
    preset_path.write_text(
        """
segmentation:
  model:
    grounding_dino: local/test-grounding
    sam2: local/test-sam2
matting:
  model: local/test-matting
""",
        encoding="utf-8",
    )
    monkeypatch.setitem(cli.PRESET_CONFIGS, "world_best", str(preset_path))

    exit_code = cli.main(
        [
            "doctor",
            "--preset",
            "world_best",
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
    segmentation = {item["name"]: item for item in report["backend_registry"]["segmentation"]}
    matting = {item["name"]: item for item in report["backend_registry"]["matting"]}
    assert segmentation["grounded_sam2"]["version"] == "local/test-grounding + local/test-sam2"
    assert matting["birefnet"]["version"] == "local/test-matting"
