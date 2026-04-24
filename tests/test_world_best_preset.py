from __future__ import annotations

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
