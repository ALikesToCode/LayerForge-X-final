from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from layerforge.cli import build_frontier_base_kwargs
from layerforge.cli import default_frontier_output_root
from layerforge.cli import main


def test_frontier_base_promotes_fast_config_to_best_score(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "input": str(tmp_path / "sample.png"),
            "config": "configs/fast.yaml",
            "segmenter": None,
            "depth": None,
            "frontier_skip_native": False,
            "frontier_peeling_config": "configs/recursive_peeling.yaml",
            "frontier_peeling_segmenter": None,
            "frontier_peeling_depth": None,
            "frontier_skip_peeling": False,
            "frontier_qwen_model": "Qwen/Qwen-Image-Layered",
            "frontier_qwen_layers": "3,4,6,8",
            "frontier_qwen_resolution": 640,
            "frontier_qwen_steps": 10,
            "frontier_qwen_device": "cuda",
            "frontier_qwen_dtype": "bfloat16",
            "frontier_qwen_offload": "sequential",
            "frontier_qwen_hybrid_modes": "preserve,reorder",
            "frontier_qwen_merge_external_layers": False,
        },
    )()

    kwargs = build_frontier_base_kwargs(args, output_root=tmp_path / "frontier")
    assert kwargs["native_config"] == "configs/best_score.yaml"


def test_default_frontier_output_root_is_sibling_of_output() -> None:
    assert default_frontier_output_root("runs/sample") == Path("runs/sample_frontier")
    assert default_frontier_output_root(".") == Path("output_frontier")


def test_run_frontier_materializes_selected_run(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (120, 140, 180)).save(image_path)

    winner_dir = tmp_path / "winner_run"
    winner_dir.mkdir(parents=True)
    (winner_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (winner_dir / "metrics.json").write_text(json.dumps({"num_layers": 5}), encoding="utf-8")

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
            "selection": {"label": "Qwen + graph preserve (3)", "self_eval_score": 0.61},
        }

    def fake_materialize(selection, target_dir, *, frontier_root=None):  # type: ignore[no-untyped-def]
        captured["selection"] = selection
        captured["target_dir"] = Path(target_dir)
        captured["frontier_root"] = Path(frontier_root) if frontier_root is not None else None
        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / "manifest.json"
        metrics = out / "metrics.json"
        manifest.write_text("{}", encoding="utf-8")
        metrics.write_text(json.dumps({"num_layers": 5}), encoding="utf-8")
        return {
            "manifest_path": manifest,
            "metrics_path": metrics,
            "output_dir": out,
            "selection_path": out / "frontier_selection.json",
        }

    monkeypatch.setattr("layerforge.cli.LayerForgePipeline", fail_pipeline)
    monkeypatch.setattr("layerforge.cli.run_single_image_frontier_selection", fake_frontier, raising=False)
    monkeypatch.setattr("layerforge.cli.materialize_frontier_selection", fake_materialize, raising=False)

    exit_code = main(
        [
            "run",
            "--input",
            str(image_path),
            "--output",
            str(tmp_path / "run_output"),
            "--config",
            "configs/frontier.yaml",
            "--frontier",
        ]
    )

    assert exit_code == 0
    assert captured["target_dir"] == tmp_path / "run_output"
    assert captured["frontier_root"] == tmp_path / "run_output_frontier"


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


def test_extract_geometry_only_reruns_prompted_base_when_selection_misses_cue(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (120, 140, 180)).save(image_path)

    run_calls: list[dict[str, object]] = []
    export_calls: list[dict[str, object]] = []

    class FakeOutputs:
        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir
            self.manifest_path = output_dir / "manifest.json"
            self.metrics_path = output_dir / "metrics.json"
            self.ordered_layer_paths = []

    class FakePipeline:
        def __init__(self, config, device="auto") -> None:  # type: ignore[no-untyped-def]
            self.config = config
            self.device = device

        def run(self, input_path, output_dir, **kwargs):  # type: ignore[no-untyped-def]
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text("{}", encoding="utf-8")
            (out / "metrics.json").write_text("{}", encoding="utf-8")
            run_calls.append({"output_dir": out, "prompts": kwargs.get("prompts"), "prompt_source": kwargs.get("prompt_source")})
            return FakeOutputs(out)

    def fake_export(path_or_dir, **kwargs):  # type: ignore[no-untyped-def]
        export_calls.append({"path_or_dir": Path(path_or_dir), "kwargs": kwargs})
        matched = Path(path_or_dir).name == "geometry_prompted_base"
        return {
            "selected_target": {"name": "building" if matched else "wrong_region"},
            "resolved_prompt": "building",
            "geometry_match": {"matches": matched},
        }

    monkeypatch.setattr("layerforge.cli.LayerForgePipeline", FakePipeline)
    monkeypatch.setattr("layerforge.cli.export_target_assets", fake_export)

    exit_code = main(
        [
            "extract",
            "--input",
            str(image_path),
            "--output",
            str(tmp_path / "extract_output"),
            "--config",
            "configs/fast.yaml",
            "--point",
            "10,12",
        ]
    )

    assert exit_code == 0
    assert len(run_calls) == 2
    assert run_calls[1]["output_dir"] == tmp_path / "extract_output" / "geometry_prompted_base"
    assert run_calls[1]["prompts"] == ["building"]
    assert run_calls[1]["prompt_source"] == "manual"
    assert len(export_calls) == 2
    assert export_calls[1]["kwargs"]["prompt"] == "building"


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
    assert captured["export_kwargs"]["device"] == "auto"
