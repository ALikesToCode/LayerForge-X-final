from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module():
    script_path = ROOT / "scripts" / "run_extract_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_extract_benchmark", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_benchmark_reuses_prompted_and_unguided_runs(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    dataset_dir = tmp_path / "dataset"
    scene_dir = dataset_dir / "scene_000"
    scene_dir.mkdir(parents=True)
    Image.new("RGB", (32, 24), (220, 220, 220)).save(scene_dir / "image.png")
    layer_rgba = np.zeros((24, 32, 4), dtype=np.uint8)
    layer_rgba[4:16, 8:20, :3] = np.array([20, 120, 240], dtype=np.uint8)
    layer_rgba[4:16, 8:20, 3] = 255
    Image.fromarray(layer_rgba, mode="RGBA").save(scene_dir / "target.png")
    (scene_dir / "ground_truth.json").write_text(
        json.dumps({"layers_near_to_far": [{"name": "blue_bike", "path": "target.png"}]}),
        encoding="utf-8",
    )

    run_calls: list[dict] = []

    class FakeOutputs:
        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir

    class FakePipeline:
        def __init__(self, config: str, device: str = "auto") -> None:
            self.config = config
            self.device = device

        def run(self, image_path: Path, output_dir: Path, **kwargs) -> FakeOutputs:
            run_calls.append(
                {
                    "image_path": str(image_path),
                    "output_dir": str(output_dir),
                    "prompts": kwargs.get("prompts"),
                    "prompt_source": kwargs.get("prompt_source"),
                }
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            return FakeOutputs(output_dir)

    def fake_export_target_assets(run_dir: Path, output_dir: Path, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(layer_rgba[..., 3], mode="L").save(output_dir / "target_alpha.png")
        return {"selected_target": {"name": "blue_bike"}}

    monkeypatch.setattr(module, "LayerForgePipeline", FakePipeline)
    monkeypatch.setattr(module, "export_target_assets", fake_export_target_assets)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(tmp_path / "out"),
                "config": "configs/fast.yaml",
                "segmenter": "classical",
                "depth": "geometric_luminance",
                "device": "cpu",
                "max_scenes": 1,
            },
        )(),
    )

    rc = module.main()
    assert rc == 0
    assert len(run_calls) == 2
    assert {tuple(call["prompts"]) if call["prompts"] else None for call in run_calls} == {("blue bike",), None}

    summary = json.loads((tmp_path / "out" / "extract_benchmark_summary.json").read_text(encoding="utf-8"))
    assert len(summary["rows"]) == 5
    assert {row["query_type"] for row in summary["rows"]} == {"text", "point", "box", "text_point", "text_box"}


def test_extract_benchmark_uses_semantic_identity_from_target_metadata(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    dataset_dir = tmp_path / "dataset"
    scene_dir = dataset_dir / "scene_000"
    scene_dir.mkdir(parents=True)
    Image.new("RGB", (32, 24), (220, 220, 220)).save(scene_dir / "image.png")
    layer_rgba = np.zeros((24, 32, 4), dtype=np.uint8)
    layer_rgba[4:16, 8:20, :3] = np.array([20, 120, 240], dtype=np.uint8)
    layer_rgba[4:16, 8:20, 3] = 255
    Image.fromarray(layer_rgba, mode="RGBA").save(scene_dir / "target.png")
    (scene_dir / "ground_truth.json").write_text(
        json.dumps({"layers_near_to_far": [{"name": "mid_building", "path": "target.png"}]}),
        encoding="utf-8",
    )

    class FakeOutputs:
        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir

    class FakePipeline:
        def __init__(self, config: str, device: str = "auto") -> None:
            self.config = config
            self.device = device

        def run(self, image_path: Path, output_dir: Path, **kwargs) -> FakeOutputs:
            output_dir.mkdir(parents=True, exist_ok=True)
            return FakeOutputs(output_dir)

    def fake_export_target_assets(run_dir: Path, output_dir: Path, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(layer_rgba[..., 3], mode="L").save(output_dir / "target_alpha.png")
        return {
            "selected_target": {
                "name": "000_object_region",
                "label": "object region",
                "semantic_name": "mid_building",
                "semantic_label": "mid building",
            },
            "resolved_prompt": "mid building",
        }

    monkeypatch.setattr(module, "LayerForgePipeline", FakePipeline)
    monkeypatch.setattr(module, "export_target_assets", fake_export_target_assets)

    row = module.evaluate_query(
        tmp_path / "run",
        scene_dir,
        "point",
        {"point": (12, 10)},
        "mid_building",
        layer_rgba[..., 3].astype(np.float32) / 255.0,
        tmp_path / "out",
    )

    assert row["selected_name"] == "000_object_region"
    assert row["selected_semantic_name"] == "mid_building"
    assert row["target_hit"] is True
