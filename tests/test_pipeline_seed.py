from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import layerforge.pipeline as pipeline_mod
from layerforge.pipeline import LayerForgePipeline


def make_synthetic_dataset(root: Path, *, count: int, seed: int, width: int = 160, height: int = 120) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/make_synthetic_dataset.py",
            "--output",
            str(root),
            "--count",
            str(count),
            "--seed",
            str(seed),
            "--width",
            str(width),
            "--height",
            str(height),
        ],
        check=True,
    )


def test_pipeline_run_reseeds_each_invocation(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "synthetic"
    make_synthetic_dataset(dataset_dir, count=1, seed=3)
    image_path = dataset_dir / "scene_000" / "image.png"

    calls: list[int] = []
    real_seed_everything = pipeline_mod.seed_everything

    def record_seed(seed: int) -> None:
        calls.append(seed)
        real_seed_everything(seed)

    monkeypatch.setattr(pipeline_mod, "seed_everything", record_seed)

    pipe = LayerForgePipeline("configs/fast.yaml")
    pipe.run(image_path, tmp_path / "run_a", segmenter="classical", depth_method="geometric_luminance", save_parallax=False)
    pipe.run(image_path, tmp_path / "run_b", segmenter="classical", depth_method="geometric_luminance", save_parallax=False)

    assert calls[-2:] == [7, 7]
