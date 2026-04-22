from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from layerforge.pipeline import LayerForgePipeline


def make_synthetic_dataset(root: Path, *, count: int, seed: int, width: int = 320, height: int = 224) -> None:
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


@pytest.mark.slow
def test_train_synthetic_order_ranker(tmp_path) -> None:
    from layerforge.ranker import load_ranker, train_synthetic_order_ranker

    dataset_dir = tmp_path / "synthetic"
    model_path = tmp_path / "order_ranker.json"

    make_synthetic_dataset(dataset_dir, count=4, seed=7)
    report = train_synthetic_order_ranker(
        dataset_dir=dataset_dir,
        output_path=model_path,
        config_path="configs/fast.yaml",
        segmenter="classical",
        depth="geometric_luminance",
    )

    assert model_path.exists()
    assert report["scene_count"] == 4
    assert report["pair_count"] > 0
    assert report["train_accuracy"] >= 0.70

    model = load_ranker(model_path)
    assert model.feature_names
    assert len(model.weights) == len(model.feature_names)


@pytest.mark.slow
def test_pipeline_accepts_learned_ordering(tmp_path) -> None:
    from layerforge.ranker import train_synthetic_order_ranker

    dataset_dir = tmp_path / "synthetic"
    model_path = tmp_path / "order_ranker.json"
    output_dir = tmp_path / "run"

    make_synthetic_dataset(dataset_dir, count=5, seed=11)
    train_synthetic_order_ranker(
        dataset_dir=dataset_dir,
        output_path=model_path,
        config_path="configs/fast.yaml",
        segmenter="classical",
        depth="geometric_luminance",
    )

    pipe = LayerForgePipeline("configs/fast.yaml")
    outputs = pipe.run(
        dataset_dir / "scene_004" / "image.png",
        output_dir,
        depth_method="geometric_luminance",
        segmenter="classical",
        ordering_method="learned",
        ranker_model_path=model_path,
        save_parallax=False,
    )

    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    assert metrics["ordering_method"] == "learned"
    assert len(outputs.ordered_layer_paths) > 0
