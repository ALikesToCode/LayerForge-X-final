from __future__ import annotations

from pathlib import Path

from layerforge.diode_benchmark import enumerate_diode_depth_samples, resolve_diode_split_root, scene_type_from_path


def test_resolve_diode_split_root_handles_parent_or_split_dir(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "val" / "indoors").mkdir(parents=True)
    assert resolve_diode_split_root(root) == root / "val"
    assert resolve_diode_split_root(root / "val") == root / "val"


def test_scene_type_from_path_extracts_indoor_and_outdoor() -> None:
    assert scene_type_from_path("val/indoors/scene_00001/sample_depth.npy") == "indoors"
    assert scene_type_from_path("val/outdoor/scene_00001/sample_depth.npy") == "outdoor"


def test_enumerate_diode_depth_samples_discovers_triplets(tmp_path: Path) -> None:
    split_root = tmp_path / "val" / "indoors" / "scene_00001" / "scan_00001"
    split_root.mkdir(parents=True)
    (split_root / "frame_00001.png").write_bytes(b"png")
    (split_root / "frame_00001_depth.npy").write_bytes(b"npy")
    (split_root / "frame_00001_depth_mask.npy").write_bytes(b"mask")
    samples = enumerate_diode_depth_samples(tmp_path / "val")
    assert len(samples) == 1
    assert Path(samples[0]["image"]).name == "frame_00001.png"
    assert Path(samples[0]["depth"]).name == "frame_00001_depth.npy"
    assert Path(samples[0]["mask"]).name == "frame_00001_depth_mask.npy"
    assert samples[0]["scene_type"] == "indoors"
