from __future__ import annotations

import json

import numpy as np

from layerforge.segment import fuse_proposals, make_segment


def _mask(shape: tuple[int, int], y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[y0:y1, x0:x1] = True
    return out


def test_fuse_proposals_merges_duplicate_masks_and_writes_diagnostics(tmp_path) -> None:
    shape = (32, 32)
    car_a = make_segment(0, "car", _mask(shape, 6, 6, 18, 20), 0.91, "sam")
    car_b = make_segment(1, "vehicle", _mask(shape, 6, 7, 18, 21), 0.86, "detector")

    fused, diagnostics = fuse_proposals([car_a, car_b], shape=shape, diagnostics_path=tmp_path / "fusion.json")

    assert len(fused) == 1
    assert fused[0].group == "vehicle"
    assert diagnostics["merges"][0]["reason"] == "iou"
    assert json.loads((tmp_path / "fusion.json").read_text(encoding="utf-8"))["output_count"] == 1


def test_fuse_proposals_preserves_nested_semantically_distinct_masks() -> None:
    shape = (40, 40)
    person = make_segment(0, "person", _mask(shape, 4, 10, 34, 28), 0.92, "sam")
    backpack = make_segment(1, "backpack object", _mask(shape, 16, 14, 29, 24), 0.80, "open-vocab")

    fused, diagnostics = fuse_proposals([person, backpack], shape=shape)

    assert len(fused) == 2
    assert {seg.group for seg in fused} == {"person", "object"}
    assert diagnostics["merges"] == []


def test_fuse_proposals_splits_stuff_object_conflicts() -> None:
    shape = (36, 36)
    road = make_segment(0, "road background stuff", _mask(shape, 0, 0, 36, 36), 0.7, "panoptic")
    car = make_segment(1, "car", _mask(shape, 10, 10, 24, 26), 0.95, "sam")

    fused, diagnostics = fuse_proposals([road, car], shape=shape, stuff_overlap_threshold=0.01)

    road_out = next(seg for seg in fused if seg.group == "road")
    car_out = next(seg for seg in fused if seg.group == "vehicle")
    assert not np.any(road_out.mask & car_out.mask)
    assert diagnostics["stuff_object_splits"][0]["label"] == "road background stuff"
    assert diagnostics["semantic_families"]["vehicles"] == 1
    assert diagnostics["semantic_families"]["background/stuff"] == 1


def test_fuse_proposals_merges_contained_semantic_duplicates() -> None:
    shape = (40, 40)
    animal = make_segment(0, "dog", _mask(shape, 6, 6, 32, 34), 0.82, "panoptic")
    animal_crop = make_segment(1, "animal", _mask(shape, 10, 10, 25, 26), 0.88, "sam")

    fused, diagnostics = fuse_proposals([animal, animal_crop], shape=shape, containment_threshold=0.92)

    assert len(fused) == 1
    assert fused[0].group == "animal"
    assert diagnostics["merges"][0]["reason"] == "containment"
