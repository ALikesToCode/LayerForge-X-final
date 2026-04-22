from __future__ import annotations

import numpy as np

from layerforge.coco_benchmark import coco_category_to_group, panoptic_rgb_to_id


def test_panoptic_rgb_to_id_decodes_official_encoding() -> None:
    rgb = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.uint8)
    ids = panoptic_rgb_to_id(rgb)
    assert ids.tolist() == [[1, 256, 65536]]


def test_coco_category_to_group_maps_common_stuff_labels() -> None:
    assert coco_category_to_group("wall-brick", "wall") == "building"
    assert coco_category_to_group("pavement-merged", "floor") == "road"
    assert coco_category_to_group("grass-merged", "plant") == "plant"
    assert coco_category_to_group("river", "water") == "water"
    assert coco_category_to_group("backpack", "accessory") == "object"
