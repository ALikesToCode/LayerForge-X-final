from __future__ import annotations

import numpy as np
from PIL import Image

from layerforge.ade20k_benchmark import ade20k_category_to_group, load_ade20k_category_names, load_ade20k_ground_truth_group_masks


def test_load_ade20k_category_names_parses_tab_file(tmp_path):
    path = tmp_path / "objectInfo150.txt"
    path.write_text(
        "Idx\tRatio\tTrain\tVal\tName\n"
        "1\t0.1\t10\t2\twall\n"
        "2\t0.1\t10\t2\tceiling\n"
        "3\t0.1\t10\t2\tperson, individual\n",
        encoding="utf-8",
    )
    categories = load_ade20k_category_names(path)
    assert categories == {1: "wall", 2: "ceiling", 3: "person"}


def test_ade20k_category_to_group_maps_scene_labels():
    assert ade20k_category_to_group("ceiling") == "building"
    assert ade20k_category_to_group("floor") == "ground"
    assert ade20k_category_to_group("person") == "person"


def test_load_ade20k_ground_truth_group_masks_merges_by_group(tmp_path):
    ann = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    path = tmp_path / "label.png"
    Image.fromarray(ann).save(path)
    masks = load_ade20k_ground_truth_group_masks(path, {1: "wall", 2: "floor", 3: "person"})
    assert bool(masks["building"][0, 0])
    assert bool(masks["ground"][0, 1])
    assert bool(masks["person"][1, 0])
