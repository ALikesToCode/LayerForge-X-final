from __future__ import annotations

import numpy as np

from layerforge.compose import rgba_from_rgb_alpha
from layerforge.graph import merge_compatible_layers
from layerforge.types import Layer


def make_layer(
    *,
    name: str,
    label: str,
    group: str,
    rank: int,
    mask: np.ndarray,
    rgb: tuple[int, int, int],
    depth: float,
) -> Layer:
    alpha = mask.astype(np.float32)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask] = np.array(rgb, dtype=np.uint8)
    rgba = rgba_from_rgb_alpha(color, alpha)
    return Layer(
        id=rank,
        name=name,
        label=label,
        group=group,
        rank=rank,
        depth_median=depth,
        depth_p10=depth,
        depth_p90=depth,
        area=int(mask.sum()),
        bbox=(0, 0, mask.shape[1], mask.shape[0]),
        alpha=alpha,
        rgba=rgba,
        albedo_rgba=rgba,
        shading_rgba=rgba,
        visible_mask=mask.copy(),
        amodal_mask=mask.copy(),
        source_segment_ids=[rank],
    )


def test_merge_compatible_layers_merges_background_planes() -> None:
    mask_a = np.zeros((16, 16), dtype=bool)
    mask_b = np.zeros((16, 16), dtype=bool)
    mask_a[2:8, 2:6] = True
    mask_b[3:9, 9:13] = True
    layers = [
        make_layer(name="000_building_window_plane_0", label="window plane 0", group="building", rank=0, mask=mask_a, rgb=(120, 120, 130), depth=0.40),
        make_layer(name="001_building_window_plane_1", label="window plane 1", group="building", rank=1, mask=mask_b, rgb=(122, 122, 132), depth=0.43),
    ]
    merged = merge_compatible_layers(layers, {"merge_enabled": True, "merge_depth_threshold": 0.05, "merge_color_threshold": 0.20, "merge_spatial_gap_px": 12})
    assert len(merged) == 1
    assert merged[0].metadata["merged_count"] == 2


def test_merge_compatible_layers_keeps_distinct_things_apart() -> None:
    mask_a = np.zeros((16, 16), dtype=bool)
    mask_b = np.zeros((16, 16), dtype=bool)
    mask_a[2:8, 2:6] = True
    mask_b[3:9, 9:13] = True
    layers = [
        make_layer(name="000_person_person", label="person", group="person", rank=0, mask=mask_a, rgb=(220, 80, 80), depth=0.30),
        make_layer(name="001_vehicle_car", label="car", group="vehicle", rank=1, mask=mask_b, rgb=(80, 80, 220), depth=0.32),
    ]
    merged = merge_compatible_layers(layers, {"merge_enabled": True, "merge_depth_threshold": 0.05, "merge_color_threshold": 0.20, "merge_spatial_gap_px": 12})
    assert len(merged) == 2


def test_merge_compatible_layers_does_not_skip_across_intervening_bucket() -> None:
    mask_a = np.zeros((20, 20), dtype=bool)
    mask_b = np.zeros((20, 20), dtype=bool)
    mask_c = np.zeros((20, 20), dtype=bool)
    mask_a[2:8, 2:6] = True
    mask_b[4:12, 8:12] = True
    mask_c[3:9, 14:18] = True
    layers = [
        make_layer(name="000_building_wall_plane_0", label="wall plane 0", group="building", rank=0, mask=mask_a, rgb=(120, 120, 130), depth=0.40),
        make_layer(name="001_person_person", label="person", group="person", rank=1, mask=mask_b, rgb=(220, 80, 80), depth=0.41),
        make_layer(name="002_building_wall_plane_1", label="wall plane 1", group="building", rank=2, mask=mask_c, rgb=(121, 121, 131), depth=0.42),
    ]
    merged = merge_compatible_layers(layers, {"merge_enabled": True, "merge_depth_threshold": 0.05, "merge_color_threshold": 0.20, "merge_spatial_gap_px": 20})
    assert len(merged) == 3
    assert [layer.label for layer in merged] == ["wall plane 0", "person", "wall plane 1"]
