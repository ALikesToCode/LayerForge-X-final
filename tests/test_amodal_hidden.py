from __future__ import annotations

import sys

import numpy as np

from layerforge.graph import build_layers, resolve_amodal_mask
from layerforge.segment import make_segment


def test_amodal_backend_missing_falls_back_to_heuristic() -> None:
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:16, 8:14] = True

    amodal, metadata = resolve_amodal_mask(mask, {"method": "sameo"}, expand_px=8)

    assert amodal is not None
    assert int(amodal.sum()) >= int(mask.sum())
    assert metadata["fallback_used"] is True
    assert metadata["backend"] == "heuristic"


def test_external_amodal_backend_reads_configured_command(tmp_path) -> None:
    script = tmp_path / "amodal_backend.py"
    script.write_text(
        """
import sys
import numpy as np
from PIL import Image

visible = np.asarray(Image.open(sys.argv[1]).convert("L")) > 127
amodal = visible.copy()
amodal[:, 2:] |= visible[:, :-2]
Image.fromarray(amodal.astype(np.uint8) * 255, mode="L").save(sys.argv[2])
""",
        encoding="utf-8",
    )
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 4:9] = True
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)

    amodal, metadata = resolve_amodal_mask(
        mask,
        {"method": "external", "external_command": f"{sys.executable} {script} {{visible_mask}} {{amodal_mask}}"},
        expand_px=0,
        rgb=rgb,
    )

    assert amodal is not None
    assert int(amodal.sum()) > int(mask.sum())
    assert metadata["backend"] == "external"
    assert metadata["backend_used"] is True
    assert metadata["fallback_used"] is False


def test_build_layers_saves_hidden_only_completion_metadata() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    rgb[8:20, 8:16] = [220, 40, 40]
    visible = np.zeros((32, 32), dtype=bool)
    visible[8:20, 8:16] = True
    depth = np.full((32, 32), 0.4, dtype=np.float32)
    segment = make_segment(0, "person", visible, 1.0, "synthetic")

    layers, _ = build_layers(
        rgb,
        [segment],
        depth,
        rgb,
        rgb,
        {"min_layer_area_ratio": 0.001, "split_stuff_depth_bins": 1, "max_layers": 4, "amodal_enabled": True, "amodal_expand_px": 10, "merge_enabled": False},
        {"method": "heuristic", "alpha_band_px": 3, "preserve_depth_edges": False},
        device="cpu",
        amodal_cfg={"method": "heuristic"},
    )

    layer = layers[0]
    assert layer.hidden_mask is not None
    assert layer.completed_rgba is not None
    assert int(layer.hidden_mask.sum()) > 0
    assert np.array_equal(layer.completed_rgba[layer.visible_mask, :3], layer.rgba[layer.visible_mask, :3])
    assert layer.metadata["hidden_area_ratio"] > 0.0
    assert 0.0 <= layer.metadata["edge_continuity_score"] <= 1.0
