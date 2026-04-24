from __future__ import annotations

import sys

import numpy as np

from layerforge.inpaint import complete_hidden_layer, inpaint_background, inpainting_quality_metrics


def test_lama_mode_falls_back_to_opencv_when_dependency_missing(monkeypatch) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 120
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    def fail_loader():
        raise ModuleNotFoundError("simple_lama_inpainting")

    monkeypatch.setattr("layerforge.inpaint._load_simple_lama", fail_loader)

    out, used_mask, method = inpaint_background(rgb, mask, {"method": "lama", "radius": 3})

    assert out.shape == rgb.shape
    assert used_mask.shape == mask.shape
    assert method == "opencv_telea_fallback"


def test_auto_and_diffusion_inpainting_use_cpu_safe_fallback() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 160
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    auto_out, _, auto_method = inpaint_background(rgb, mask, {"method": "auto", "radius": 3})
    diffusion_out, _, diffusion_method = inpaint_background(rgb, mask, {"method": "diffusion", "radius": 3, "prompt": "red cloth"})

    assert auto_out.shape == rgb.shape
    assert diffusion_out.shape == rgb.shape
    assert auto_method == "opencv_telea_auto"
    assert diffusion_method == "opencv_telea_fallback"


def test_external_inpainting_uses_configured_command(tmp_path) -> None:
    script = tmp_path / "inpaint_backend.py"
    script.write_text(
        """
import sys
import numpy as np
from PIL import Image

image = np.array(Image.open(sys.argv[1]).convert("RGB"))
mask = np.asarray(Image.open(sys.argv[2]).convert("L")) > 127
image[mask] = [10, 20, 30]
Image.fromarray(image, mode="RGB").save(sys.argv[3])
""",
        encoding="utf-8",
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    out, _, method = inpaint_background(
        rgb,
        mask,
        {"method": "external", "external_command": f"{sys.executable} {script} {{image}} {{mask}} {{output}}"},
    )

    assert method == "external"
    assert np.array_equal(out[mask][0], np.array([10, 20, 30], dtype=np.uint8))


def test_complete_hidden_layer_preserves_visible_pixels_and_reports_metrics() -> None:
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    rgb[2:8, 2:6] = [220, 40, 30]
    alpha = np.zeros((12, 12), dtype=np.float32)
    visible = np.zeros((12, 12), dtype=bool)
    hidden = np.zeros((12, 12), dtype=bool)
    visible[2:8, 2:6] = True
    hidden[2:8, 6:9] = True
    alpha[visible] = 1.0

    completed, metadata = complete_hidden_layer(rgb, alpha, visible, hidden, {"method": "telea", "radius": 3})

    assert completed.shape == (12, 12, 4)
    assert np.array_equal(completed[visible, :3], rgb[visible])
    assert int(completed[..., 3][hidden].mean()) == 255
    assert metadata["hidden_pixels"] == int(hidden.sum())
    assert 0.0 <= metadata["boundary_consistency"] <= 1.0


def test_inpainting_quality_metrics_are_identity_without_mask() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    metrics = inpainting_quality_metrics(rgb, rgb.copy(), np.zeros((4, 4), dtype=bool))
    assert metrics["boundary_consistency"] == 1.0
    assert metrics["recomposition_residual_outside_mask"] == 0.0
