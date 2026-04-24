from __future__ import annotations

import numpy as np

from layerforge.intrinsics import decompose_intrinsics, identity_decompose, intrinsic_residual


def test_identity_intrinsics_recompose_known_synthetic_image() -> None:
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(12, dtype=np.uint8)[None, :] * 10
    rgb[..., 1] = 120
    albedo, shading = identity_decompose(rgb)

    assert intrinsic_residual(rgb, albedo, shading) == 0.0


def test_intrinsics_methods_fallback_without_optional_models() -> None:
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)

    for method in ["none", "auto", "ordinal", "intrinsic_model", "external"]:
        albedo, shading, used = decompose_intrinsics(rgb, {"method": method, "sigma": 4.0, "external_command": ""})
        assert albedo.shape == rgb.shape
        assert shading.shape == rgb.shape
        assert used in {"none", "retinex_auto", "retinex_fallback"}


def test_intrinsic_residual_respects_mask() -> None:
    rgb = np.full((6, 6, 3), 200, dtype=np.uint8)
    albedo = rgb.copy()
    shading = np.full_like(rgb, 255)
    mask = np.zeros((6, 6), dtype=bool)
    mask[:3, :3] = True
    albedo[~mask] = 0

    assert intrinsic_residual(rgb, albedo, shading, mask) == 0.0
    assert intrinsic_residual(rgb, albedo, shading, ~mask) > 0.5
