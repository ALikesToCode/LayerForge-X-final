from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_brand_assets_exist() -> None:
    required = [
        "docs/assets/brand/layerforge-logo-source.png",
        "docs/assets/brand/layerforge-logo.png",
        "docs/assets/brand/layerforge-mark.png",
        "docs/assets/brand/favicon-32x32.png",
        "docs/assets/brand/favicon-16x16.png",
        "docs/assets/brand/apple-touch-icon.png",
        "docs/assets/brand/favicon.ico",
    ]
    for rel_path in required:
        assert (ROOT / rel_path).exists(), rel_path


def test_site_pages_reference_brand_assets() -> None:
    pages = [
        ROOT / "docs" / "index.html",
        ROOT / "docs" / "about.html",
        ROOT / "docs" / "webui.html",
    ]
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert "assets/brand/favicon-32x32.png" in text
        assert "assets/brand/favicon-16x16.png" in text
        assert "assets/brand/favicon.ico" in text
        assert "assets/brand/apple-touch-icon.png" in text
        assert "assets/brand/layerforge-mark.png" in text
