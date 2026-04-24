from __future__ import annotations

from pathlib import Path

from scripts.build_pages_artifact import build_pages_artifact


ROOT = Path(__file__).resolve().parents[1]


def test_build_pages_artifact_routes_markdown_through_reader(tmp_path: Path) -> None:
    output = build_pages_artifact(tmp_path / "site")

    assert (output / "index.html").exists()
    assert (output / "reader.html").exists()
    assert (output / "404.html").exists()
    assert (output / "assets" / "site.js").exists()
    assert (output / "api" / "openapi.yaml").exists()

    assert not (output / "FIGURES.md").exists()
    assert not (output / "api" / "README.md").exists()
    assert (output / "markdown-source" / "FIGURES.md.txt").exists()
    assert (output / "markdown-source" / "api" / "README.md.txt").exists()

    site_data = (output / "site-data" / "project_site.json").read_text(encoding="utf-8")
    assert "reader.html?path=FIGURES.md" in site_data
    assert "markdown-source/FIGURES.md.txt" in site_data
