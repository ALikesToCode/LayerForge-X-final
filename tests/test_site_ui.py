from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_webui_static_surface_is_pages_safe() -> None:
    script_text = (ROOT / "docs" / "assets" / "webui.js").read_text(encoding="utf-8")
    assert "window.__LAYERFORGE_RUNTIME__ === true" in script_text
    assert 'fetchJson("site-data/project_site.json")' in script_text
    assert "Static Pages mode" in script_text
    assert "Local launch" in script_text
    assert "inspector-composite-canvas" in script_text
    assert "drawImage" in script_text

    html_text = (ROOT / "docs" / "webui.html").read_text(encoding="utf-8")
    assert 'class="workflow-strip"' in html_text
    assert 'class="form-stack"' in html_text
    assert 'class="form-cluster"' in html_text
    assert "Frontier best-of selection" in html_text
    assert "configs/world_best.yaml" in html_text
    assert 'id="result-inspector"' in html_text
    assert "Use frontier candidate-bank selection as the strongest base run" in html_text
    assert "use_frontier_base" in html_text

    css_text = (ROOT / "docs" / "assets" / "site.css").read_text(encoding="utf-8")
    assert "overflow-x: clip;" in css_text
    assert "flex-wrap: wrap;" in css_text
    assert ".workflow-step__index" in css_text
    assert ".form-cluster__head" in css_text
    assert ".inspector-layer-card" in css_text
    assert ".inspector-composite" in css_text

    webui_server_text = (ROOT / "src" / "layerforge" / "webui.py").read_text(encoding="utf-8")
    assert "window.__LAYERFORGE_RUNTIME__ = true" in webui_server_text
    assert "_inject_runtime_marker" in webui_server_text
    assert "_collect_layer_inspector" in webui_server_text


def test_documents_page_is_wired_into_static_site() -> None:
    script_text = (ROOT / "docs" / "assets" / "site.js").read_text(encoding="utf-8")
    assert 'page === "documents"' in script_text
    assert "renderMarkdownCatalog" in script_text

    documents_html = (ROOT / "docs" / "documents.html").read_text(encoding="utf-8")
    assert 'data-page="documents"' in documents_html
    assert 'id="markdown-library"' in documents_html
    assert 'href="documents.html"' in documents_html
