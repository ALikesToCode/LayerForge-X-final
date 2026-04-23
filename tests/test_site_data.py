from __future__ import annotations

from pathlib import Path

from layerforge.site_data import build_project_site_payload


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_project_site_payload_exposes_submission_surfaces() -> None:
    payload = build_project_site_payload(REPO_ROOT)
    assert payload["project"]["name"] == "LayerForge-X-final"
    assert payload["project"]["repo_url"].startswith("https://github.com/")
    assert payload["project"]["pages_url"].startswith("https://")
    assert payload["validated"]["pytest"] == "76 passed"
    assert len(payload["comparisons"]) >= 5
    assert any(item["key"] == "frontier_review" for item in payload["figures"])
    assert any(item["label"] == "Final report (DOCX)" for item in payload["docs_links"])
    assert payload["local_lab"]["entrypoint"] == "layerforge webui --open-browser"


def test_build_project_site_payload_lists_all_publishable_markdown_files() -> None:
    payload = build_project_site_payload(REPO_ROOT)
    catalog = {item["source_path"]: item for item in payload["markdown_catalog"]}

    publishable_docs = {
        f"docs/{path.relative_to(REPO_ROOT / 'docs').as_posix()}"
        for path in (REPO_ROOT / "docs").rglob("*.md")
    }

    assert publishable_docs
    assert publishable_docs.issubset(catalog)
    assert catalog["docs/FIGURES.md"]["href"] == "FIGURES.md"
    assert catalog["docs/api/README.md"]["href"] == "api/README.md"


def test_build_project_site_payload_lists_repo_markdown_references() -> None:
    payload = build_project_site_payload(REPO_ROOT)
    catalog = {item["source_path"]: item for item in payload["markdown_catalog"]}

    assert catalog["README.md"]["href"].startswith(payload["project"]["repo_url"])
    assert catalog["report_artifacts/README.md"]["href"].startswith(payload["project"]["repo_url"])
    assert any(item["href"] == "documents.html" for item in payload["docs_links"])
