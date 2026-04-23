from __future__ import annotations

from pathlib import Path

from layerforge.site_data import build_project_site_payload


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_project_site_payload_exposes_submission_surfaces() -> None:
    payload = build_project_site_payload(REPO_ROOT)
    assert payload["project"]["name"] == "LayerForge-X-final"
    assert payload["project"]["repo_url"].startswith("https://github.com/")
    assert payload["project"]["pages_url"].startswith("https://")
    assert payload["validated"]["pytest"] == "71 passed"
    assert len(payload["comparisons"]) >= 5
    assert any(item["key"] == "frontier_review" for item in payload["figures"])
    assert any(item["label"] == "Final report (DOCX)" for item in payload["docs_links"])
    assert payload["local_lab"]["entrypoint"] == "layerforge webui --open-browser"
