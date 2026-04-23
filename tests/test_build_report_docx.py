from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_report_docx.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_report_docx", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_display_path_handles_repo_and_non_repo_paths() -> None:
    module = _load_module()
    in_repo = ROOT / "docs" / "README.md"
    out_of_repo = Path("/tmp/layerforge-report.docx")
    assert module._display_path(in_repo) == "docs/README.md"
    assert module._display_path(out_of_repo) == str(out_of_repo)
