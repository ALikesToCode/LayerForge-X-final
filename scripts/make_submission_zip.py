#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]

INCLUDE_TOP_LEVEL = {
    "AGENTS.md": False,
    "Makefile": True,
    "PROJECT_MANIFEST.json": True,
    "README.md": True,
    "configs": True,
    "docs": True,
    "examples": True,
    "models": True,
    "pyproject.toml": True,
    "pytest.ini": True,
    "report_artifacts": True,
    "requirements-models.txt": True,
    "requirements.txt": True,
    "schemas": True,
    "scripts": True,
    "src": True,
    "tests": True,
}

EXCLUDED_DIR_NAMES = {
    ".codex",
    ".git",
    ".pytest_cache",
    ".venv",
    "data",
    "results",
    "runs",
}

EXCLUDED_PATH_PREFIXES = (
    "docs/internal/",
    "docs/final_report_pack/sources/",
)

EXCLUDED_SUFFIXES = {
    ".pyc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a submission ZIP without heavyweight local artifacts or internal notes.")
    parser.add_argument(
        "--output",
        default="/tmp/LayerForge-X-final-submission.zip",
        help="Output ZIP path.",
    )
    return parser.parse_args()


def should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    if not parts:
        return False
    top = parts[0]
    if top in EXCLUDED_DIR_NAMES:
        return False
    if top not in INCLUDE_TOP_LEVEL:
        return False
    if any(rel_path.as_posix().startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
        return False
    if rel_path.suffix in EXCLUDED_SUFFIXES:
        return False
    if "__pycache__" in parts:
        return False
    return True


def iter_submission_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(ROOT)
        if should_include(rel_path):
            files.append(rel_path)
    return sorted(files)


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = iter_submission_files()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as zf:
        for rel_path in files:
            zf.write(ROOT / rel_path, arcname=rel_path.as_posix())
    print(output)
    print(f"files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
