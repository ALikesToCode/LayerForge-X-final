#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
DEFAULT_OUTPUT = ROOT / "_site"
sys.path.insert(0, str(ROOT / "src"))

from layerforge.site_data import write_project_site_payload  # noqa: E402


def _validate_output_dir(output: Path) -> Path:
    resolved = output.resolve()
    forbidden = {ROOT.resolve(), DOCS_ROOT.resolve(), Path("/").resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing to write Pages artifact to protected path: {resolved}")
    return resolved


def build_pages_artifact(output: Path = DEFAULT_OUTPUT) -> Path:
    output = _validate_output_dir(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_root = output / "markdown-source"
    for src in DOCS_ROOT.rglob("*"):
        if not src.is_file():
            continue

        rel = src.relative_to(DOCS_ROOT)
        if src.suffix.lower() == ".md":
            target = source_root / f"{rel.as_posix()}.txt"
        else:
            target = output / rel

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    write_project_site_payload(ROOT, output / "site-data" / "project_site.json")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the static GitHub Pages artifact without publishing raw .md files at website paths.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Artifact output directory")
    args = parser.parse_args()

    output = build_pages_artifact(args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
