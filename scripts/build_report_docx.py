#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a report DOCX from the final report pack and current figures.")
    p.add_argument("--output", default="docs/final_report_pack/LayerForge_X_Final_Report_2026_04_22.docx")
    p.add_argument("--full-markdown", default="docs/final_report_pack/LayerForge_X_Final_Report_FULL.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    full_markdown = (ROOT / args.full_markdown).resolve() if not Path(args.full_markdown).is_absolute() else Path(args.full_markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    full_markdown.parent.mkdir(parents=True, exist_ok=True)

    base = ROOT / "docs" / "final_report_pack" / "LayerForge_X_Final_Report_2026_04_22.md"
    sections = [
        ROOT / "docs" / "final_report_pack" / "01_LITERATURE_REVIEW_ADVANCED.md",
        ROOT / "docs" / "final_report_pack" / "02_BENCHMARKING_PROTOCOL.md",
        ROOT / "docs" / "final_report_pack" / "03_NOVELTY_AND_METHOD.md",
        ROOT / "docs" / "final_report_pack" / "04_ABLATIONS_AND_TABLES.md",
    ]

    text = base.read_text(encoding="utf-8")
    replacements = {
        "<!-- include: 01 -->": sections[0].read_text(encoding="utf-8"),
        "<!-- include: 02 -->": sections[1].read_text(encoding="utf-8"),
        "<!-- include: 03 -->": sections[2].read_text(encoding="utf-8"),
        "<!-- include: 04 -->": sections[3].read_text(encoding="utf-8"),
    }
    for marker, content in replacements.items():
        text = text.replace(marker, content)

    full_markdown.write_text(text, encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        stitched = Path(tmpdir) / "stitched_report.md"
        stitched.write_text(text, encoding="utf-8")
        cmd = [
            "pandoc",
            str(stitched),
            "--from",
            "markdown",
            "--resource-path",
            str(ROOT / "docs" / "final_report_pack") + ":" + str(ROOT / "docs"),
            "-o",
            str(output),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

    print(output)
    print(full_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
