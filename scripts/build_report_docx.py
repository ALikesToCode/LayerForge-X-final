#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the final report deliverables from the stitched report sources.")
    p.add_argument("--output", default="docs/final_report_pack/LayerForge_X_Final_Report_2026_04_22.docx")
    p.add_argument("--pdf-output", default="docs/final_report_pack/LayerForge_X_Final_Report_2026_04_22.pdf")
    p.add_argument("--full-markdown", default="docs/final_report_pack/LayerForge_X_Final_Report_FULL.md")
    p.add_argument("--reference-doc", default="docs/final_report_pack/reference.docx")
    p.add_argument("--build-manifest", default="docs/final_report_pack/build_manifest.json")
    p.add_argument("--skip-pdf", action="store_true", help="Skip LibreOffice PDF generation.")
    return p.parse_args()


def _resolve_path(value: str) -> Path:
    return (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"required tool not found: {name}")
    return path


def main() -> int:
    args = parse_args()
    output = _resolve_path(args.output)
    pdf_output = _resolve_path(args.pdf_output)
    full_markdown = _resolve_path(args.full_markdown)
    reference_doc = _resolve_path(args.reference_doc)
    build_manifest = _resolve_path(args.build_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    full_markdown.parent.mkdir(parents=True, exist_ok=True)
    build_manifest.parent.mkdir(parents=True, exist_ok=True)

    base = ROOT / "docs" / "final_report_pack" / "LayerForge_X_Final_Report_SOURCE.md"
    sections = [
        ROOT / "docs" / "final_report_pack" / "sources" / "01_LITERATURE_REVIEW_ADVANCED.md",
        ROOT / "docs" / "final_report_pack" / "sources" / "02_BENCHMARKING_PROTOCOL.md",
        ROOT / "docs" / "final_report_pack" / "sources" / "03_NOVELTY_AND_METHOD.md",
        ROOT / "docs" / "final_report_pack" / "sources" / "04_ABLATIONS_AND_TABLES.md",
        ROOT / "docs" / "final_report_pack" / "sources" / "05_REFERENCES.md",
    ]

    text = base.read_text(encoding="utf-8")
    replacements = {
        "<!-- include: 01 -->": sections[0].read_text(encoding="utf-8"),
        "<!-- include: 02 -->": sections[1].read_text(encoding="utf-8"),
        "<!-- include: 03 -->": sections[2].read_text(encoding="utf-8"),
        "<!-- include: 04 -->": sections[3].read_text(encoding="utf-8"),
        "<!-- include: 05 -->": sections[4].read_text(encoding="utf-8"),
    }
    for marker, content in replacements.items():
        text = text.replace(marker, content)

    full_markdown.write_text(text, encoding="utf-8")

    _require_tool("pandoc")
    with tempfile.TemporaryDirectory() as tmpdir:
        stitched = Path(tmpdir) / "stitched_report.md"
        stitched.write_text(text, encoding="utf-8")
        cmd = [
            "pandoc",
            str(stitched),
            "--from",
            "markdown",
            "--resource-path",
            str(ROOT / "docs" / "final_report_pack") + ":" + str(ROOT / "docs" / "final_report_pack" / "sources") + ":" + str(ROOT / "docs"),
            "--reference-doc",
            str(reference_doc),
            "-o",
            str(output),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

    pdf_generated = False
    if not args.skip_pdf:
        libreoffice = shutil.which("libreoffice")
        if libreoffice:
            subprocess.run(
                [
                    libreoffice,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(pdf_output.parent),
                    str(output),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            generated = pdf_output.parent / f"{output.stem}.pdf"
            if generated != pdf_output and generated.exists():
                generated.replace(pdf_output)
            pdf_generated = pdf_output.exists()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_shell": _display_path(base),
        "sources": [_display_path(section) for section in sections],
        "outputs": {
            "docx": _display_path(output),
            "markdown": _display_path(full_markdown),
            "pdf": _display_path(pdf_output) if pdf_generated else None,
        },
        "reference_doc": _display_path(reference_doc),
        "tools": {
            "pandoc": shutil.which("pandoc"),
            "libreoffice": shutil.which("libreoffice"),
        },
    }
    build_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(output)
    print(full_markdown)
    if pdf_generated:
        print(pdf_output)
    print(build_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
