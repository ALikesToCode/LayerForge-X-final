#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from layerforge.editability import evaluate_run_editability


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate editability-focused metrics on LayerForge/Qwen run directories.")
    p.add_argument("--frontier-summary", default="runs/frontier_review/frontier_summary.json", help="Frontier summary JSON to read run directories from")
    p.add_argument("--output", default="runs/frontier_review/editability_suite_summary.json", help="Where to write the compact JSON summary")
    return p.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def main() -> int:
    args = parse_args()
    summary_path = repo_path(args.frontier_summary)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("rows", []):
        if row.get("status") != "ok":
            continue
        run_dir = repo_path(row["run_dir"])
        metrics = evaluate_run_editability(run_dir)
        rows.append(
            {
                "image": row["image"],
                "label": row["label"],
                "run_dir": row["run_dir"],
                "edit_response_remove": metrics.get("edit_response_remove"),
                "edit_response_move": metrics.get("edit_response_move"),
                "edit_response_recolor": metrics.get("edit_response_recolor"),
                "edit_success_score": metrics.get("edit_success_score"),
                "semantic_purity": metrics.get("semantic_purity"),
                "alpha_quality_score": metrics.get("alpha_quality_score"),
                "background_hole_ratio": metrics.get("background_hole_ratio"),
                "non_edited_region_preservation": metrics.get("non_edited_region_preservation"),
            }
        )
    output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(output_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
