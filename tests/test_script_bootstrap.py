from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_WITH_BOOTSTRAP = [
    "scripts/build_site_data.py",
    "scripts/export_report_artifacts.py",
    "scripts/run_editability_suite.py",
    "scripts/run_effects_demo.py",
    "scripts/run_extract_benchmark.py",
    "scripts/run_frontier_comparison.py",
    "scripts/run_qwen_image_layered.py",
    "scripts/run_transparent_benchmark.py",
    "scripts/score_qwen_raw_layers.py",
]


def test_layerforge_scripts_bootstrap_src_path() -> None:
    for rel_path in SCRIPTS_WITH_BOOTSTRAP:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert 'sys.path.insert(0, str(ROOT / "src"))' in text, rel_path
