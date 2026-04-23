# Results Summary (current submission)

This file is the human-readable bridge to the current measured evidence pack.

## Source of truth

Treat these artifacts as the canonical submission evidence:

- `PROJECT_MANIFEST.json`
- `report_artifacts/metrics_snapshots/*.json`
- `report_artifacts/command_log.md`
- `docs/figures/*.png`

The heavyweight raw `runs/`, `results/`, and `data/` directories are used to produce those artifacts locally, but they are commonly omitted from the submission ZIP.

## Current measured highlights

- `truck_candidate_search_v2_best`
  - PSNR: `32.1053`
  - SSIM: `0.9848`
- `qwen_truck_layers_raw_640_20`
  - PSNR: `29.8806`
  - SSIM: `0.8826`
- `qwen_truck_enriched_640_20`
  - PSNR: `27.4633`
  - SSIM: `0.7949`
- `frontier_review`
  - `LF native` mean self-eval: `0.6283`
  - `LF native` best-image wins: `4/5`
- `extract_benchmark_prompted_grounded`
  - text hit rate: `1.0000`
  - point-only hit rate: `0.0000`
- `transparent_benchmark`
  - transparent alpha MAE: `0.1131`
  - background PSNR: `25.9863`
  - recompose PSNR: `56.0066` (sanity check only)
- `effects_groundtruth_demo_cutting_edge`
  - effect IoU: `0.3529`

## Interpretation notes

- Transparent recomposition is a sanity check; alpha error and clean-background quality are the primary transparent-layer metrics.
- Point-only and box-only prompt queries can have high overlap but still fail the semantic hit criterion because no text label is available for target verification.
- The associated-effect extractor is a heuristic prototype and should be framed that way in the report.

## Legacy note

The previous development-era summary now lives at `docs/legacy/RESULTS_SUMMARY_2026_04_19_legacy.md`. It is kept only as historical working material and should not be treated as the final submission source of truth.
