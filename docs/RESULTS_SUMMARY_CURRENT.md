# Results Summary (Current Submission)

This document provides a concise overview of the experimental results and quantitative evidence presented in the current project submission.

## Canonical Evidence and Artifacts

The following artifacts constitute the canonical reported artifacts for the current repository state:

- **Project Manifest:** `PROJECT_MANIFEST.json`
- **Metric Snapshots:** `report_artifacts/metrics_snapshots/*.json`
- **Audit Logs:** `report_artifacts/command_log.md`
- **Reference Figures:** `docs/figures/*.png`

The underlying data and run directories (`runs/`, `results/`, `data/`) are used to generate these artifacts locally but are generally excluded from the public repository and submission ZIP to keep the distribution compact.

## Performance Highlights

### High-Fidelity Decomposition (Truck Scenario)
- **Configuration:** `truck_candidate_search_v2_best`
  - **PSNR:** `32.1053`
  - **SSIM:** `0.9848`
  - **Intrinsic Decomposition:** Retinex-based albedo and shading components.
  - **Layer Exports:** 20 discrete albedo and shading layers.

### Baseline Comparison (Qwen-Image-Layered)
- **Direct Qwen Output:** `qwen_truck_layers_raw_640_20`
  - **PSNR:** `29.8806`
  - **SSIM:** `0.8826`
- **Enriched Qwen (LayerForge-X Augmented):** `qwen_truck_enriched_640_20`
  - **PSNR:** `27.4633`
  - **SSIM:** `0.7949`
- **Five-Image Layer-Count Sweep:** `qwen_five_image_review`
  - **Best raw mean PSNR / SSIM:** `Qwen raw (3)` at `29.7574 / 0.8874`
  - **Best preserve-hybrid mean PSNR / SSIM:** `Qwen + graph preserve (3)` at `29.2311 / 0.8663`
  - **Observed reorder behavior:** graph-driven reorder remains a true graph-order export at `3/4` layers, while the fidelity guardrail now triggers on `3/5` six-layer runs and `4/5` eight-layer runs to fall back to the selected external visual stack instead of exporting catastrophic graph-order results

### Frontier Review and Generalization
- **Aggregate Performance:** `frontier_review`
  - **Mean Self-Evaluation Score (LayerForge Native):** `0.6283`
  - **Target Image Success Rate:** 4 out of 5 images correctly decomposed and ordered.

### Prompt-Conditioned Extraction
- **Configuration:** `extract_benchmark_prompted_grounded`
  - **Semantic Text Hit Rate:** `1.0000` (100% successful extraction of prompted entities)
  - **Point-Only Interaction:** Demonstrated limited semantic verification without accompanying text labels.

### Transparent Layer Recovery
- **Benchmark:** `transparent_benchmark`
  - **Mean Absolute Error (Alpha):** `0.1131`
  - **Background PSNR:** `25.9863`
  - **Recomposition PSNR (Sanity Check):** `56.0066`

### Associated-Effect Extraction
- **Configuration:** `effects_groundtruth_demo_cutting_edge`
  - **Effect Mask IoU:** `0.3529`

## Interpretative Analysis

- **Recomposition Fidelity:** The high recomposition PSNR in the transparent benchmark serves as a verification of the alpha-blending logic, while alpha error and background inpainting quality remain the primary performance indicators for transparent-layer recovery.
- **Semantic Prompting:** Successful text-conditioned extraction confirms the efficacy of the open-vocabulary grounded segmentation pipeline.
- **Effect Extraction:** The current associated-effect extractor represents a heuristic approach; while it successfully identifies shadow and reflection regions, it remains an area for further refinement.
- **Qwen Layer Count:** The measured `3/4/6/8` sweep indicates that compact raw stacks (`3` or `6` layers) preserve fidelity better than deeper exported stacks on the shipped five-image bank. Preserve-style hybrid enrichment remains the cleanest metadata-first comparison, while deeper reorder rows are now held at usable fidelity by an explicit fallback guardrail instead of being allowed to collapse.

## Scope of Documentation

This summary serves as the prose companion to the canonical reported artifacts. Historical working notes and intermediate metrics are excluded so that the public repository remains focused on the verified measurements and figures.
