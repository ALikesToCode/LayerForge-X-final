# Diagnostic Visualization and Figure Index

This document serves as the authoritative index for the diagnostic visualizations and comparative figures generated from the experimental runs. These assets constitute the primary qualitative evidence for the LayerForge-X submission.

## Reproduction and Asset Generation

To regenerate the complete figure pack, execute the following command:

```bash
.venv/bin/python scripts/generate_report_figures.py
```

*Note: Full regeneration requires access to the high-volume `runs/`, `results/`, and `data/` directories. For the final submission, pre-generated PNG assets are provided in `docs/figures/`, with their underlying dependencies documented in `report_artifacts/figure_sources/figure_manifest.json`.*

## Figure Catalog

- **[Truck Recomposition Analysis](figures/truck_recomposition_comparison.png):** A comparative study featuring the reference RGB, raw Qwen-Image-Layered output, LayerForge-augmented Qwen, and various LayerForge-X native configurations (including the autotune-optimized winner).
- **[Truck Layer Stack Decomposition](figures/truck_layer_stack_comparison.png):** Visualizes the internal layer structure across baselines, contrasting raw generative layers with semantically grouped and depth-ordered DALG stacks.
- **[Truck Performance Metrics](figures/truck_metrics_comparison.png):** Quantitative comparison of layer count, recomposition PSNR, and SSIM across all truck-centric evaluation tracks.
- **[Ablation: Prompt Conditioning](figures/truck_prompt_ablation.png):** Evaluates the influence of prompt engineering—contrasting LLM-generated, manually curated, and augmented prompts—on decomposition fidelity.
- **[Ablation: Depth-Ordering Logic](figures/synthetic_ordering_ablation.png):** Comparison of `boundary-based` versus `learning-based` (learned ranker) ordering on held-out synthetic scenes, highlighting improvements in recomposition PSNR.
- **[Qualitative Performance Gallery](figures/qualitative_gallery.png):** Representative results for the `astronaut`, `coffee`, and `cat` scenarios, including input RGB, panoptic segmentation overlays, and ordered layer contact sheets.
- **[Associated-Effect Extraction Demo](figures/effects_layer_demo.png):** Demonstrates the extraction of shadows and reflections using a controlled `layerbench_pp` synthetic scene.
- **[Intrinsic Decomposition Analysis](figures/intrinsic_layer_demo.png):** Visualizes the decoupling of global and per-layer albedo and shading components for the optimized truck scenario.
- **[Segmentation Benchmark (COCO vs. ADE20K)](figures/public_benchmark_comparison.png):** Comparative performance charts across standard panoptic datasets, measuring mIoU across thing/stuff categories.
- **[Monocular Geometry Benchmark (DIODE)](figures/public_depth_comparison.png):** Evaluates depth estimation error and indoor/outdoor generalization for the geometric baseline against the Depth Pro model.
- **[Frontier Candidate Review](figures/frontier_review.png):** A comprehensive summary of the five-image self-evaluation bank, contrasting native, peeling, and hybrid decomposition strategies.
- **[Prompt-Conditioned Extraction Benchmark](figures/prompt_extract_benchmark.png):** Quantitative analysis of semantic hit rates and extraction fidelity across text, point, box, and hybrid query types.
- **[Transparent Layer Recovery Benchmark](figures/transparent_benchmark.png):** Evaluation of alpha-composited decomposition, focusing on alpha MAE and background inpainting quality.

## Strategic Integration in Research Report

The following mapping identifies the recommended placement of these figures within the final research report:

| Research Section | Recommended Figure |
|---|---|
| **Executive Summary / Introduction** | `truck_recomposition_comparison.png` |
| **System Architecture / Methodology** | `truck_layer_stack_comparison.png` |
| **Quantitative Results and Analysis** | `truck_metrics_comparison.png` |
| **Native Pipeline Optimization** | `truck_prompt_ablation.png` |
| **Ablation Studies and Novelty** | `synthetic_ordering_ablation.png` |
| **Qualitative Assessment** | `qualitative_gallery.png` |
| **Specialized Track: Associated Effects** | `effects_layer_demo.png` |
| **Specialized Track: Intrinsic Factors** | `intrinsic_layer_demo.png` |
| **Public Dataset Validation** | `public_benchmark_comparison.png` |
| **Geometry and Depth Validation** | `public_depth_comparison.png` |
| **Frontier Selector Performance** | `frontier_review.png` |
| **Prompt-Conditioned Extraction** | `prompt_extract_benchmark.png` |
| **Transparent Scene Decomposition** | `transparent_benchmark.png` |
