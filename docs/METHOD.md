# Methodology

## Problem Formulation

The primary challenge addressed by LayerForge-X is the decomposition of a single RGB image into a structured stack of semantically meaningful RGBA layers. This representation must facilitate downstream tasks such as recomposition, spatial reordering, semantic editing, and relighting.

## Depth-Aware Amodal Layer Graph (DALG)

The DALG serves as the canonical representation for a decomposed scene. Each node within the graph encapsulates a discrete semantic entity, storing the following attributes:

- **Masking and Alpha:** A visible mask accompanied by a refined soft alpha matte.
- **Semantic Metadata:** A semantic label and a hierarchical semantic group identifier.
- **Geometry:** Three distinct depth statistics: median depth, a "near" percentile, and a "far" percentile.
- **Amodal Extent:** An estimated amodal mask representing the full object extent, including occluded regions.
- **Intrinsic Properties:** Decoupled RGBA albedo and shading layers.
- **Topology:** Pairwise occlusion links and relative depth relationships to other nodes.

## Pipeline Architecture

1. **Segmentation Proposal:** The system supports multiple segmentation architectures, including SLIC for deterministic baselines, Mask2Former for closed-set panoptic segmentation, and an ensemble of GroundingDINO and SAM2 for open-vocabulary, prompt-conditioned extraction.
2. **Monocular Depth Estimation:** Depth is estimated using strong public models such as Depth Anything V2, Depth Pro, or Marigold, with support for ensemble-based refinement.
3. **Overlap Resolution:** Overlapping mask proposals are resolved by assigning pixels based on relative depth or pre-defined semantic priority.
4. **Spatial Partitioning of Background Regions:** Large "stuff" regions (e.g., sky, terrain, architectural surfaces) are partitioned into depth-quantile planes to prevent them from dominating the global ordering process.
5. **Alpha Matting and Refinement:** Binary masks are converted into soft alpha mattes, with depth discontinuities used to refine and harden boundaries.
6. **Order Inference and Occlusion Graph Construction:** Adjacent regions are analyzed for depth discrepancies at their shared boundaries to establish front-to-back edges. A learned pairwise ranker, trained on synthetic data, provides an optional high-fidelity ordering mechanism.
7. **Amodal Completion:** Object masks are expanded using morphological operations and hull-limited dilation to estimate occluded regions.
8. **Background Completion (Inpainting):** Foreground entities are removed and the resulting voids are filled using advanced inpainting techniques (e.g., LaMa) to create an edit-ready background layer.
9. **Intrinsic Decomposition:** Global albedo and shading components are computed and subsequently masked to generate per-layer intrinsic properties.
10. **Structured Export:** The final output includes an ordered layer stack, grouped layers, a DALG manifest (JSON), quantitative metrics, and diagnostic visualization panels.

## Learning-Based Pairwise Order Estimation

The framework incorporates an optional, lightweight module for pairwise depth ranking:

- **Training Data:** Derived from the synthetic **LayerBench** dataset, which provides ground-truth depth and occlusion metadata.
- **Feature Engineering:** Region-wise features are extracted, including depth statistics, mask overlap metrics, and geometric properties.
- **Model Architecture:** A logistic regression model is trained on these features to predict the probability of a "nearer" relationship between layer pairs.
- **Global Ordering:** Pairwise scores are aggregated to produce a globally consistent near-to-far ordering of the layer stack.

## System Novelty and Architectural Contributions

The primary contribution of LayerForge-X lies in the holistic integration of diverse computer vision tasks into a unified, inspectable representation:

> *LayerForge-X establishes a robust framework for single-image scene decomposition by synthesizing semantic grouping, monocular geometry, amodal reasoning, soft alpha matting, and intrinsic decomposition into a coherent Depth-Aware Amodal Layer Graph.*

While individual modules build on existing segmentation and depth architectures, the contribution lies in their integration into a single DALG pipeline and in the evaluation protocol used to measure layered scene understanding.

## Performance Evaluation and Analysis

Experimental results on held-out synthetic datasets demonstrate the efficacy of the learning-based ordering module:

- The learned pairwise ranker improves recomposition PSNR from **19.1589** to **19.4138**.
- Current performance bottlenecks are primarily associated with over-segmentation in the initial proposal stage, rather than the ordering logic itself. This indicates that future improvements in segmentation fidelity will yield significant gains in overall system performance.
