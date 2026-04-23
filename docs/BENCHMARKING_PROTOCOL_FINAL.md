# Benchmarking Protocol and Evaluation Metrics

Single-image layer decomposition is an inherently multi-dimensional task, necessitating a rigorous and multi-axial evaluation framework. A system may demonstrate superior semantic segmentation while failing to establish correct depth hierarchy, or achieve precise ordering at the expense of recomposition fidelity. Consequently, the LayerForge-X protocol decomposes evaluation into distinct tracks, each designed to isolate and stress specific pipeline components.

## Evaluated Methods and Baseline Configurations

| Identifier | Method Configuration | Evaluation Objective |
|---:|---|---|
| **M0** | Classical (SLIC) + Luminance Depth | Deterministic baseline |
| **M1** | Mask2Former + Global Median Depth | Modular panoptic segmentation baseline |
| **M2** | Mask2Former + Boundary Graph | Validation of graph-based ordering logic |
| **M3** | Grounded SAM2 + Boundary Graph | Open-vocabulary control and extraction |
| **M4** | Grounded SAM2 + Learned Ranker | Learning-based order optimization |
| **M5** | Qwen-Image-Layered | Frontier generative baseline |
| **M6** | Qwen + DALG (LayerForge Augmented) | Hybrid enrichment and re-ordering |
| **M7** | Full LayerForge-X (Native) | Integrated system performance |
| **M8** | LayerForge Peeling (Recursive) | Iterative residual decomposition |

## Dataset Strategic Allocation

Each reference dataset is selected to validate a specific architectural claim:

| Reference Dataset | Validation Role |
|---|---|
| **Synthetic-LayerBench** | Ground-truth for amodal extents, ordering, alpha matting, and hidden regions. |
| **COCO Panoptic** | Coarse-grained semantic grouping and instance-level fidelity. |
| **ADE20K** | Scene parsing and "stuff" region categorization. |
| **NYU Depth V2** | Indoor monocular depth and relative ordering. |
| **DIODE** | Diverse indoor/outdoor monocular geometry. |
| **KINS / COCOA** | Amodal mask estimation and occlusion reasoning. |
| **IIW** | Intrinsic decomposition (reflectance and shading). |
| **Curated Real-World Set** | Qualitative demonstration and interactive validation. |

## Evaluation Track A: Segmentation Fidelity

This track utilizes standard panoptic and semantic metrics to evaluate visible layer proposals. While Panoptic Quality (PQ) is considered, the current framework prioritizes group-level mIoU for consistency:

- **mIoU:** Mean Intersection-over-Union across all classes.
- **Pixel Accuracy:** Global percentage of correctly classified pixels.
- **Thing/Stuff mIoU:** Disaggregated performance for discrete instances and amorphous regions.

## Evaluation Track B: Monocular Geometry and Ordering

### 1. Pairwise Layer Order Accuracy (PLOA)
Measures the accuracy of front-to-back relationships between valid layer pairs. To ensure statistical significance, pairs with depth discrepancies below a pre-defined threshold are excluded.

### 2. Boundary-Weighted PLOA (BW-PLOA)
Assigns higher significance to adjacent layers that share a visible boundary, as these interactions are the most visually impactful for recomposition.

### 3. Occlusion Edge F1-Score
Treats the inferred occlusion graph as a set of directed edges and evaluates the precision and recall of the predicted topology against ground truth.

## Evaluation Track C: Recomposition and Reconstruction

Measures the fidelity of the composite image ($ \hat{I} $) generated from the ordered DALG stack against the original input RGB ($ I $).

**Metrics:**
- **PSNR / SSIM:** Peak Signal-to-Noise Ratio and Structural Similarity Index.
- **LPIPS:** Learned Perceptual Image Patch Similarity.
- **Alpha Coverage Error:** Quantitative measure of matting-induced artifacts or missing coverage.

## Evaluation Track D: Amodal Reasoning and Background Completion

Distinguishes between observed (modal) fidelity, estimated (amodal) spatial extent, and the quality of synthesized hidden content ($ M_{hidden} = M_{amodal} - M_{visible} $).

**Metrics:**
- Amodal and Hidden-Region IoU.
- Masked PSNR/SSIM for disoccluded regions.

## Evaluation Track E: Intrinsic Decomposition

Validates the decoupling of albedo and shading components.
- **WHDR:** Weighted Human Disagreement Rate (standard for IIW).
- **Albedo/Shading MSE:** Mean Squared Error against synthetic ground truth.

## Evaluation Track F: Editability and Functional Utility

The ultimate validation of the DALG representation is its utility in practical editing workflows. This track evaluates performance across six core operations:

1. **Object Removal / Movement**
2. **Parallax Preview Generation**
3. **Intrinsic Recoloring (Albedo Edit)**
4. **Relighting (Shading Adjustment)**
5. **Depth-Based Post-Processing (e.g., Background Blur)**

**Evaluation Criteria:**
- **Region Preservation MAE:** Ensuring non-edited areas remain unchanged.
- **Artifact Scoring:** Identifying seams, halos, or inpainting inconsistencies.
- **Operational Performance:** System latency and memory footprint diagnostics.

## Reproducibility and Synthetic Harness

To ensure experimental transparency, the repository provides a lightweight synthetic benchmarking harness (`scripts/make_synthetic_dataset.py` and `layerforge benchmark`). These utilities facilitate the generation of controlled environments and the automated execution of the evaluation tracks described above.
