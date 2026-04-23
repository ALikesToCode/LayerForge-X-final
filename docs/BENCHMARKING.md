# Evaluation Framework and Benchmarking Protocol

This document outlines the evaluation tracks, reference datasets, and performance metrics employed to validate the LayerForge-X framework.

## Benchmarking Taxonomy

The following table summarizes the primary evaluation axes and their corresponding data sources:

| Evaluation Objective | Reference Dataset | Metric |
|---|---|---|
| **Visible Semantic Grouping** | COCO Panoptic, ADE20K | Mean Intersection-over-Union (mIoU) |
| **Open-Vocabulary Detection** | LVIS, ODinW, custom prompts | Average Precision (AP), Recall@K |
| **Monocular Geometry** | NYU Depth V2, DIODE, KITTI | Absolute Relative Error (AbsRel), RMSE |
| **Layer Ordering Logic** | Synthetic Composites, NYU V2 | Pairwise Order Accuracy, Occlusion-Edge F1 |
| **Alpha Matting Fidelity** | Synthetic Alpha, Layered Scenes | Sum of Absolute Differences (SAD), MSE |
| **Background Completion** | Masked Places, Synthetic Hidden | LPIPS, FID, KID |
| **Intrinsic Decomposition** | Intrinsic Images in the Wild (IIW) | Weighted Human Disagreement Rate (WHDR) |
| **End-to-End Recomposition** | Synthetic LayerBench Scenes | PSNR, SSIM, Order Accuracy |

## Synthetic Benchmarking Protocol

As real-world photographic datasets lack ground-truth layer ordering and amodal extent, LayerForge-X utilizes a custom synthetic generator (`scripts/make_synthetic_dataset.py`) to produce controlled evaluation data. The advanced `layerbench_pp` format provides high-fidelity targets for:

- **Modal and Amodal Masks:** Precise visible and estimated object extents.
- **Alpha Mattes:** Ground-truth soft alpha for matting validation.
- **Associated Effects:** Dedicated RGBA layers for shadows and reflections.
- **Intrinsic Components:** Ground-truth albedo and shading maps.
- **Graph Metadata:** Authoritative occlusion and depth-ordering manifests.

This synthetic environment allows for isolated testing of depth-ordering accuracy and hidden-region completion that cannot be reliably measured in the wild.

## Qualitative Assessment and Visualization

Qualitative results for curated real-world images are presented using a standardized multi-panel layout, which includes:

1. **Input RGB:** The source image.
2. **Segmentation Overlay:** Visualization of semantic and instance proposals.
3. **Inferred Depth:** Monocular geometry estimate.
4. **Layer Contact Sheet:** Individual decomposition nodes.
5. **Completed Background:** Output of the inpainting module.
6. **Recomposed Scene:** Final composite from the ordered DALG.
7. **Intrinsic Split:** Decoupled albedo and shading layers.

For the **Recursive Peeling** workflow, a dedicated storyboard is used to demonstrate the iterative extraction process, highlighting the residual image and effect extraction at each stage.

## Ablation Study Matrix

The following ablation study evaluates the performance impact of individual architectural components:

| Configuration | Segmentation Architecture | Depth Model | Inpainting Module | Intrinsic Method | Primary Objective |
|---|---|---|---|---|---|
| **A (Baseline)** | SLIC | Geometric (Luminance) | Telea | Retinex | Establish performance baseline |
| **B** | Mask2Former | Geometric (Luminance) | Telea | Retinex | Evaluate semantic influence |
| **C** | Mask2Former | Depth Anything V2 | Telea | Retinex | Measure monocular geometry impact |
| **D** | Grounded SAM2 | Depth Pro | LaMa | Retinex | Validate high-fidelity masks + metric depth |
| **E (Native)** | Grounded SAM2 | Ensemble | LaMa | External Hook | Optimize for the strongest native configuration |
| **F (Peeling)** | Grounded SAM2 + Peel | Depth Pro | Iterative LaMa | Retinex | Evaluate recursive decomposition |
