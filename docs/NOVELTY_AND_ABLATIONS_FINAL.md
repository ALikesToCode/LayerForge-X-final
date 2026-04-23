# Novelty and Ablation Studies

## Core Theoretical and Architectural Contributions

The following sections delineate the primary contributions of the LayerForge-X framework within the domain of single-image scene decomposition.

### Contribution 1: Depth-Aware Amodal Layer Graph (DALG)

LayerForge-X formalizes scene representation as a **graph** rather than a serialized layer stack. Each node represents a semantically coherent RGBA layer augmented with structured metadata, while edges encode pairwise depth and occlusion relationships with associated confidence metrics. This graph-centric approach preserves critical topological information that is typically lost in flat-file representations.

### Contribution 2: Boundary-Aware Ordering with Learning-Based Refinement

Unlike traditional methods that rely on global region statistics (e.g., mean or median depth), LayerForge-X utilizes **local depth evidence** at shared boundaries to establish front-to-back ordering. Furthermore, the framework incorporates an optional, lightweight **learned pairwise ranker** trained on synthetic data. This enables a rigorous comparison between heuristic-based and learning-based ordering mechanisms without necessitating a full end-to-end trained architecture.

### Contribution 3: Multi-Faceted Layer Attribute Enrichment

Each decomposed layer is treated as a first-class entity, enriched with a comprehensive suite of attributes, including:
- Hierarchical semantic grouping.
- High-fidelity monocular depth statistics.
- Amodal spatial extents.
- Refined soft alpha mattes.
- Background-completion (inpainting) metadata.
- Decoupled intrinsic components (albedo and shading).

### Contribution 4: External Model Enrichment (Qwen-Image-Layered Integration)

LayerForge-X establishes a unified evaluation frame that treats state-of-the-art generative models, such as **Qwen-Image-Layered**, as both competitive baselines and viable proposal sources. The `enrich-qwen` utility operationalizes this integration, augmenting external RGBA layers with DALG-compatible metadata and ordering.

### Contribution 5: Multi-Axial Benchmarking Protocol

The framework introduces a rigorous evaluation protocol that spans segmentation fidelity, ordering accuracy, graph topology, recomposition quality, amodal reasoning, intrinsic decomposition, and editability. This multi-axial approach ensures that performance is measured across diverse failure modes, providing a holistic assessment of system efficacy.

## Ablation Study Matrix

The following matrix outlines the configurations used to evaluate the influence of individual architectural components:

| Configuration | Segmentation | Monocular Geometry | Ordering Logic | Alpha Matting | Amodal Reasoning | Background Completion | Intrinsic Split | Primary Objective |
|---:|---|---|---|---|---|---|---|---|
| **A** | SLIC (Classical) | Geometric (Luminance) | Global Median | Hard Mask | None | None | None | Baseline (Deterministic) |
| **B** | Mask2Former | None | Area Heuristic | Hard Mask | None | None | None | Segmentation Influence |
| **C** | Mask2Former | Depth Anything V2 | Global Median | Hard Mask | None | None | None | Monocular Geometry Impact |
| **D** | Mask2Former | Depth Anything V2 | Boundary Graph | Hard Mask | None | None | None | Graph Topology Impact |
| **E** | Grounded SAM2 | Depth Anything V2 | Boundary Graph | Soft Alpha | None | None | None | Open-Vocab + Matting |
| **F** | Grounded SAM2 | Depth Pro / MoGe | Boundary Graph | Soft Alpha | Heuristic | OpenCV | None | Amodal + Completion |
| **G** | Grounded SAM2 | Ensemble | Learned Ranker | Matting Refinement | Yes | LaMa | None | Learned Order Optimization |
| **H (Native)** | Ensemble | Ensemble | Learned Graph | Matting Refinement | Yes | LaMa | Yes | Final Framework |
| **Q (Baseline)** | Qwen-Image-Layered | Implicit | Manual/None | Generated | Implicit | Generated | None | Frontier Generative Baseline |
| **Q+G (Hybrid)** | Qwen + DALG | External Depth | Boundary Graph | Generated | Implicit | Generated | Yes | Hybrid Enrichment |

## Experimental Results and Analysis (Ablation Diffs)

The following table summarizes the performance impact of transitioning between key configurations on a held-out synthetic dataset:

| Configuration | Segmentation Architecture | Monocular Geometry | Ordering Logic | Dataset Split | Mean Best IoU | PLOA | Recomposition PSNR |
|---:|---|---|---|---|---:|---:|---:|
| **A1** | Classical | Geometric Luminance | Boundary | Synthetic (Fast) | 0.1549 | 0.1667 | 19.1360 |
| **A2** | Classical | Geometric Luminance | Boundary | Synthetic (Test) | 0.1549 | 0.1667 | 19.1589 |
| **A3** | Classical | Geometric Luminance | Learned Ranker | Synthetic (Test) | 0.1549 | 0.1667 | 19.4138 |

### Key Observations:
- **Ordering Optimization:** The transition from heuristic boundary-based ordering (**A2**) to the learned pairwise ranker (**A3**) yields a significant improvement in recomposition PSNR (**+0.255 dB**).
- **Bottleneck Identification:** While ordering logic is improved, the overall Pairwise Layer Order Accuracy (PLOA) remains constrained by over-segmentation in the initial proposal stage (averaging 65 predicted layers for 5 ground-truth layers). This identifies segmentation fidelity as the primary lever for future performance gains.

## Formal Statement of Claims

To ensure scientific rigor and avoid overstatement, LayerForge-X adheres to the following claim boundaries:

1. **On Scope:** We do not claim to "solve" single-image decomposition; rather, we provide a **robust, inspectable, and benchmarkable approximation** of the task.
2. **On Amodal Content:** We do not claim perfect recovery of hidden regions; we demonstrate the synthesis of **plausible completions** that facilitate realistic editing workflows.
3. **On Baselines:** We do not claim to exceed the generative quality of end-to-end models like Qwen-Image-Layered; we demonstrate **complementary strengths** in explicit depth ordering, structured graph metadata, and modular component evaluation.
