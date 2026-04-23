# Comparative Analysis: Qwen-Image-Layered Baseline

*Submission note: the high-volume `runs/` and `data/` directories used for these experiments are omitted from the submission archive. The source-of-truth evidence for this comparison is `PROJECT_MANIFEST.json` together with the metric snapshots in `report_artifacts/metrics_snapshots/`.*

## Role of Qwen-Image-Layered in Evaluation

Qwen-Image-Layered is a strong public end-to-end generative baseline for image decomposition. LayerForge-X treats it as a **frontier baseline** and as an optional **proposal source**. This comparison is intended to separate the value of end-to-end generative layers from the value of explicit geometry, ordering, and graph metadata.

## Architectural and Functional Comparison

| Axis of Comparison | Qwen-Image-Layered | LayerForge-X |
|---|---|---|
| **Core Architecture** | End-to-end generative decomposition | Modular, interpretable scene graph (DALG) |
| **Primary Output** | Serialized RGBA layer stack | DALG manifest (graph, metadata, benchmarks) |
| **Depth Representation** | Implicit / not primary metadata | Explicit near-to-far ordering |
| **Topology** | Implicit occlusion reasoning | Confidence-weighted graph edges |
| **Spatial Reasoning** | Generated layers | Modal, amodal, and hidden-region masks |
| **Intrinsic Properties** | Not a primary objective | Per-layer albedo and shading decomposition |
| **Evaluation Axis** | Visual/Generative quality | Multi-axial (segmentation, order, fidelity, editability) |
| **Interpretability** | Model-driven (Black-box) | Metadata-driven (Inspectable nodes/edges) |

## Hybrid Enrichment Strategy

LayerForge-X facilitates a unique hybrid approach where external generative layers (e.g., from Qwen) are enriched with structured DALG metadata. This allows for a direct assessment of the value added by explicit depth estimation, amodal reasoning, and intrinsic decomposition.

### Execution: Qwen Baseline Generation
```bash
.venv/bin/python scripts/run_qwen_image_layered.py \
  --input data/demo/truck.jpg \
  --output-dir runs/qwen_truck_layers_raw_640_20 \
  --layers 4
```

### Execution: LayerForge-X Enrichment
```bash
.venv/bin/layerforge enrich-qwen \
  --input data/demo/truck.jpg \
  --layers-dir runs/qwen_truck_layers_raw_640_20 \
  --output runs/qwen_truck_enriched_640_20 \
  --config configs/cutting_edge.yaml \
  --preserve-external-order
```

## Experimental Results: Five-Image Frontier Review

Our evaluation includes a comparative review across five diverse images. The results contrast native LayerForge-X performance against raw Qwen outputs and hybrid (enriched) variants.

### Aggregate Performance Summary

| Method | Mean PSNR | Mean SSIM | Mean Self-Eval Score | Evaluation Wins |
|---|---:|---:|---:|---:|
| **LayerForge Native** | 37.6688 | 0.9708 | 0.6981 | 4 / 5 |
| Qwen Raw (4) | 29.0757 | 0.8850 | 0.2824 | 0 / 5 |
| Qwen + Graph (Preserve) | 28.5539 | 0.8638 | 0.5843 | 0 / 5 |
| Qwen + Graph (Reorder) | 28.5397 | 0.8637 | 0.5834 | 1 / 5 |

### Interpretative Analysis

1. **Fidelity vs. Structure:** While raw generative outputs achieve competitive PSNR, the LayerForge Native pipeline consistently produces superior SSIM and higher self-evaluation scores. This indicates that explicit geometric and semantic structure yields a more coherent and editable representation.
2. **Hybrid Row Utility:** The `Qwen + Graph (Preserve)` configuration shows what is gained when strong generative layers are augmented with explicit structural metadata. Its value is the added graph, amodal, and intrinsic representation, not a blanket claim of higher raw fidelity.
3. **Ordering Influence:** The comparison between `Preserve` and `Reorder` modes highlights the influence of explicit depth-graph ordering on recomposition fidelity, showing that graph-driven reordering maintains high performance while providing an inspectable depth hierarchy.

## Formal Comparative Statement

LayerForge-X acknowledges Qwen-Image-Layered as an important frontier generative baseline. The contribution of LayerForge-X is the explicit structural layer on top: depth ordering, graph relations, amodal support, and audit-ready manifests. The repository therefore positions Qwen and LayerForge-X as complementary rather than interchangeable systems.
