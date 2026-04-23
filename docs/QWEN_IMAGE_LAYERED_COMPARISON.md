# Comparative Analysis: Qwen-Image-Layered Baseline

*Repository note: the high-volume `runs/` and `data/` directories used for these experiments are omitted from the public repository and submission ZIP. The canonical reported artifacts for this comparison are `PROJECT_MANIFEST.json` together with the metric snapshots in `report_artifacts/metrics_snapshots/`.*

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

## Experimental Results: Five-Image Layer-Count Sweep

Our evaluation includes a measured five-image sweep across `3`, `4`, `6`, and `8` requested Qwen layers. The goal is to separate the effect of layer count from the effect of structural LayerForge enrichment.

### Aggregate Performance by Requested Qwen Layer Count

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Qwen Raw (3) | 29.7574 | 0.8874 |
| Qwen Raw (4) | 29.0757 | 0.8850 |
| Qwen Raw (6) | 29.1079 | 0.8800 |
| Qwen Raw (8) | 27.1419 | 0.8663 |
| Qwen + Graph Preserve (3) | 29.2311 | 0.8663 |
| Qwen + Graph Preserve (4) | 28.5539 | 0.8638 |
| Qwen + Graph Preserve (6) | 28.6464 | 0.8588 |
| Qwen + Graph Preserve (8) | 26.7452 | 0.8444 |
| Qwen + Graph Reorder (3) | 29.2263 | 0.8663 |
| Qwen + Graph Reorder (4) | 28.5397 | 0.8637 |
| Qwen + Graph Reorder (6) | 21.4064 | 0.8133 |
| Qwen + Graph Reorder (8) | 18.4597 | 0.7827 |

### Interpretative Analysis

1. **Compact raw stacks remain strongest on fidelity.** `Qwen raw (3)` is the best pure-PSNR/SSIM setting on the shipped five-image bank, with `Qwen raw (6)` close behind and both outperforming the deeper `8`-layer raw export.
2. **Preserve-style enrichment is the stable hybrid.** `Qwen + graph preserve` drops below raw Qwen on recomposition fidelity, but the degradation remains moderate through `3/4/6` layers while adding explicit graph, amodal, and intrinsic structure.
3. **Graph reorder does not scale monotonically with layer count.** Reorder is acceptable at `3/4` layers, but the `6`- and `8`-layer runs degrade sharply. The current measured evidence therefore supports preserve-style enrichment as the safer default hybrid path.

## Experimental Results: Five-Image Frontier Review

The repository also keeps a five-image frontier candidate-bank review that compares native LayerForge, recursive peeling, and the `4`-layer Qwen family under an explicit editability-aware selector.

### Frontier Summary

| Method | Mean PSNR | Mean SSIM | Mean Self-Eval Score | Evaluation Wins |
|---|---:|---:|---:|---:|
| **LayerForge Native** | 37.6688 | 0.9708 | 0.6981 | 4 / 5 |
| Qwen Raw (4) | 29.0757 | 0.8850 | 0.2824 | 0 / 5 |
| Qwen + Graph Preserve (4) | 28.5539 | 0.8638 | 0.5843 | 0 / 5 |
| Qwen + Graph Reorder (4) | 28.5397 | 0.8637 | 0.5834 | 1 / 5 |

### Frontier Interpretation

1. **Fidelity vs. structure:** raw generative outputs remain competitive on PSNR, but the native LayerForge candidate-bank winner still leads on the explicit editability-aware selector.
2. **Hybrid row utility:** the hybrid rows show what is gained when strong generative layers are augmented with explicit structural metadata. Their value is graph, amodal, and intrinsic representation, not a blanket claim of higher raw fidelity.
3. **Ordering influence:** the `4`-layer frontier comparison remains a useful controlled view of preserve vs reorder under the current selector, even though the broader `3/4/6/8` sweep shows that reorder becomes brittle at higher layer counts.

## Formal Comparative Statement

LayerForge-X acknowledges Qwen-Image-Layered as an important frontier generative baseline. The contribution of LayerForge-X is the explicit structural layer on top: depth ordering, graph relations, amodal support, and explicit DALG manifests. The repository therefore positions Qwen and LayerForge-X as complementary rather than interchangeable systems.
