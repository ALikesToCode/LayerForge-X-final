# Qwen-Image-Layered Comparison

## Why Qwen matters

Qwen-Image-Layered is directly relevant because it targets the same broad endpoint: generating semantically meaningful RGBA layers from one image. A strong project should not ignore it. LayerForge-X uses Qwen as a **frontier baseline** and optional **proposal source**.

## Difference in project positioning

| Aspect | Qwen-Image-Layered | LayerForge-X |
|---|---|---|
| Core approach | End-to-end generative layer decomposition | Modular interpretable layer graph |
| Primary output | RGBA layer stack | RGBA stack + graph + metadata + benchmarks |
| Depth order | Implicit / not primary metadata | Explicit near → far order |
| Occlusion | Implicit in generated layers | Explicit confidence-weighted edges |
| Amodal support | May be implicit or generated | Visible mask, amodal mask, hidden mask |
| Intrinsics | Not the core output | Per-layer albedo/shading approximation |
| Benchmarking | Visual / editing quality | Segmentation, order, graph, recomposition, amodal, intrinsic, editing |
| Interpretability | Lower; model-driven | Higher; node/edge metadata |
| Use in this repo | Baseline / external proposal source | Main method |

## Recommended experiment

Run:

```text
M0: classical + luminance depth
M1: Mask2Former + global median depth
M2: Mask2Former + boundary graph
M3: GroundingDINO + SAM2 + boundary graph
M4: full LayerForge-X
Q: Qwen-Image-Layered
Q+G: Qwen layers + LayerForge graph enrichment
```

## How to run Qwen enrichment

After exporting Qwen RGBA layers:

```bash
layerforge enrich-qwen \
  --input original.png \
  --layers-dir qwen_layers/ \
  --output runs/qwen_enriched \
  --config configs/cutting_edge.yaml \
  --depth depth_pro
```

This creates:

```text
layers_ordered_rgba/
layers_albedo_rgba/
layers_shading_rgba/
layers_amodal_masks/
debug/layer_graph.json
metrics.json
manifest.json
```

## Viva answer

If asked “Is this just Qwen-Image-Layered?”:

> Qwen-Image-Layered is the strongest recent generative baseline for RGB-to-RGBA layer decomposition, and I compare against it directly. My project focuses on a different part of the problem: turning layer decomposition into an interpretable, benchmarkable scene representation. LayerForge-X adds explicit near-to-far ordering, pairwise occlusion graph edges, amodal visible/hidden masks, background-completion metadata, per-layer albedo/shading, and component-level metrics. It can use Qwen as a baseline or proposal source, but the contribution is the depth-aware amodal layer graph and evaluation protocol.
