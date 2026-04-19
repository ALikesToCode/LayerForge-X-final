# LayerForge-X

**LayerForge-X** converts a single RGB image into a depth-aware, semantically grouped, amodal RGBA layer graph.

The project is designed for the assignment:

> Given a single bitmap RGB image, output re-composable RGBA layers with semantic grouping, depth order, and optional per-layer albedo/shading.

LayerForge-X does more than export segmentation masks. It builds a **Depth-Aware Amodal Layer Graph (DALG)**:

- each node is an editable RGBA layer;
- each node stores semantic label, semantic group, visible mask, soft alpha, amodal mask, depth statistics, and optional albedo/shading layers;
- each edge stores pairwise occlusion / near-far evidence;
- the stack is exported near → far with debug visualizations and metrics.

Qwen-Image-Layered is treated as a frontier baseline, not ignored. This repo includes an `enrich-qwen` command that imports RGBA layers from Qwen or any other external decomposer, then adds LayerForge-X depth ordering, occlusion graph metadata, amodal masks, and intrinsic layers.

## Repo layout

```text
configs/
  fast.yaml                  # deterministic fallback; works without huge model downloads
  cutting_edge.yaml          # GroundingDINO/SAM2 + ensemble depth config

src/layerforge/
  alpha.py                   # soft alpha refinement
  benchmark.py               # synthetic benchmark runner
  cli.py                     # command-line interface
  compose.py                 # straight-alpha RGBA compositing
  depth.py                   # luminance/DepthPro/DepthAnything/Marigold hooks
  graph.py                   # DALG construction and boundary-weighted ordering
  inpaint.py                 # OpenCV/LaMa-style completion hooks
  intrinsics.py              # Retinex-style albedo/shading split + external hook
  pipeline.py                # full pipeline
  qwen_io.py                 # import/enrich Qwen or external RGBA layers
  segment.py                 # classical, Mask2Former, GroundingDINO+SAM2 modes
  visualize.py               # overlays/contact sheets

scripts/
  make_synthetic_dataset.py  # synthetic LayerBench generator
  run_grid.py                # run ablation grids

docs/
  FINAL_PROJECT_SPEC.md
  QWEN_IMAGE_LAYERED_COMPARISON.md
  BENCHMARKING_PROTOCOL_FINAL.md
  NOVELTY_AND_ABLATIONS_FINAL.md
  REPORT_TABLES.md
  final_report_pack/
```

## Install

```bash
cd LayerForge-X-final
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Fast mode dependencies are in `requirements.txt`. Model-backed runs use `requirements-models.txt`.

```bash
pip install -r requirements-models.txt
```

## Fast smoke run

```bash
python scripts/make_synthetic_dataset.py --output examples/synth --count 3

layerforge run \
  --input examples/synth/scene_000/image.png \
  --output runs/smoke \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance \
  --no-parallax
```

## Full model-backed run

```bash
layerforge run \
  --input path/to/image.jpg \
  --output runs/full_image \
  --config configs/cutting_edge.yaml \
  --segmenter grounded_sam2 \
  --depth ensemble \
  --prompts "person,car,chair,table,window,sky,road,tree,animal"
```

## Qwen / external layer enrichment

First generate RGBA layers using Qwen-Image-Layered or another external layer decomposer. Put the layer PNGs in a folder, then run:

```bash
layerforge enrich-qwen \
  --input path/to/original_image.png \
  --layers-dir path/to/qwen_rgba_layers \
  --output runs/qwen_enriched \
  --config configs/fast.yaml \
  --depth geometric_luminance
```

For stronger enrichment:

```bash
layerforge enrich-qwen \
  --input path/to/original_image.png \
  --layers-dir path/to/qwen_rgba_layers \
  --output runs/qwen_enriched_full \
  --config configs/cutting_edge.yaml \
  --depth depth_pro
```

The output will contain ordered layers, albedo/shading layers, amodal masks, and `debug/layer_graph.json`.

## Synthetic benchmark

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 20

layerforge benchmark \
  --dataset-dir data/synthetic_layerbench \
  --output-dir results/synthetic_fast \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```

Outputs:

```text
results/synthetic_fast/synthetic_benchmark.csv
results/synthetic_fast/synthetic_benchmark_summary.json
results/synthetic_fast/runs/<scene>/
```

## Main outputs

```text
layers_ordered_rgba/       # near → far layer stack
layers_grouped_rgba/       # semantic/depth grouped layers
layers_albedo_rgba/        # per-layer albedo approximation
layers_shading_rgba/       # per-layer shading approximation
layers_amodal_masks/       # estimated amodal support masks
debug/depth_gray.png
debug/semantic_overlay.png
debug/layer_graph.json
debug/background_completion.png
debug/parallax_preview.gif
manifest.json
metrics.json
```

## Final project thesis

LayerForge-X is not positioned as “another Qwen.” Qwen-Image-Layered is a strong generative RGB→RGBA decomposer. LayerForge-X is an interpretable, depth-aware, benchmarked layer representation that adds explicit near/far ordering, occlusion edges, amodal support, intrinsic appearance factors, and component-level evaluation.

Use Qwen as:
1. a baseline,
2. a proposal source,
3. a hybrid: Qwen layers + LayerForge graph enrichment.

That is the strongest defense for grading.
