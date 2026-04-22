# LayerForge-X

**LayerForge-X** takes a single RGB image and turns it into a depth-aware, semantically grouped, amodal RGBA layer graph.

The assignment it was built against reads roughly like this:

> Given a single bitmap RGB image, output re-composable RGBA layers with semantic grouping, depth order, and optional per-layer albedo/shading.

Nothing about the wording forces you to export more than a bag of cutouts, but that felt unsatisfying to me. LayerForge-X goes a step further and builds what I call a **Depth-Aware Amodal Layer Graph (DALG)**:

- every node is an editable RGBA layer;
- each node carries a semantic label, a semantic group, the visible mask, a soft alpha, the estimated amodal mask, some depth statistics, and (optionally) matched albedo/shading layers;
- edges record pairwise occlusion and near/far evidence between neighbouring nodes;
- the final stack is written out near → far, together with debug visualisations and quantitative metrics.

Qwen-Image-Layered deserves special mention. It's a strong frontier baseline for this same problem, and I didn't want to pretend it doesn't exist. The repo therefore ships an `enrich-qwen` command that imports RGBA layers produced by Qwen (or any other external decomposer) and bolts LayerForge-X's depth ordering, occlusion graph metadata, amodal masks, and intrinsic layers on top.

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
  peeling.py                 # recursive semantic layer peeling + effects extraction
  pipeline.py                # full pipeline
  qwen_io.py                 # import/enrich Qwen or external RGBA layers
  ranker.py                  # lightweight learned pairwise near/far ranker
  segment.py                 # classical, Mask2Former, GroundingDINO+SAM2 modes
  visualize.py               # overlays/contact sheets

scripts/
  collect_run_metrics.py      # compact markdown/json tables from run directories
  export_report_artifacts.py # copy audit-safe metrics snapshots for ZIP submissions
  generate_report_figures.py # report-ready comparison panels and graphs
  make_synthetic_dataset.py  # synthetic LayerBench generator
  run_qwen_image_layered.py  # official Qwen-Image-Layered baseline runner
  run_grid.py                # run ablation grids

docs/
  FIGURES.md
  FINAL_PROJECT_SPEC.md
  QWEN_IMAGE_LAYERED_COMPARISON.md
  BENCHMARKING_PROTOCOL_FINAL.md
  NOVELTY_AND_ABLATIONS_FINAL.md
  REPORT_TABLES.md
  figures/
  final_report_pack/
```

## Install

```bash
# If you extracted a GitHub archive, the folder is usually LayerForge-X-final-main.
# If you cloned the repo directly, just cd into the repo root.
cd LayerForge-X-final-main
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Fast mode dependencies live in `requirements.txt`. For model-backed runs (the heavier stuff with Depth Pro, GroundingDINO, SAM2, and friends), install the extras:

```bash
pip install -r requirements-models.txt
```

On Python `3.14`, `simple-lama-inpainting` currently fails to build because of an older Pillow dependency. The repo still works because it falls back to OpenCV inpainting, and the model-backed stack used for the runs in this repo can be installed with:

```bash
pip install torch torchvision transformers accelerate diffusers safetensors
```

## Fast smoke run

If you just want to see the pipeline move end-to-end without downloading gigabytes of checkpoints, the synthetic generator plus fast config is the way to go:

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

For a quick regression gate that stays away from heavyweight model downloads:

```bash
./.venv/bin/pytest -q tests/test_smoke.py tests/test_merge.py tests/test_segment_api.py
```

## Full model-backed run

Once the model extras are installed, this is the recommended "the works" invocation:

```bash
layerforge run \
  --input path/to/image.jpg \
  --output runs/full_image \
  --config configs/best_score.yaml \
  --segmenter grounded_sam2 \
  --depth ensemble \
  --prompts "person,car,truck,chair,table,window,sky,road,tree,animal" \
  --prompt-source augment
```

The strongest current recipe in the repo is `configs/best_score.yaml`: ensemble depth, stronger boundary settings, and adaptive merge. The single biggest control knob is still the prompt list. For maximum score on a known scene, use curated prompts with `--prompt-source manual`; for a stronger automatic default, use `--prompt-source augment` to keep manual seed classes and let Gemini add extras.

Measured truck runs already in the repo:

| Run | Layers | PSNR | SSIM |
|---|---:|---:|---:|
| `runs/demo_grounded_depthpro_final` | 45 | 14.6477 | 0.8348 |
| `runs/truck_best_score` | 26 | 30.8214 | 0.9812 |
| `runs/truck_best_score_manual` | 19 | 31.3040 | 0.9813 |
| `runs/truck_best_score_augment` | 19 | 31.1524 | 0.9804 |
| `runs/truck_state_of_art_search_v2/best` | 20 | 32.1053 | 0.9848 |

So the repo is no longer stuck in the "interpretable but low-fidelity" regime. The upgraded native LayerForge recipe is now materially better than the old native run on both fidelity and stack compactness.

## State-of-the-art search

For the strongest native result on a specific image, use the search mode. It runs a small ladder of strong candidates, including manual prompts, augment mode, and Gemini-assisted prompt generation with multiple threshold presets, then keeps the best measured output.

```bash
layerforge autotune \
  --input data/demo/truck.jpg \
  --output runs/truck_state_of_art_search_v2 \
  --config configs/best_score.yaml \
  --prompts "truck,road,sky,tree,building,window,wheel,car" \
  --device cuda \
  --no-parallax
```

That command writes:

- `runs/truck_state_of_art_search_v2/search_summary.json`
- `runs/truck_state_of_art_search_v2/candidates/*`
- `runs/truck_state_of_art_search_v2/best`

For submission-safe auditing, `python scripts/export_report_artifacts.py` also copies the selected summary into `report_artifacts/metrics_snapshots/truck_search_summary.json`.

The current reproducible winner on the truck scene is `manual_precision` at `PSNR 32.1053`, `SSIM 0.9848`, `20` layers.

## Recursive semantic peeling

The repo now includes a second decomposition path aimed at the "frontier" formulation: recursively peel the frontmost editable entity, inpaint the residual canvas, and keep iterating until the background is reached.

```bash
layerforge peel \
  --input data/demo/truck.jpg \
  --output runs/truck_recursive_peel \
  --config configs/best_score.yaml \
  --segmenter grounded_sam2 \
  --depth ensemble \
  --prompts "truck,road,sky,tree,building,window,wheel,car" \
  --prompt-source augment \
  --max-layers 6
```

This writes:

```text
iterations/iteration_00/input.png
iterations/iteration_00/selected_mask.png
iterations/iteration_00/selected_layer.png
iterations/iteration_00/residual_inpainted.png
layers_effects_rgba/
debug/peeling_strip.png
```

The recursive path is intentionally explicit: each iteration logs the selected layer, the residual image after inpainting, and any associated effect layer recovered from the object-local residual.

## Frontier comparison and self-evaluation

The repo now includes a frontier orchestration script that treats LayerForge native, recursive peeling, raw Qwen, and fair Qwen hybrids as one candidate bank, then scores the successful candidates per image and records the repo's best editable pick.

```bash
python scripts/run_frontier_comparison.py \
  --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png \
  --output-root runs/frontier_review \
  --native-config configs/frontier.yaml \
  --peeling-config configs/recursive_peeling.yaml \
  --qwen-layers 4,6 \
  --qwen-steps 20 \
  --qwen-resolution 640 \
  --qwen-device cuda \
  --qwen-dtype bfloat16 \
  --qwen-offload sequential \
  --skip-existing
```

This writes:

```text
runs/frontier_review/
  frontier_summary.json
  frontier_summary.md
  <image>/
    best_decomposition.json
    why_selected.md
```

The current self-evaluation score is deliberately heuristic. It uses measured recomposition fidelity plus structural, editability, and runtime signals to choose the most useful editable candidate for each image. Details and caveats are in [docs/FRONTIER_WORKFLOW.md](docs/FRONTIER_WORKFLOW.md).

## Qwen / external layer enrichment

The idea here is simple: let a generative decomposer produce the initial RGBA layers, then use LayerForge-X to add structure (depth order, occlusion edges, amodal support, intrinsics). First generate layers with Qwen-Image-Layered or another decomposer, drop the PNGs in a folder, and then run:

Export official Qwen layers:

```bash
.venv/bin/python scripts/run_qwen_image_layered.py \
  --input data/demo/truck.jpg \
  --output-dir runs/qwen_truck_layers_raw_640_20 \
  --layers 4 \
  --resolution 640 \
  --steps 20 \
  --device cuda \
  --dtype bfloat16 \
  --offload sequential
```

Then enrich them with LayerForge-X:

```bash
.venv/bin/layerforge enrich-qwen \
  --input data/demo/truck.jpg \
  --layers-dir runs/qwen_truck_layers_raw_640_20 \
  --output runs/qwen_truck_enriched_640_20 \
  --config configs/cutting_edge.yaml \
  --depth depth_pro
```

For a stronger enrichment using the cutting-edge config:

```bash
layerforge enrich-qwen \
  --input path/to/original_image.png \
  --layers-dir path/to/qwen_rgba_layers \
  --output runs/qwen_enriched_full \
  --config configs/cutting_edge.yaml \
  --depth depth_pro
```

Add `--preserve-external-order` when you want the enriched export to keep Qwen's interpreted visual stack and only add LayerForge metadata. Leave that flag off when you want the exported stack reordered by the depth graph.

The output directory will contain ordered layers, albedo/shading layers, amodal masks, and a `debug/layer_graph.json` describing the graph structure.

## Submission-safe report artifacts

The local checkout includes heavyweight `runs/`, `results/`, and `data/` directories, but those are commonly excluded from submission ZIPs. To keep the report auditable even when those folders are omitted, export a compact artifact pack:

```bash
python scripts/export_report_artifacts.py
```

That writes:

```text
report_artifacts/
  README.md
  command_log.md
  metrics_snapshots/
  figure_sources/figure_manifest.json
```

The README and `PROJECT_MANIFEST.json` now point to these compact JSON snapshots instead of assuming the raw benchmark directories will always be present in the archive.

## Collecting report metrics

To avoid hand-copying numbers into the report tables:

```bash
python scripts/collect_run_metrics.py \
  runs/demo_grounded_depthpro_final \
  runs/truck_best_score \
  runs/truck_best_score_manual \
  runs/truck_best_score_augment \
  runs/truck_state_of_art_search_v2/best \
  runs/qwen_truck_layers_raw_640_20 \
  runs/qwen_truck_enriched_640_20
```

This prints a compact markdown table from the `metrics.json` files.

## Generating report figures

To regenerate the measured comparison figures used in the report pack:

```bash
python scripts/generate_report_figures.py
```

This writes:

```text
docs/figures/truck_recomposition_comparison.png
docs/figures/truck_layer_stack_comparison.png
docs/figures/truck_metrics_comparison.png
docs/figures/synthetic_ordering_ablation.png
docs/figures/qualitative_gallery.png
docs/figures/figure_manifest.json
```

Use `docs/FIGURES.md` as the index of which figure says what.

## Synthetic benchmark

Because real images don't come with ground-truth layers, a synthetic benchmark is essential for honest numbers. Generate one and run the benchmark harness:

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 20

layerforge benchmark \
  --dataset-dir data/synthetic_layerbench \
  --output-dir results/synthetic_fast \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```

This writes:

```text
results/synthetic_fast/synthetic_benchmark.csv
results/synthetic_fast/synthetic_benchmark_summary.json
results/synthetic_fast/runs/<scene>/
```

For the richer benchmark artifact format used by the updated report narrative, the generator now has an opt-in `layerbench_pp` mode:

```bash
python scripts/make_synthetic_dataset.py \
  --output data/synthetic_layerbench_pp \
  --count 20 \
  --output-format layerbench_pp \
  --with-effects
```

Each scene then also includes:

```text
visible_masks/
amodal_masks/
alpha_mattes/
layers_effects_rgba/
intrinsics/albedo.png
intrinsics/shading.png
depth.png
depth.npy
occlusion_graph.json
scene_metadata.json
```

The original `basic` mode is still the default so the existing lightweight benchmark path remains reproducible.

For the next external review, the remaining measured-work checklist is tracked in [docs/NEXT_REVIEW_CHECKLIST_2026_04_22.md](docs/NEXT_REVIEW_CHECKLIST_2026_04_22.md).

### Measured fast-path baseline

The deterministic fallback was actually run on `12` synthetic scenes. The submission-safe summary snapshot is `report_artifacts/metrics_snapshots/synthetic_benchmark_summary.json`.

| Split | Segmenter | Depth | Ordering | Mean best IoU | PLOA | Recompose PSNR |
|---|---|---|---|---:|---:|---:|
| synthetic fast | classical | geometric luminance | boundary | 0.1549 | 0.1667 | 19.1360 |

These numbers are intentionally modest. The fast fallback over-segments the synthetic scenes into about `65` layers for `5` ground-truth layers, which makes it useful as a deterministic baseline but not the final story for the report.

## Public real-data benchmarks

The repo now includes a real-data evaluator for visible semantic grouping on **COCO Panoptic val2017**. This is a coarse-group benchmark, not a full panoptic PQ implementation: COCO categories are mapped into the LayerForge groups (`person`, `animal`, `vehicle`, `furniture`, `plant`, `sky`, `road`, `ground`, `building`, `water`, `stuff`, `object`) and scored with dataset-level IoU.

Download the official validation split and panoptic annotations:

```bash
python scripts/download_coco_panoptic_val.py \
  --output-dir data/coco_panoptic_val \
  --archive-dir data/downloads/coco
```

Run the benchmark:

```bash
layerforge benchmark-coco-panoptic \
  --dataset-dir data/coco_panoptic_val \
  --output-dir results/coco_panoptic_mask2former_512 \
  --config configs/fast.yaml \
  --segmenter mask2former \
  --device cuda \
  --max-images 512 \
  --seed 7
```

Measured result already in the repo:

| Split | Segmenter | Images | Group mIoU | Thing mIoU | Stuff mIoU | Mean predicted segments |
|---|---|---:|---:|---:|---:|---:|
| COCO Panoptic val2017 sample | mask2former | 512 | 0.5660 | 0.5842 | 0.5479 | 6.45 |

Per-group IoU from `report_artifacts/metrics_snapshots/coco_panoptic_group_benchmark_summary.json`:

| Group | IoU |
|---|---:|
| person | 0.6854 |
| animal | 0.6483 |
| vehicle | 0.6801 |
| furniture | 0.4245 |
| plant | 0.7115 |
| sky | 0.8632 |
| road | 0.3975 |
| ground | 0.4634 |
| building | 0.5785 |
| water | 0.7798 |
| stuff | 0.2048 |
| object | 0.3556 |

This benchmark is intentionally scoped to what COCO can supervise honestly: visible grouping quality. Depth ordering, amodal completion, and intrinsics still need different public datasets.

The repo now also includes a second real-data evaluator for **ADE20K SceneParse150 validation**. This uses the same coarse-group IoU protocol as the COCO benchmark, but on a much broader scene-parsing dataset with denser indoor/outdoor background supervision. For ADE20K, the best available in-repo setup is the ADE-tuned Mask2Former config in [configs/ade20k_mask2former.yaml](configs/ade20k_mask2former.yaml).

Download the official ADE benchmark zip:

```bash
python scripts/download_ade20k.py \
  --output-dir data/ade20k \
  --archive-dir data/downloads/ade20k
```

Run the benchmark:

```bash
layerforge benchmark-ade20k \
  --dataset-dir data/ade20k \
  --output-dir results/ade20k_mask2former_512 \
  --config configs/ade20k_mask2former.yaml \
  --segmenter mask2former \
  --device cuda \
  --max-images 512 \
  --seed 7
```

Measured result already in the repo:

| Split | Segmenter | Images | Group mIoU | Thing mIoU | Stuff mIoU | Mean image mIoU | Mean predicted segments |
|---|---|---:|---:|---:|---:|---:|---:|
| ADE20K validation sample | mask2former (ADE) | 512 | 0.6015 | 0.5579 | 0.6451 | 0.5569 | 6.25 |

Per-group IoU from `report_artifacts/metrics_snapshots/ade20k_group_benchmark_summary.json`:

| Group | IoU |
|---|---:|
| animal | 0.3710 |
| building | 0.7757 |
| furniture | 0.6403 |
| ground | 0.5105 |
| object | 0.4669 |
| person | 0.5412 |
| plant | 0.6387 |
| road | 0.8300 |
| sky | 0.8455 |
| stuff | 0.3504 |
| vehicle | 0.6896 |
| water | 0.5585 |

Interpretation:

- COCO and ADE20K are now both covered by the same coarse-group external benchmark path;
- ADE20K is a stronger background/scene-structure benchmark than COCO for this project;
- these public benchmarks validate visible grouping only, while synthetic LayerBench remains the source of truth for full-layer metrics like order and recomposition under known ground truth.

The repo now also includes a public depth benchmark on **DIODE validation**. This benchmark is complementary to COCO and ADE20K: it scores the depth subsystem directly on a public RGB-D dataset with both indoor and outdoor scenes.

Download the official validation tarball:

```bash
python scripts/download_diode_val.py \
  --output-dir data/diode \
  --archive-dir data/downloads/diode
```

Run the honest metric-depth evaluation:

```bash
layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_depthpro_full \
  --config configs/diode_depthpro.yaml \
  --depth depth_pro \
  --device cuda \
  --seed 7
```

Run the apples-to-apples relative comparison against the geometric fallback:

```bash
layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_geometric_full \
  --config configs/diode_depthpro.yaml \
  --depth geometric_luminance \
  --alignment scale \
  --device cpu \
  --seed 7

layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_depthpro_scale_full \
  --config configs/diode_depthpro.yaml \
  --depth depth_pro \
  --alignment scale \
  --device cuda \
  --seed 7
```

Measured full-split results already in the repo:

| Variant | Alignment | Images | AbsRel | RMSE | delta1 | SILog |
|---|---|---:|---:|---:|---:|---:|
| geometric luminance | scale | 771 | 0.6298 | 7.0934 | 0.2714 | 184.6629 |
| depth_pro | none | 771 | 0.5230 | 29.0380 | 0.4057 | 26.8766 |
| depth_pro | scale | 771 | 0.3629 | 6.1891 | 0.6452 | 26.8766 |

Indoor/outdoor `AbsRel` split:

| Variant | Alignment | Indoor AbsRel | Outdoor AbsRel |
|---|---|---:|---:|
| geometric luminance | scale | 0.4822 | 0.7373 |
| depth_pro | none | 0.1995 | 0.7588 |
| depth_pro | scale | 0.0880 | 0.5632 |

Interpretation:

- raw `depth_pro` is the honest metric-depth result and is much stronger indoors than outdoors on DIODE;
- for a fair shape-only comparison, scale-aligned `depth_pro` beats the geometric fallback on `AbsRel`, `RMSE`, `delta1`, and `SILog`;
- DIODE now gives the repo a real public depth benchmark, not just synthetic ordering supervision.

## Learned ordering experiment

The repo now includes a lightweight pairwise ranker trained on the synthetic benchmark. This lets you compare hand-built boundary ordering against a learned near/far scorer without introducing a heavy external training stack.

Train the ranker:

```bash
layerforge train-ranker \
  --dataset-dir data/synth_train \
  --output models/order_ranker_fast.json \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```

Run the held-out evaluation:

```bash
layerforge benchmark \
  --dataset-dir data/synth_test \
  --output-dir results/synth_boundary_test \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance \
  --ordering boundary

layerforge benchmark \
  --dataset-dir data/synth_test \
  --output-dir results/synth_learned_test \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance \
  --ordering learned \
  --ranker-model models/order_ranker_fast.json
```

Held-out comparison:

| Split | Ordering | Mean best IoU | PLOA | Recompose PSNR |
|---|---|---:|---:|---:|
| synth test | boundary | 0.1549 | 0.1667 | 19.1589 |
| synth test | learned ranker | 0.1549 | 0.1667 | 19.4138 |

Interpretation: the learned ranker improves recomposition PSNR by about `+0.255` dB on held-out synthetic scenes, but it does not improve pairwise layer-order accuracy yet because the classical proposal stage still over-segments aggressively. That is a good, honest ablation result to report.

## Main outputs

Every run (whether `run`, `enrich-qwen`, or a per-scene benchmark run) produces the same set of artifacts:

```text
layers_ordered_rgba/       # near → far layer stack
layers_grouped_rgba/       # semantic/depth grouped layers
layers_albedo_rgba/        # per-layer albedo approximation
layers_shading_rgba/       # per-layer shading approximation
layers_amodal_masks/       # estimated amodal support masks
debug/depth_gray.png
debug/segmentation_overlay.png
debug/layer_graph.json
debug/background_completion.png
debug/parallax_preview.gif
manifest.json
metrics.json
```

## Final project thesis

LayerForge-X is not trying to be "another Qwen." Qwen-Image-Layered is a genuinely strong generative RGB→RGBA decomposer, and I'm not going to win a pixel-perfect generation contest against it. What LayerForge-X offers instead is an interpretable, depth-aware, benchmarked layer representation — one that adds explicit near/far ordering, occlusion edges, amodal support, intrinsic appearance factors, and component-level evaluation.

So the honest framing is: use Qwen as

1. a baseline to compare against,
2. a proposal source when its generative layers are better than classical segmentation,
3. a hybrid partner: Qwen layers + LayerForge graph enrichment.

That positioning is, I think, the strongest defense for grading.

To make the multi-image comparison reproducible rather than ad hoc, the repo now includes:

```bash
python scripts/run_curated_comparison.py \
  --input-dir data/qualitative_pack \
  --output-root runs/curated_comparison \
  --native-config configs/cutting_edge.yaml \
  --native-segmenter grounded_sam2 \
  --native-depth depth_pro \
  --qwen-layers 3,4,6,8
```

This writes a per-image comparison tree covering native LayerForge, raw Qwen, `Qwen + graph preserve`, and `Qwen + graph reorder`, plus aggregate `comparison_summary.json` and `comparison_summary.md`.

## Public benchmark roadmap

The current public-benchmark status and links for the next datasets are tracked in [docs/PUBLIC_BENCHMARKS_2026_04_22.md](docs/PUBLIC_BENCHMARKS_2026_04_22.md).
