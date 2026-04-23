# Qwen-Image-Layered Comparison

Submission note: the raw local `runs/` and `data/` directories used for these experiments are omitted from the submission ZIP. The shipped source-of-truth artifacts for the measured comparison are `PROJECT_MANIFEST.json` plus `report_artifacts/metrics_snapshots/qwen_baseline_metrics.json`, `qwen_enriched_metrics.json`, and `qwen_five_image_review_summary.json`.

## Why Qwen matters

Qwen-Image-Layered targets the same broad endpoint as LayerForge-X: semantically meaningful RGBA layers from one image, produced by a large end-to-end generative model. LayerForge-X therefore treats Qwen as a **frontier baseline** and an optional **proposal source** rather than ignoring it.

As of April 2026, the official public baseline is the Apache-2.0 release `Qwen/Qwen-Image-Layered` together with the paper *Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition* (arXiv `2512.15603`, published December 17, 2025).

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

Short version: Qwen is still the clean frontier generative baseline, while LayerForge-X is the stronger structured pipeline after the recent proposal and merge upgrades.

## Recommended experiment

Recommended sweep:

```text
M0: classical + luminance depth
M1: Mask2Former + global median depth
M2: Mask2Former + boundary graph
M3: GroundingDINO + SAM2 + boundary graph
M4: full LayerForge-X
Q: Qwen-Image-Layered
Q+G: Qwen layers + LayerForge graph enrichment
```

M0 through M4 form a clean ablation ladder. `Q` is the generative frontier baseline. `Q+G` is the most informative hybrid row because it isolates the contribution of external generative layers from the contribution of explicit graph enrichment.

## How to run Qwen enrichment

Official baseline export:

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

After exporting Qwen RGBA layers into a folder, LayerForge enrichment is:

```bash
.venv/bin/layerforge enrich-qwen \
  --input data/demo/truck.jpg \
  --layers-dir runs/qwen_truck_layers_raw_640_20 \
  --output runs/qwen_truck_enriched_640_20 \
  --config configs/cutting_edge.yaml \
  --depth depth_pro \
  --preserve-external-order
```

That produces the same set of artifacts as a normal run:

```text
layers_ordered_rgba/
layers_albedo_rgba/
layers_shading_rgba/
layers_amodal_masks/
debug/layer_graph.json
metrics.json
manifest.json
```

For the multi-image review run, use the harness rather than isolated manual commands:

```bash
python scripts/run_curated_comparison.py \
  --input-dir data/qualitative_pack \
  --output-root runs/curated_comparison \
  --native-config configs/cutting_edge.yaml \
  --native-segmenter grounded_sam2 \
  --native-depth depth_pro \
  --qwen-layers 3,4,6,8
```

That writes both the per-image run trees and an aggregate `comparison_summary.json` / `comparison_summary.md`.

## Measured five-image review

The same-image five-image comparison was measured locally, and the submission archive ships its summary in `report_artifacts/metrics_snapshots/qwen_five_image_review_summary.json`. The review covers:

- `data/demo/truck.jpg`
- `data/qualitative_pack/astronaut.png`
- `data/qualitative_pack/coffee.png`
- `data/qualitative_pack/chelsea_cat.png`
- `examples/synth/scene_000/image.png`

Each image now has four directly comparable rows:

- `LayerForge native`: native grounded-SAM2 plus ensemble-depth pipeline;
- `Qwen raw (4)`: direct RGBA export, scored by the better of manifest or reversed-manifest interpretation;
- `Qwen + graph preserve (4)`: the same external stack with LayerForge graph, amodal, and intrinsic metadata while preserving Qwen's interpreted visual order;
- `Qwen + graph reorder (4)`: the same external stack exported in graph order.

In this repo, `preserve-external-order` means **preserve the best external visual stack chosen from Qwen's own candidate orderings** (manifest order versus reversed-manifest order), rather than replacing that stack with LayerForge's depth-graph order.

Aggregate mean results over the five images:

| Method | Images | Graph | Mean PSNR | Mean SSIM | Mean amodal extra ratio |
|---|---:|---|---:|---:|---:|
| LayerForge native | 5 | yes | 27.3438 | 0.9464 | 0.3057 |
| Qwen raw (4) | 5 | no | 29.0757 | 0.8850 | 0.0000 |
| Qwen + graph preserve (4) | 5 | yes | 28.5539 | 0.8638 | 2.9970 |
| Qwen + graph reorder (4) | 5 | yes | 28.5397 | 0.8637 | 2.9970 |

Per-image takeaways:

- `truck`: native LayerForge narrowly exceeds raw Qwen on PSNR and records a substantially higher SSIM; the preserve-order hybrid remains the most direct structured comparison row.
- `astronaut`: raw Qwen keeps the best PSNR, while both hybrid modes slightly improve SSIM over raw.
- `coffee`: native LayerForge is the strongest row on both PSNR and SSIM.
- `chelsea_cat`: raw Qwen keeps the best PSNR, while native LayerForge has the best SSIM.
- `synth image`: raw Qwen keeps the best PSNR, while native LayerForge has the best SSIM.

Interpretation:

- raw Qwen still wins mean PSNR on this five-image review, which keeps it as the clean compact generative baseline;
- native LayerForge now wins mean SSIM on the same image set, but it does so with a much larger average stack (`16.6` layers versus Qwen's `4`);
- `Qwen + graph preserve` is now the honest metadata-first hybrid row, because it keeps the best external visual stack while adding graph, amodal, and intrinsic structure;
- `Qwen + graph reorder` is the graph-altered export row, and its mean metrics are only slightly below the preserve-order variant on this review.

## What the hybrid row means

The `Q+G preserve` row should not be described as “our model with Qwen.” A more precise description is:

> Qwen provides the initial semantically disentangled RGBA layers, while LayerForge-X adds explicit depth estimation, graph edges, amodal support, and intrinsic appearance layers without changing the best interpreted external visual stack.

The `Q+G reorder` row answers a different question:

> What happens if the same Qwen layers are exported in the order preferred by the LayerForge depth graph?

That phrasing makes the complementarity obvious and avoids confusing the baseline with the contribution.

## Measured truck comparison

Both the original Qwen comparison and the upgraded native LayerForge truck runs on `data/demo/truck.jpg` were measured locally. The submission archive ships their summaries through the copied metric snapshots and the manifest.

| Run | Layers | PSNR | SSIM | Notes |
|---|---:|---:|---:|---|
| `runs/qwen_truck_layers_raw_640_20` | 4 | 29.8806 | 0.8826 | official Qwen raw RGBA export; best reconstruction obtained by scoring both manifest and reversed-manifest interpretations |
| `runs/qwen_truck_enriched_640_20` | 4 | 27.4633 | 0.7949 | fair preserve-order Qwen + LayerForge graph enrichment with layer merging disabled |
| `runs/demo_grounded_depthpro_final` | 45 | 14.6477 | 0.8348 | old native LayerForge decomposition before the new merge/depth recipe |
| `runs/truck_best_score` | 26 | 30.8214 | 0.9812 | improved native LayerForge with Gemini-assisted prompting |
| `runs/truck_best_score_manual` | 19 | 31.3040 | 0.9813 | highest measured native LayerForge run with curated prompts |
| `runs/truck_candidate_search_v2/best` | 20 | 32.1053 | 0.9848 | autotune-selected reproducible winner from the native candidate ladder |

Interpretation:

- the old native LayerForge run clearly lost to Qwen on this image;
- the `Q+G` hybrid adds graph, amodal, and intrinsic structure to the raw Qwen layers, but on this truck run it scores below raw Qwen on PSNR and SSIM; its value is structural enrichment rather than raw-fidelity gain;
- after the new proposal, depth, and merge upgrades, the best measured native LayerForge run now exceeds both Qwen rows on this truck example while staying interpretable and explicitly ordered;
- the strongest row is now the autotune-selected native LayerForge run, which formalises per-image recipe search instead of relying on a hand-picked single command;
- the honest comparative story is now stronger: Qwen remains the right frontier baseline, but LayerForge-X is no longer confined to "worse pixels, better metadata."

Companion figures:

- `docs/figures/truck_recomposition_comparison.png`
- `docs/figures/truck_layer_stack_comparison.png`
- `docs/figures/truck_metrics_comparison.png`
- `docs/figures/truck_prompt_ablation.png`

## Concise comparative statement

For a concise comparative statement:

> Qwen-Image-Layered is the strongest recent generative baseline for RGB-to-RGBA layer decomposition, and this project compares against it directly. LayerForge-X targets a different part of the problem: turning layer decomposition into an interpretable, benchmarkable scene representation. It adds explicit near-to-far ordering, pairwise occlusion graph edges, amodal visible/hidden masks, background-completion metadata, per-layer albedo/shading, and component-level metrics. Qwen can be used as a baseline or proposal source, while the main contribution remains the depth-aware amodal layer graph and the associated evaluation protocol.
