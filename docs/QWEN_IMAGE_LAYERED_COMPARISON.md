# Qwen-Image-Layered Comparison

## Why Qwen matters

Qwen-Image-Layered is the elephant in the room. It targets more or less the same endpoint LayerForge-X does — producing semantically meaningful RGBA layers from one image — and it does so with a big end-to-end generative model. Pretending it doesn't exist would be a weak way to write this project up. So instead, LayerForge-X treats it as a **frontier baseline** and an optional **proposal source**.

As of April 2026, the official public baseline is the Apache-2.0 release `Qwen/Qwen-Image-Layered` together with the paper *Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition* (arXiv `2512.15603`, published December 17, 2025).

## Difference in project positioning

The quick version of the comparison:

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

The sweep I'd actually run:

```text
M0: classical + luminance depth
M1: Mask2Former + global median depth
M2: Mask2Former + boundary graph
M3: GroundingDINO + SAM2 + boundary graph
M4: full LayerForge-X
Q: Qwen-Image-Layered
Q+G: Qwen layers + LayerForge graph enrichment
```

M0 through M4 gives you a clean ablation ladder. Q is the generative frontier. Q+G is the hybrid that — I suspect — is the most interesting single row in the table, because it shows what each system is actually contributing.

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
  --depth depth_pro
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

For the multi-image review run, use the harness instead of ad hoc one-off commands:

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

The repo now contains a real five-image comparison in `runs/qwen_five_image_review/` covering:

- `data/demo/truck.jpg`
- `data/qualitative_pack/astronaut.png`
- `data/qualitative_pack/coffee.png`
- `data/qualitative_pack/chelsea_cat.png`
- `examples/synth/scene_000/image.png`

Each image has both:

- `Qwen raw (4)`: direct RGBA export, scored by recomposing the manifest-ordered layers;
- `Qwen + graph (4)`: the same RGBA stack enriched with LayerForge depth ordering, graph edges, and amodal/intrinsic artifacts.

Aggregate mean results over the five images:

| Method | Images | Graph | Mean PSNR | Mean SSIM | Mean amodal extra ratio |
|---|---:|---|---:|---:|---:|
| Qwen raw (4) | 5 | no | 29.0757 | 0.8850 | 0.0000 |
| Qwen + graph (4) | 5 | yes | 28.5408 | 0.8637 | 2.9970 |

Per-image measured rows:

| Image | Qwen raw PSNR | Qwen raw SSIM | Qwen + graph PSNR | Qwen + graph SSIM | Graph output |
|---|---:|---:|---:|---:|---|
| truck | 28.8062 | 0.8614 | 26.8214 | 0.7826 | yes |
| astronaut | 29.4532 | 0.9091 | 29.3737 | 0.9153 | yes |
| coffee | 28.2737 | 0.8762 | 28.0333 | 0.8718 | yes |
| chelsea_cat | 29.5828 | 0.9050 | 29.4704 | 0.8918 | yes |
| synth image | 29.2627 | 0.8733 | 29.0052 | 0.8571 | yes |

Interpretation:

- on this five-image set, raw Qwen wins mean PSNR and mean SSIM;
- the hybrid row is still valuable because it adds explicit graph structure and amodal artifacts rather than only pixels;
- the honest framing is therefore not "hybrid beats raw Qwen visually", but "hybrid adds structure while remaining competitive on some images";
- the best hybrid qualitative case in this sweep is `astronaut`, where SSIM slightly improves while the graph becomes explicit.

## What the hybrid row means

The `Q+G` row in the report should not be described as "our model with Qwen." It is more precise to say:

> Qwen provides the initial semantically disentangled RGBA layers, while LayerForge-X adds explicit depth estimation, near-to-far ordering, graph edges, amodal support, and intrinsic appearance layers.

That phrasing makes the complementarity obvious and avoids confusing the baseline with the contribution.

## Measured truck comparison

The repo now contains both the original Qwen comparison and the upgraded native LayerForge truck runs on `data/demo/truck.jpg`:

| Run | Layers | PSNR | SSIM | Notes |
|---|---:|---:|---:|---|
| `runs/qwen_truck_layers_raw_640_20` | 4 | 26.7874 | 0.7723 | official Qwen raw RGBA export; best reconstruction obtained by interpreting manifest order as far-to-near |
| `runs/qwen_truck_enriched_640_20` | 2 | 27.4612 | 0.7953 | Qwen layers enriched with LayerForge depth ordering and graph metadata |
| `runs/demo_grounded_depthpro_final` | 45 | 14.6477 | 0.8348 | old native LayerForge decomposition before the new merge/depth recipe |
| `runs/truck_best_score` | 26 | 30.8214 | 0.9812 | improved native LayerForge with Gemini-assisted prompting |
| `runs/truck_best_score_manual` | 19 | 31.3040 | 0.9813 | highest measured native LayerForge run with curated prompts |
| `runs/truck_state_of_art_search_v2/best` | 20 | 32.1053 | 0.9848 | autotune-selected reproducible winner from the native candidate ladder |

Interpretation:

- the old native LayerForge run clearly lost to Qwen on this image;
- the `Q+G` hybrid still improves over the raw Qwen stack in both PSNR and SSIM;
- after the new proposal, depth, and merge upgrades, the best measured native LayerForge run now exceeds both Qwen rows on this truck example while staying interpretable and explicitly ordered;
- the strongest row is now the autotune-selected native LayerForge run, which formalises per-image recipe search instead of relying on a hand-picked single command;
- the honest comparative story is now stronger: Qwen remains the right frontier baseline, but LayerForge-X is no longer confined to "worse pixels, better metadata."

Companion figures:

- `docs/figures/truck_recomposition_comparison.png`
- `docs/figures/truck_layer_stack_comparison.png`
- `docs/figures/truck_metrics_comparison.png`
- `docs/figures/truck_prompt_ablation.png`

## Viva answer

In case someone in the viva asks "Isn't this just Qwen-Image-Layered?", the answer I'd give is roughly:

> Qwen-Image-Layered is the strongest recent generative baseline for RGB-to-RGBA layer decomposition, and I compare against it directly. My project targets a different part of the problem: turning layer decomposition into an interpretable, benchmarkable scene representation. LayerForge-X adds explicit near-to-far ordering, pairwise occlusion graph edges, amodal visible/hidden masks, background-completion metadata, per-layer albedo/shading, and component-level metrics. It can use Qwen as a baseline or a proposal source, but the actual contribution is the depth-aware amodal layer graph and the evaluation protocol built around it. After the latest upgrades, the native LayerForge recipe is also competitive on recomposition fidelity on the truck benchmark in this repo, not just on structure.
