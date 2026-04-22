# Ablations, Tables, and Figures for LayerForge-X

This document collects every planned ablation, every table the report needs, every figure the report needs, and the slide sequence for the presentation. It is the working document — fill values in as runs complete.

## Completed runs snapshot

The rows below are no longer placeholders; they correspond to runs already present in the repo:

| Variant | Segmentation | Depth | Ordering | Split | Mean best IoU | PLOA | Recompose PSNR |
|---|---|---|---|---|---:|---:|---:|
| A1 | classical | geometric luminance | boundary | synthetic fast | 0.1549 | 0.1667 | 19.1360 |
| A2 | classical | geometric luminance | boundary | synth test | 0.1549 | 0.1667 | 19.1589 |
| A3 | classical | geometric luminance | learned ranker | synth test | 0.1549 | 0.1667 | 19.4138 |

Interpretation:

- `A2 → A3` gives a real learned-ordering result worth reporting.
- the current bottleneck is proposal quality, because the fast classical segmenter still produces about `65` predicted layers for `5` ground-truth layers.
- therefore the strongest next qualitative row is the real-image `grounded_sam2 + depth_pro` system, not more tuning on the classical baseline.

## Main ablation matrix

The core sweep. Each row changes exactly one axis relative to the next so the contribution of each component is readable off the table:

| Variant | Segmentation | Depth | Ordering | Alpha | Amodal | Inpaint | Intrinsic | Purpose |
|---|---|---|---|---|---|---|---|---|
| A | SLIC/classical | luminance | global median | hard | no | no | no | weak baseline |
| B | Mask2Former | none | area/heuristic | hard | no | no | no | semantic-only baseline |
| C | Mask2Former | Depth Anything V2 | global median | hard | no | no | no | tests depth addition |
| D | Mask2Former | Depth Anything V2 | boundary graph | hard | no | no | no | tests graph ordering |
| E | Grounded-SAM2 | Depth Anything V2 | boundary graph | soft | no | no | no | tests promptable masks + alpha |
| F | Grounded-SAM2 | Depth Pro / MoGe | boundary graph | soft | heuristic | OpenCV | no | tests amodal + completion |
| G | Grounded-SAM2 | ensemble | learned edge ranker | soft/matting | amodal | LaMa | no | strong non-intrinsic system |
| H | full | ensemble | learned edge ranker | soft/matting | amodal | LaMa | Retinex / Marigold-IID | full LayerForge-X |
| I | full + peel | ensemble | graph-guided peeling | soft/matting | amodal | iterative completion | Retinex / Marigold-IID | recursive peeling variant |

---

## Table 1: Literature comparison

A gap analysis across the most relevant families of prior work. The `LayerForge-X` row is intentionally the most densely populated — that's the point:

| Method family | Semantic layers | Depth order | Amodal hidden parts | Soft alpha | Inpainting | Intrinsics | Single image | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LDI | no | yes | partial | no | no | no | yes/varies | rendering representation |
| 3D photo inpainting | no/limited | yes | yes | no | yes | no | RGB-D/depth | parallax focus |
| Panoptic segmentation | yes | no | no | no | no | no | yes | visible masks only |
| Grounded-SAM | yes/open vocab | no | no | no | no | no | yes | promptable visible masks |
| Matting | foreground only | no | no | yes | no | no | yes | excellent alpha boundaries |
| Amodal segmentation | object masks | limited | yes | no | sometimes | no | yes | hidden shape, not full layer stack |
| LayerDecomp-style | foreground/background | partial | yes | yes | yes | no | yes | strong generative editing baseline |
| Qwen-Image-Layered-style | yes | implicit | yes | yes | yes | no/limited | yes | end-to-end generative RGBA layers |
| LayerForge-X | yes | explicit graph | yes | yes | yes | optional | yes | modular and benchmarkable |

---

## Table 2: Dataset plan

Which dataset gets used for which track, with the available ground truth in each:

| Dataset | Used for | Ground truth available | Metrics |
|---|---|---|---|
| Synthetic-LayerBench / layerbench_pp | full pipeline | RGBA layers, z-order, masks, clean background, optional albedo/shading, optional effects | PLOA, PSNR, SSIM, LPIPS, alpha MAE, amodal IoU |
| COCO Panoptic | visible semantic grouping | panoptic masks | coarse-group mIoU, thing/stuff mIoU |
| ADE20K | scene/stuff parsing | semantic masks | coarse-group mIoU, pixel accuracy |
| NYU Depth V2 | indoor depth order | RGB-D, labels | AbsRel, RMSE, PLOA |
| DIODE | indoor/outdoor depth | RGB-D | AbsRel, RMSE, PLOA |
| KINS / COCOA | amodal segmentation | amodal masks | modal IoU, amodal IoU, invisible IoU |
| IIW | intrinsic decomposition | reflectance comparisons | WHDR |
| Real curated set | qualitative editing | no full GT | visual comparison, user preference |

---

## Table 3: Main quantitative results template

| Method | group mIoU ↑ | PLOA ↑ | BW-PLOA ↑ | Recon PSNR ↑ | Recon SSIM ↑ | LPIPS ↓ | Amodal IoU ↑ | Runtime ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical baseline | | | | | | | | |
| Panoptic only | | | | | | | | |
| Panoptic + depth | | | | | | | | |
| Panoptic + depth + graph | | | | | | | | |
| Open-vocab + depth + graph | | | | | | | | |
| Full LayerForge-X | | | | | | | | |

---

## Table 4: Component ablation template

A leave-one-out table. For each row, the expected-damage column says what I think should break, and the observed-result column is what the numbers actually say. If the observed column doesn't match the expected column, that's interesting and deserves a paragraph in the discussion.

| Removed component | Expected damage | Metric most affected | Observed result |
|---|---|---|---|
| remove semantic segmenter | no meaningful object layers | group mIoU | |
| remove depth | wrong near/far order | PLOA/BW-PLOA | |
| remove boundary graph | large stuff/object order errors | BW-PLOA/Occlusion F1 | |
| remove soft alpha | jagged boundaries | alpha MAE/recomposition edge error | |
| remove amodal masks | no hidden object support | amodal IoU/editing score | |
| remove inpainting | holes after edits | masked LPIPS/hole ratio | |
| remove intrinsic split | no recoloring/shading control | WHDR/edit demo | |
| remove recursive peeling | weaker hidden-region recovery in iterative scenes | edit demo / masked LPIPS | |

---

## Table 5: Failure analysis template

One row per failure example. Talking about failures explicitly is one of the easiest ways to make the report read as mature rather than salesy:

| Image | Failure type | Cause | Visible symptom | Fix/future work |
|---|---|---|---|---|
| image_01 | depth ambiguity | mirror/glass | wrong order | uncertainty + user correction |
| image_02 | mask merge | same-colored objects | two objects in one layer | prompt refinement |
| image_03 | alpha failure | hair/fur | jagged edge | matting backend |
| image_04 | inpaint failure | large hidden background | blurry fill | stronger diffusion inpaint |
| image_05 | amodal failure | extreme occlusion | wrong hidden shape | SAMEO/amodal backend |

---

# Required figures

## Figure 1: Pipeline diagram

A flow diagram in the order the pipeline runs:

```text
RGB input
→ semantic proposals
→ monocular depth
→ soft alpha
→ amodal masks
→ occlusion graph
→ completion
→ intrinsic split
→ RGBA layer stack
```

## Figure 2: DALG representation

A graph drawing. Nodes are layers; arrows are "occludes" edges. Use colour to distinguish semantic groups.

## Figure 3: Ordered layer contact sheet

All layers for a single scene, laid out near → far. This one sells the project visually better than any table.

Measured file available now:

```text
docs/figures/truck_layer_stack_comparison.png
```

## Figure 4: Baseline vs full

A comparison grid. Columns step through the key ablations:

```text
Input | hard masks | semantic only | depth-aware | full LayerForge-X
```

Rows cover a mix of domains so no single content type dominates:

```text
person street
indoor furniture
vehicle scene
animal scene
anime/vector scene
```

Measured file available now:

```text
docs/figures/truck_recomposition_comparison.png
```

## Figure 5: Occlusion graph improvement

Pick a single scene where the global-median-depth ordering fails and the boundary graph gets it right. Highlight the contested edge.

## Figure 6: Editing demo

Rows:

```text
object removal
object movement
parallax
albedo recolor
background blur
```

## Figure 7: Failure cases

Do not skip this. A report without a failure-cases figure looks like the author hid the hard examples.

Additional measured figure files already generated:

```text
docs/figures/truck_metrics_comparison.png
docs/figures/synthetic_ordering_ablation.png
docs/figures/qualitative_gallery.png
```

---

# Presentation pitch

Slide sequence for the talk:

1. Problem — raster images are entangled.
2. Motivation — editing needs layers.
3. Related work map — LDI → 3D photo → panoptic / open-vocab → modern layer decomposition.
4. Key idea — Depth-Aware Amodal Layer Graph.
5. Pipeline.
6. Boundary-weighted occlusion graph (the algorithmic contribution slide).
7. Outputs.
8. Benchmark tracks.
9. Quantitative results.
10. Qualitative edits.
11. Ablations.
12. Failure cases.
13. Conclusion.

---

# One-minute oral explanation

The short script if someone asks "what's your project?" at a poster session:

Single images are hard to edit because every scene element gets fused into one RGB canvas. My system converts that canvas into a structured layer graph. Each node is a semantic RGBA layer with a visible mask, a soft alpha, depth statistics, an estimated amodal extent, and optional albedo and shading. The edges record which layers occlude which. Instead of sorting masks by average depth, I build a boundary-weighted occlusion graph from local depth evidence near object contacts. That yields a near-to-far stack you can recompose, edit, inpaint, and animate. The evaluation doesn't stop at segmentation metrics — it also covers layer-order accuracy, recomposition error, amodal completion quality, and actual editing demonstrations.
