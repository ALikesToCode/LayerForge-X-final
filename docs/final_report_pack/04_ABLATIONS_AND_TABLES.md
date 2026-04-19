# Ablations, Tables, and Figures for LayerForge-X

## Main ablation matrix

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

---

## Table 1: Literature comparison

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

| Dataset | Used for | Ground truth available | Metrics |
|---|---|---|---|
| Synthetic-LayerBench | full pipeline | RGBA layers, z-order, masks, clean background, optional albedo/shading | PLOA, PSNR, SSIM, LPIPS, alpha MAE, amodal IoU |
| COCO Panoptic | semantic/panoptic grouping | panoptic masks | PQ, SQ, RQ, mIoU |
| ADE20K | scene/stuff parsing | semantic masks | mIoU, pixel accuracy |
| NYU Depth V2 | indoor depth order | RGB-D, labels | AbsRel, RMSE, PLOA |
| DIODE | indoor/outdoor depth | RGB-D | AbsRel, RMSE, PLOA |
| KINS / COCOA | amodal segmentation | amodal masks | modal IoU, amodal IoU, invisible IoU |
| IIW | intrinsic decomposition | reflectance comparisons | WHDR |
| Real curated set | qualitative editing | no full GT | visual comparison, user preference |

---

## Table 3: Main quantitative results template

| Method | PQ ↑ | PLOA ↑ | BW-PLOA ↑ | Recon PSNR ↑ | Recon SSIM ↑ | LPIPS ↓ | Amodal IoU ↑ | Runtime ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical baseline | | | | | | | | |
| Panoptic only | | | | | | | | |
| Panoptic + depth | | | | | | | | |
| Panoptic + depth + graph | | | | | | | | |
| Open-vocab + depth + graph | | | | | | | | |
| Full LayerForge-X | | | | | | | | |

---

## Table 4: Component ablation template

| Removed component | Expected damage | Metric most affected | Observed result |
|---|---|---|---|
| remove semantic segmenter | no meaningful object layers | PQ/mIoU | |
| remove depth | wrong near/far order | PLOA/BW-PLOA | |
| remove boundary graph | large stuff/object order errors | BW-PLOA/Occlusion F1 | |
| remove soft alpha | jagged boundaries | alpha MAE/recomposition edge error | |
| remove amodal masks | no hidden object support | amodal IoU/editing score | |
| remove inpainting | holes after edits | masked LPIPS/hole ratio | |
| remove intrinsic split | no recoloring/shading control | WHDR/edit demo | |

---

## Table 5: Failure analysis template

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

Show:

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

Draw graph nodes as layers and arrows as “occludes” edges.

## Figure 3: Ordered layer contact sheet

Show all layers near → far.

## Figure 4: Baseline vs full

Columns:

```text
Input | hard masks | semantic only | depth-aware | full LayerForge-X
```

Rows:

```text
person street
indoor furniture
vehicle scene
animal scene
anime/vector scene
```

## Figure 5: Occlusion graph improvement

Show one case where global median depth fails but boundary graph succeeds.

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

Do not skip this. It makes the report look honest and mature.

---

# Presentation pitch

Use this slide sequence:

1. Problem: raster images are entangled.
2. Motivation: editing needs layers.
3. Related work map: LDI → 3D photo → panoptic/open-vocab → modern layer decomposition.
4. Key idea: Depth-Aware Amodal Layer Graph.
5. Pipeline.
6. Boundary-weighted occlusion graph.
7. Outputs.
8. Benchmark tracks.
9. Quantitative results.
10. Qualitative edits.
11. Ablations.
12. Failure cases.
13. Conclusion.

---

# One-minute oral explanation

Single images are hard to edit because all scene elements are fused into one RGB canvas. Our system converts that canvas into a structured layer graph. Each node is a semantic RGBA layer with visible mask, soft alpha, depth statistics, estimated amodal extent, and optional albedo/shading. Edges encode which layers occlude others. Instead of simply sorting masks by average depth, we build a boundary-weighted occlusion graph using local depth evidence near object contacts. This produces a near-to-far stack that can be recomposed, edited, inpainted, and animated. We evaluate the method not only with segmentation metrics but also with layer-order accuracy, recomposition error, amodal completion quality, and editing demonstrations.
