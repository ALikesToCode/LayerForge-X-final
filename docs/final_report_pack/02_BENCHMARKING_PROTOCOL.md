# 5. Experiments and Evaluation Protocol

## Goal

The project is evaluated across multiple measurable properties rather than through a single qualitative comparison. A layered representation can fail in several distinct ways:

1. Are the layer regions semantically correct?
2. Is the near-to-far depth and occlusion order correct?
3. Do the layers recompose back into the original image?
4. Are alpha boundaries usable for editing?
5. Are hidden and background regions completed plausibly?
6. Does the intrinsic split behave sensibly?
7. Does the representation actually support edits better than baselines?

Because any of those can be wrong while the others look fine, the benchmark runs on multiple tracks.

For the present repository state, treat `PROJECT_MANIFEST.json`, `report_artifacts/metrics_snapshots/*.json`, and `report_artifacts/command_log.md` as the canonical evidence pack for reported numbers. `docs/RESULTS_SUMMARY_CURRENT.md` provides a prose summary of those artifacts.

---

# Track A: Modal semantic / panoptic segmentation

## Datasets

- **COCO Panoptic** for common objects and stuff.
- **ADE20K** for dense scene parsing and a broader set of stuff classes.
- Optional: a small hand-labelled project test set for whichever domains the demos lean on.

## Metrics

### Panoptic Quality

When ground-truth panoptic annotations are available, report PQ:

```text
PQ = sum_{(p,g) in TP} IoU(p,g) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Also report the two components:

```text
SQ = segmentation quality over matched segments
RQ = recognition quality
```

### Semantic mIoU

For semantic-only labels, the standard per-class IoU averaged over classes:

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU = mean_c IoU_c
```

## Baselines

| Baseline | Purpose |
|---|---|
| SLIC / classical connected components | Low-level non-semantic baseline |
| Mask2Former panoptic | strong closed-set panoptic baseline |
| GroundingDINO + SAM2 | open-vocabulary promptable baseline |
| Florence-2 + SAM2, optional | prompt-conditioned alternative |

## Report table

| Method | Dataset | group mIoU ↑ | thing mIoU ↑ | stuff mIoU ↑ | mean image mIoU ↑ | Avg layers | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | COCO-val subset | | | | | | |
| Mask2Former | COCO-val subset | | | | | | |
| Grounded-SAM2 | curated prompts | | | | | | |
| LayerForge-X | mixed | | | | | | |

For the present repository state, note the distinction clearly:

- the implemented COCO and ADE20K evaluators are **coarse-group IoU** benchmarks rather than official PQ pipelines;
- PQ/SQ/RQ remain reserved for a future full panoptic evaluator;
- do not relabel the current JSON summaries as PQ.

---

# Track B: Depth-order and occlusion-graph quality

## Why this matters

The output is a stack, so order is as important as mask quality. Get the ordering wrong and recomposition breaks, parallax looks incoherent, and the whole representation stops being useful even if each individual mask is perfect. Global average-depth sorting fails when objects are large, slanted, or span multiple depth planes — exactly the cases most scenes contain — so evaluation has to measure pairwise ordering, not just a global ranking.

## Datasets

- **Synthetic-LayerBench**: generated composites with known z-order.
- **NYU Depth V2**: indoor RGB-D scenes with object and instance labels.
- **DIODE**: indoor / outdoor RGB-D for generalisation.
- Optional: KITTI for outdoor road scenes if vehicles and roads are a focus.

## Ground-truth pair construction

For each image, define a ground-truth depth value per layer using median ground-truth depth inside the visible mask:

```text
z_i = median(Depth_GT[p] for p in visible_mask_i)
```

Then, for each candidate pair `(i, j)`, include it only if:

```text
|z_i - z_j| > tau_depth
```

That threshold rejects near-ties, which otherwise get penalised as "wrong" even when the order is genuinely ambiguous.

## Metrics

### Pairwise Layer Order Accuracy (PLOA)

```text
PLOA = (# correctly ordered valid pairs) / (# valid pairs)
```

A pair is correct if:

```text
sign(pred_depth_i - pred_depth_j) == sign(gt_depth_i - gt_depth_j)
```

Stick to one depth convention consistently: here, smaller depth means nearer when using metric depth.

### Boundary-Weighted PLOA (BW-PLOA)

Weight pairs by shared boundary length or adjacency confidence, so pairs that actually touch count more than pairs sitting in different corners of the image:

```text
BW-PLOA = sum_{i,j} w_ij * correct_ij / sum_{i,j} w_ij
```

Recommended weight:

```text
w_ij = shared_boundary_length(i,j) * min(area_i, area_j)^0.5
```

### Occlusion Edge F1

Build a ground-truth occlusion graph from synthetic z-order or RGB-D boundary reasoning. Then compare predicted graph edges as a set:

```text
Precision = correct_pred_edges / pred_edges
Recall    = correct_pred_edges / gt_edges
F1        = 2PR / (P + R)
```

### Kendall tau / inversion count

For images with a total ground-truth layer order, also report Kendall tau or the normalised inversion count, which provides a compact summary when the order is fully defined.

## Report table

| Method | Depth source | Ordering rule | PLOA ↑ | BW-PLOA ↑ | Occlusion F1 ↑ | Kendall τ ↑ |
|---|---|---|---:|---:|---:|---:|
| No depth | none | layer area / heuristic | | | | |
| Luminance depth | grayscale | global median | | | | |
| Depth Anything V2 | monocular | global median | | | | |
| Depth Pro | metric mono | global median | | | | |
| LayerForge-X | ensemble | boundary graph | | | | |

---

# Track C: RGBA recomposition fidelity

## Why this matters

If the exported layers are correct, alpha-compositing them in predicted order should recover the input image closely. This track is essentially a sanity check on the representation as a whole.

## Rendering equation

For layers composited far-to-near:

```text
C_out = alpha_over(L_1, L_2, ..., L_K)
```

Equivalently: start from the farthest layer and alpha-over each nearer layer on top.

## Metrics

### Pixel reconstruction

```text
MAE  = mean(|I - I_hat|)
MSE  = mean((I - I_hat)^2)
PSNR = 20 log10(MAX_I / sqrt(MSE))
```

### Structural similarity

SSIM or MS-SSIM.

### Perceptual similarity

LPIPS, when available. Lower values are better.

### Alpha coverage error

Compare the summed alpha against the valid image area. For opaque natural images, summed alpha should cover the whole scene:

```text
coverage_error = mean(|clip(sum_k alpha_k, 0, 1) - 1|)
```

## Report table

| Method | Hard/soft alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Alpha coverage err ↓ | Edge artifacts ↓ |
|---|---|---:|---:|---:|---:|---:|
| Hard masks | hard | | | | | |
| Feathered masks | soft | | | | | |
| Depth-aware alpha | soft | | | | | |
| Matting backend | soft | | | | | |

---

# Track D: Amodal mask and hidden-region completion

## Datasets

- **Synthetic-LayerBench**: exact full masks and hidden pixels are known.
- **KINS**: driving-scene amodal instance segmentation.
- **COCOA / COCO-A**: amodal object annotations where available.
- **MP3D-Amodal**, if accessible, for real indoor amodal masks from 3D data.

## Metrics

### Modal IoU — visible mask quality

```text
IoU_visible = |M_pred_visible ∩ M_gt_visible| / |M_pred_visible ∪ M_gt_visible|
```

### Amodal IoU — full object extent quality

```text
IoU_amodal = |M_pred_amodal ∩ M_gt_amodal| / |M_pred_amodal ∪ M_gt_amodal|
```

### Invisible-region IoU

The hardest and most informative of the three, because it isolates just the hidden portion:

```text
M_invisible = M_amodal - M_visible
IoU_invisible = IoU(M_pred_invisible, M_gt_invisible)
```

### Background-completion quality

On synthetic composites where the clean background is known:

```text
PSNR_masked, SSIM_masked, LPIPS_masked
```

These are computed only inside the removed or hidden region.

## Report table

| Method | Amodal module | Inpaint module | Visible IoU ↑ | Amodal IoU ↑ | Invisible IoU ↑ | Masked LPIPS ↓ |
|---|---|---|---:|---:|---:|---:|
| Visible masks only | none | none | | | | |
| Heuristic expansion | shape prior | OpenCV | | | | |
| SAMEO-style | amodal model | OpenCV | | | | |
| Full LayerForge-X | amodal + depth | LaMa/diffusion | | | | |

---

# Track E: Intrinsic albedo/shading split

## Datasets

- **IIW** for reflectance-order judgments and WHDR.
- **Synthetic-LayerBench-Intrinsic** with known albedo and shading.
- Optional: MIT Intrinsic Images if a small controlled set is enough.

## Metrics

### WHDR on IIW

WHDR measures whether predicted reflectance comparisons agree with human judgments. Lower is better.

### Synthetic intrinsic metrics

If ground-truth albedo `A` and shading `S` are available:

```text
MSE_albedo
MSE_shading
Scale-invariant MSE
Layer-local color constancy error
```

### Recomposition consistency

Per layer:

```text
I_layer ≈ A_layer * S_layer + residual_layer
```

Scored as:

```text
intrinsic_recompose_error = mean(|I_layer - A_layer * S_layer| inside alpha > 0)
```

## Report table

| Method | WHDR ↓ | Albedo MSE ↓ | Shading MSE ↓ | Recompose error ↓ | Notes |
|---|---:|---:|---:|---:|---|
| Retinex fallback | | | | | fast but approximate |
| Marigold-IID | | | | | stronger external backend |
| LayerForge-X per-layer | | | | | mask-aware export |

---

# Track F: Editability evaluation

## Why this matters

The principal motivation for layered representations is editing. The evaluation should therefore demonstrate that the representation supports practical operations rather than only scoring well on segmentation benchmarks.

## Edits to evaluate

1. **Object removal** — remove one foreground layer and show the completed background.
2. **Object translation** — move one layer sideways and see whether background holes stay plausible.
3. **Parallax preview** — shift layers according to depth to simulate viewpoint change.
4. **Depth-of-field edit** — blur far layers more than near layers.
5. **Albedo recolour** — recolour an object through its albedo while preserving shading.
6. **Relighting-lite** — scale or alter the shading layer.

## Metrics

### Non-edited region preservation

Outside the edit mask, the image should remain unchanged:

```text
preservation_MAE = mean(|I_original - I_edited| outside affected_region)
```

### Hole artifact ratio

After movement or removal, count transparent or invalid pixels:

```text
hole_ratio = invalid_pixels / image_pixels
```

### User preference study

A small study is plenty:

- 10 to 20 images.
- 3 methods: baseline, depth-aware only, full method.
- Question: "Which edit looks more plausible?"
- Report preference percentage.

Even a dozen people can surface systematic differences.

## Report table

| Method | Removal preference ↑ | Move preference ↑ | Parallax artifacts ↓ | Preservation MAE ↓ | Notes |
|---|---:|---:|---:|---:|---|
| Hard segmentation | | | | | jagged edges |
| Soft alpha only | | | | | better boundaries |
| Depth-aware + inpaint | | | | | fewer holes |
| Full LayerForge-X | | | | | best editability |

---

# Synthetic-LayerBench design

This is the easiest path to strong, defensible numbers, precisely because ground truth is known by construction.

## Data generation

Composite scenes from known layers:

```text
background B
for each layer k from far to near:
    choose object sprite / shape / cutout
    assign z_k
    assign semantic class
    assign alpha matte
    optionally assign albedo and shading
    composite using alpha-over
save:
    final RGB image
    each GT RGBA layer
    modal mask
    amodal/full mask
    alpha matte
    depth order
    clean background
    albedo and shading if available
    optional associated-effect layer
```

## Domains

At least three domains to avoid overfitting the benchmark to one look:

| Domain | Why it matters |
|---|---|
| Flat/vector graphics | crisp shapes, clear order |
| Photographic cutouts | realistic textures and boundaries |
| Stylized/anime | line art and cel shading |

## Recommended split

```text
train/dev for order ranker: 300 images
validation: 100 images
test: 100 images
```

If time is genuinely short:

```text
30 synthetic images + 20 real qualitative images
```

Even that is a big improvement over hand-picked demos alone.

## Rich synthetic export now implemented

The repository currently supports:

```bash
python scripts/make_synthetic_dataset.py \
  --output data/synthetic_layerbench_pp \
  --count 20 \
  --output-format layerbench_pp \
  --with-effects
```

Per scene, `layerbench_pp` writes:

```text
image.png
layers_near_to_far/
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

That format is the right one to use for recursive-peeling and effect-layer evaluation because it preserves both the visible scene and the hidden/effect supervision.

---

# Ablation protocol

Run a controlled set in which one component changes at a time. Only this type of comparison can attribute gains to a specific component.

| ID | Segmentation | Depth | Ordering | Alpha | Amodal | Inpaint | Intrinsics |
|---|---|---|---|---|---|---|---|
| A | SLIC | luminance | global median | hard | off | off | off |
| B | Mask2Former | none | heuristic | hard | off | off | off |
| C | Mask2Former | Depth Anything V2 | global median | hard | off | off | off |
| D | Mask2Former | Depth Anything V2 | boundary graph | hard | off | off | off |
| E | Grounded-SAM2 | Depth Anything V2 | boundary graph | soft | off | off | off |
| F | Grounded-SAM2 | Depth Pro/MoGe | boundary graph | soft | heuristic | OpenCV | off |
| G | Grounded-SAM2 | ensemble | learned edge ranker | soft/matting | amodal | LaMa | off |
| H | full | ensemble | learned edge ranker | soft/matting | amodal | LaMa | Retinex/Marigold-IID |
| I | full + peel | ensemble | graph-guided peeling | soft/matting | amodal | iterative completion | Retinex/Marigold-IID |

## Expected interpretation

- A → B measures the semantic segmentation benefit.
- B → C measures the depth benefit.
- C → D measures the boundary graph benefit on top of depth.
- D → E measures open-vocabulary plus soft alpha.
- E → F measures amodal and inpaint.
- G → H measures intrinsic split usefulness.
- H → I measures whether recursive peeling improves editability or hidden-region completion beyond the one-shot stack.

---

# Visual evidence set

The report documents the evaluation through the following figure classes:

1. Input image.
2. Semantic overlay.
3. Depth map.
4. Layer graph visualization.
5. Ordered RGBA contact sheet.
6. Hard-mask baseline vs soft-alpha result.
7. Global-depth ordering vs boundary-graph ordering.
8. Visible mask vs amodal mask.
9. Object removal with background completion.
10. Recursive peeling storyboard.
11. Parallax GIF or frame strip.
12. Albedo/shading layer visualisation.
13. Failure cases.

Primary tables:

1. Literature comparison table.
2. Benchmark/dataset table.
3. Ablation metrics table.
4. Runtime/memory table.
5. Failure-case taxonomy.

---

# Failure-case taxonomy

Failure analysis is part of the contribution. Classifying errors makes the evaluation more credible and easier to interpret:

| Failure | Cause | Example | Fix / future work |
|---|---|---|---|
| Wrong semantic grouping | segmenter misses object or merges stuff | chair merged with table | better prompts / panoptic model |
| Wrong depth order | monocular depth ambiguity | mirror/window/flat poster | boundary ranker + uncertainty |
| Jagged edge | hard mask or bad matting | hair/fur | matting backend |
| Missing shadow/effect | object-only mask | person moved without shadow | associated-effect layer |
| Bad inpainting | large unseen region | removed foreground person | diffusion/LaMa inpaint |
| Bad amodal shape | heavy occlusion | hidden car side | amodal model/SAMEO |
| Intrinsic artifacts | single-image ambiguity | texture mistaken as shading | stronger IID model |
| Too many layers | oversegmentation | background split into fragments | graph merging |
| Too few layers | undersegmentation | person + bicycle merged | prompt refinement |

---

# Recommended final benchmark narrative

If the report needs one paragraph summarising the whole evaluation, this is the one:

> We evaluate LayerForge-X across four axes: segmentation quality, layer-order correctness, recomposition fidelity, and editability. Standard panoptic metrics measure visible semantic grouping, while a synthetic layer benchmark and RGB-D datasets measure pairwise depth-order accuracy. Recomposition metrics verify that the exported RGBA stack preserves the original image. Finally, object removal, object movement, parallax, and intrinsic recolouring demonstrate that the representation is genuinely useful for editing rather than being a segmentation visualisation in disguise.
