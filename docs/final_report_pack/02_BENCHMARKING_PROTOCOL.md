# Benchmarking Protocol for LayerForge-X

## Goal

The project should not be evaluated with one vague “looks good” comparison. A layered representation has several measurable properties:

1. Are the layer regions semantically correct?
2. Is the near-to-far depth/occlusion order correct?
3. Do the layers recompose back into the original image?
4. Are alpha boundaries usable for editing?
5. Are hidden/background regions completed plausibly?
6. Does the intrinsic split behave sensibly?
7. Does the representation actually support edits better than baselines?

The benchmark therefore uses multiple tracks.

---

# Track A: Modal semantic / panoptic segmentation

## Datasets

- **COCO Panoptic** for common objects and stuff.
- **ADE20K** for dense scene parsing and diverse stuff classes.
- Optional: a small manually labelled project test set for your chosen domains.

## Metrics

### Panoptic Quality

Use Panoptic Quality when ground-truth panoptic annotations are available:

```text
PQ = sum_{(p,g) in TP} IoU(p,g) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Also report:

```text
SQ = segmentation quality over matched segments
RQ = recognition quality
```

### Semantic mIoU

For semantic-only labels:

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

| Method | Dataset | PQ ↑ | SQ ↑ | RQ ↑ | mIoU ↑ | Avg layers | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | COCO-val subset | | | | | | |
| Mask2Former | COCO-val subset | | | | | | |
| Grounded-SAM2 | curated prompts | | | | | | |
| LayerForge-X | mixed | | | | | | |

---

# Track B: Depth-order and occlusion-graph quality

## Why this matters

The output is a stack, so order is as important as mask quality. Incorrect order creates broken recomposition and bad parallax. A global object depth can fail when objects are large, slanted, or touch multiple depth planes, so evaluate pairwise ordering.

## Datasets

- **Synthetic-LayerBench**: generated composites with known z-order.
- **NYU Depth V2**: indoor RGB-D scenes with object/instance labels.
- **DIODE**: indoor/outdoor RGB-D scenes for generalization.
- Optional: KITTI for outdoor road scenes if vehicles/roads are important.

## Ground-truth pair construction

For each image, define a ground-truth depth value per layer using median ground-truth depth inside the visible mask:

```text
z_i = median(Depth_GT[p] for p in visible_mask_i)
```

For pair `(i, j)`, include the pair only if:

```text
|z_i - z_j| > tau_depth
```

This avoids penalizing ambiguous near-ties.

## Metrics

### Pairwise Layer Order Accuracy, PLOA

```text
PLOA = (# correctly ordered valid pairs) / (# valid pairs)
```

A pair is correct if:

```text
sign(pred_depth_i - pred_depth_j) == sign(gt_depth_i - gt_depth_j)
```

Use the convention consistently: smaller depth means nearer if using metric depth.

### Boundary-Weighted Pairwise Layer Order Accuracy, BW-PLOA

Weight pairs by shared boundary length or adjacency confidence:

```text
BW-PLOA = sum_{i,j} w_ij * correct_ij / sum_{i,j} w_ij
```

Recommended weight:

```text
w_ij = shared_boundary_length(i,j) * min(area_i, area_j)^0.5
```

This focuses evaluation on pairs where ordering matters visually.

### Occlusion Edge F1

Build a ground-truth occlusion graph from synthetic z-order or RGB-D boundaries. Compare predicted graph edges:

```text
Precision = correct_pred_edges / pred_edges
Recall    = correct_pred_edges / gt_edges
F1        = 2PR / (P + R)
```

### Kendall tau / inversion count

For images with a total ground-truth layer order, report Kendall tau or normalized inversion count.

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

If the exported layers are correct, alpha-compositing them in predicted order should reconstruct the input image closely. This is a sanity check for the representation.

## Rendering equation

For layers ordered far-to-near during compositing:

```text
C_out = alpha_over(L_1, L_2, ..., L_K)
```

or equivalently build from the farthest layer first, then alpha-over nearer layers.

## Metrics

### Pixel reconstruction

```text
MAE  = mean(|I - I_hat|)
MSE  = mean((I - I_hat)^2)
PSNR = 20 log10(MAX_I / sqrt(MSE))
```

### Structural similarity

Use SSIM or MS-SSIM.

### Perceptual similarity

Use LPIPS if possible. Lower is better.

### Alpha coverage error

Compare the union of exported visible alphas against valid image area:

```text
coverage_error = mean(|clip(sum_k alpha_k, 0, 1) - 1|)
```

For opaque natural images, summed composited opacity should cover the image.

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

### Modal IoU

Visible mask quality:

```text
IoU_visible = |M_pred_visible ∩ M_gt_visible| / |M_pred_visible ∪ M_gt_visible|
```

### Amodal IoU

Full object extent quality:

```text
IoU_amodal = |M_pred_amodal ∩ M_gt_amodal| / |M_pred_amodal ∪ M_gt_amodal|
```

### Invisible-region IoU

Focus only on hidden area:

```text
M_invisible = M_amodal - M_visible
IoU_invisible = IoU(M_pred_invisible, M_gt_invisible)
```

This is harder and more informative than visible IoU.

### Background-completion quality

On synthetic composites where clean background is known:

```text
PSNR_masked, SSIM_masked, LPIPS_masked
```

Compute these only inside the removed/hidden region.

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

WHDR evaluates whether predicted reflectance comparisons agree with human judgments. Lower is better.

### Synthetic intrinsic metrics

If ground-truth albedo `A` and shading `S` are available:

```text
MSE_albedo
MSE_shading
Scale-invariant MSE
Layer-local color constancy error
```

### Recomposition consistency

For each layer:

```text
I_layer ≈ A_layer * S_layer + residual_layer
```

Score:

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

The project is about layers, not just masks. The best final section should demonstrate practical operations.

## Edits to evaluate

1. **Object removal**: remove one foreground layer and show completed background.
2. **Object translation**: move one layer sideways; check whether background holes are plausible.
3. **Parallax preview**: shift layers according to depth.
4. **Depth-of-field edit**: blur far layers more than near layers.
5. **Albedo recolor**: recolor an object through albedo while preserving shading.
6. **Relighting-lite**: scale shading layer or change shading contrast.

## Metrics

### Non-edited region preservation

Outside the edit mask, the image should remain unchanged:

```text
preservation_MAE = mean(|I_original - I_edited| outside affected_region)
```

### Hole artifact ratio

After movement/removal, measure transparent or invalid pixels:

```text
hole_ratio = invalid_pixels / image_pixels
```

### User preference study

Use a small study if possible:

- 10 to 20 images.
- 3 methods: baseline, depth-aware only, full method.
- Ask: “Which edit looks more plausible?”
- Report preference percentage.

## Report table

| Method | Removal preference ↑ | Move preference ↑ | Parallax artifacts ↓ | Preservation MAE ↓ | Notes |
|---|---:|---:|---:|---:|---|
| Hard segmentation | | | | | jagged edges |
| Soft alpha only | | | | | better boundaries |
| Depth-aware + inpaint | | | | | fewer holes |
| Full LayerForge-X | | | | | best editability |

---

# Synthetic-LayerBench design

This is the easiest way to get strong numbers because ground truth is known.

## Data generation

Create composites from known layers:

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
    depth order
    clean background
    albedo and shading if available
```

## Domains

Use at least three domains:

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

If time is short:

```text
30 synthetic images + 20 real qualitative images
```

Still better than only showing handpicked results.

---

# Ablation protocol

Run a controlled set where one component changes at a time.

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

## Expected interpretation

- A → B measures semantic segmentation benefit.
- B → C measures depth benefit.
- C → D measures boundary graph ordering benefit.
- D → E measures open-vocabulary and soft alpha benefit.
- E → F measures amodal/inpaint benefit.
- G → H measures intrinsic split usefulness.

---

# Minimum result set for best marks

Include these figures:

1. Input image.
2. Semantic overlay.
3. Depth map.
4. Layer graph visualization.
5. Ordered RGBA contact sheet.
6. Hard-mask baseline vs soft-alpha result.
7. Global-depth ordering vs boundary-graph ordering.
8. Visible mask vs amodal mask.
9. Object removal with background completion.
10. Parallax GIF or frame strip.
11. Albedo/shading layer visualization.
12. Failure cases.

Include these tables:

1. Literature comparison table.
2. Benchmark/dataset table.
3. Ablation metrics table.
4. Runtime/memory table.
5. Failure-case taxonomy.

---

# Failure-case taxonomy

Do not hide failures. Classify them.

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

The strongest narrative is:

> We evaluate LayerForge-X across four axes: segmentation quality, layer-order correctness, recomposition fidelity, and editability. Standard panoptic metrics measure visible semantic grouping, while a synthetic layer benchmark and RGB-D datasets measure pairwise depth-order accuracy. Recomposition metrics verify that the exported RGBA stack preserves the original image. Finally, object removal, object movement, parallax, and intrinsic recoloring demonstrate that the representation is useful for editing rather than being merely a segmentation visualization.
