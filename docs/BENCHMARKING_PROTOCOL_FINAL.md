# Benchmarking Protocol

Layer decomposition is multi-dimensional, and any single metric will lie to you. A method can nail semantic segmentation and still order the layers wrong. It can order the layers perfectly and still leave visible seams when you recompose them. Because of that, this protocol splits the evaluation into tracks that each stress a different part of the pipeline.

## Methods

| ID | Method | Purpose |
|---:|---|---|
| M0 | Classical/SLIC + luminance depth | weak deterministic baseline |
| M1 | Mask2Former + global median depth | strong modular segmentation baseline |
| M2 | Mask2Former + boundary graph | tests graph ordering |
| M3 | GroundingDINO + SAM2 + boundary graph | tests open-vocabulary control |
| M4 | GroundingDINO + SAM2 + learned ranker | tests trained ordering |
| M5 | Qwen-Image-Layered | frontier generative baseline |
| M6 | Qwen + LayerForge graph | hybrid baseline |
| M7 | Full LayerForge-X | final method |
| M8 | LayerForge peel | graph-guided recursive peeling variant |

## Dataset plan

Each dataset plays a specific role — don't mix their metrics:

| Dataset | Role |
|---|---|
| Synthetic-LayerBench | ground-truth masks, order, alpha, hidden regions |
| COCO Panoptic | coarse-group visible semantic grouping |
| ADE20K | coarse-group scene parsing and stuff/background |
| NYU Depth V2 | indoor RGB-D depth ordering |
| DIODE | indoor/outdoor depth |
| KINS / COCOA / MP3D-Amodal | amodal masks |
| IIW | intrinsic decomposition |
| curated images | final qualitative demos |

## Track A: segmentation

Standard visible-grouping metrics. The current repo implementation uses coarse-group IoU on COCO and ADE20K rather than official PQ:

```text
mIoU
pixel accuracy
thing mIoU
stuff mIoU
```

Illustrative summary table:

| Method | Dataset | group mIoU ↑ | thing mIoU ↑ | stuff mIoU ↑ | pixel acc ↑ | Avg layers | Runtime ↓ |
|---|---|---:|---:|---:|---:|---:|---:|

## Track B: depth/order

### Pairwise Layer Order Accuracy

Pairwise accuracy over valid layer pairs — the pair is included only if the depth gap exceeds a threshold, so near-ties don't penalise the method:

```text
PLOA = correct ordered valid layer pairs / valid layer pairs
```

### Boundary-Weighted PLOA

The weighting focuses evaluation on pairs that actually touch, because that's where ordering is visually consequential:

```text
BW-PLOA = Σ w_ij correct_ij / Σ w_ij
w_ij = shared_boundary_length(i,j) × sqrt(min(area_i, area_j))
```

### Occlusion Edge F1

Treat the occlusion graph edges as a set and compute the usual precision/recall:

```text
Precision = correct predicted edges / predicted edges
Recall    = correct predicted edges / ground-truth edges
F1        = 2PR / (P + R)
```

Illustrative summary table:

| Method | Depth | Ordering | PLOA ↑ | BW-PLOA ↑ | Edge F1 ↑ | Kendall τ ↑ |
|---|---|---|---:|---:|---:|---:|

## Track C: recomposition

Composite the output layers back and measure how close the reconstruction is to the input:

```text
Î = composite(L_near_to_far)
```

Metrics:

```text
MAE
PSNR
SSIM
LPIPS
alpha coverage error
```

| Method | Alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Coverage error ↓ |
|---|---|---:|---:|---:|---:|

## Track D: amodal and completion

This is the track most baselines can't compete on. The metrics split the evaluation between visible mask quality, full-extent amodal quality, and hidden-region completion quality:

```text
Visible IoU
Amodal IoU
Hidden-region IoU
Masked PSNR
Masked SSIM
Masked LPIPS
```

The hidden region itself is just the difference between the amodal and visible masks:

```text
M_hidden = M_amodal - M_visible
```

## Track E: intrinsic split

```text
WHDR
albedo MSE
shading MSE
intrinsic recomposition error
```

WHDR is only defined on IIW-style reflectance-judgement data; the rest can be computed on synthetic scenes where ground-truth albedo and shading are available.

## Track F: editing utility

The goal here is to demonstrate that the representation isn't just segmentation with extra metadata — it's actually useful for editing. Six operations to evaluate:

```text
object removal
object movement
parallax preview
recolor albedo
adjust shading
background blur
```

Metrics:

```text
non-edited region preservation MAE
hole ratio
boundary artifact score
manual preference score
runtime
memory
```

Some of these (preference score) require a small user study. Even 5 – 10 people with 10 – 20 scenes is enough to spot big differences.

## Lightweight benchmark included in repo

Because full-scale evaluation is heavy, the repo ships a lightweight synthetic benchmark that exercises the pipeline end-to-end on generated scenes:

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 20

layerforge benchmark \
  --dataset-dir data/synthetic_layerbench \
  --output-dir results/synthetic_fast \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```

For the richer submission/demo artifact set, generate one `layerbench_pp` scene pack as well:

```bash
python scripts/make_synthetic_dataset.py \
  --output data/synthetic_layerbench_pp \
  --count 20 \
  --output-format layerbench_pp \
  --with-effects
```

That second export is the right benchmark substrate for recursive peeling, associated-effect layers, and intrinsic-layer inspection.

These commands are the priority reproducibility harnesses for the current repository state. Any quantitative statement in the report should be reproducible from one of these evaluation paths.
