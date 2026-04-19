# Benchmarking Protocol

Layer decomposition is multi-dimensional. One metric is not enough.

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

## Dataset plan

| Dataset | Role |
|---|---|
| Synthetic-LayerBench | ground-truth masks, order, alpha, hidden regions |
| COCO Panoptic | panoptic segmentation |
| ADE20K | scene parsing and stuff/background |
| NYU Depth V2 | indoor RGB-D depth ordering |
| DIODE | indoor/outdoor depth |
| KINS / COCOA / MP3D-Amodal | amodal masks |
| IIW | intrinsic decomposition |
| curated images | final qualitative demos |

## Track A: segmentation

Metrics:

```text
mIoU
pixel accuracy
PQ
SQ
RQ
```

Table:

| Method | Dataset | mIoU ↑ | PQ ↑ | SQ ↑ | RQ ↑ | Avg layers | Runtime ↓ |
|---|---|---:|---:|---:|---:|---:|---:|

## Track B: depth/order

### Pairwise Layer Order Accuracy

```text
PLOA = correct ordered valid layer pairs / valid layer pairs
```

### Boundary-Weighted PLOA

```text
BW-PLOA = Σ w_ij correct_ij / Σ w_ij
w_ij = shared_boundary_length(i,j) × sqrt(min(area_i, area_j))
```

### Occlusion Edge F1

```text
Precision = correct predicted edges / predicted edges
Recall    = correct predicted edges / ground-truth edges
F1        = 2PR / (P + R)
```

Table:

| Method | Depth | Ordering | PLOA ↑ | BW-PLOA ↑ | Edge F1 ↑ | Kendall τ ↑ |
|---|---|---|---:|---:|---:|---:|

## Track C: recomposition

Composite the output layers back:

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

Table:

| Method | Alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Coverage error ↓ |
|---|---|---:|---:|---:|---:|

## Track D: amodal and completion

Metrics:

```text
Visible IoU
Amodal IoU
Hidden-region IoU
Masked PSNR
Masked SSIM
Masked LPIPS
```

Hidden region:

```text
M_hidden = M_amodal - M_visible
```

## Track E: intrinsic split

Metrics:

```text
WHDR
albedo MSE
shading MSE
intrinsic recomposition error
```

## Track F: editing utility

Operations:

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

## Lightweight benchmark included in repo

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 20

layerforge benchmark \
  --dataset-dir data/synthetic_layerbench \
  --output-dir results/synthetic_fast \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```
