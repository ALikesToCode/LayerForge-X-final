# Benchmarking Plan

Here's the shape of what needs measuring, and where the data for each track comes from:

| Goal | Dataset | Metric |
|---|---|---|
| Panoptic segmentation | COCO Panoptic, ADE20K | PQ, SQ, RQ, mIoU |
| Open-vocab detection | LVIS, ODinW, custom prompts | AP, recall@K |
| Depth | NYU Depth V2, DIODE, KITTI | AbsRel, RMSE, delta thresholds |
| Layer ordering | synthetic composites | pairwise order accuracy |
| Alpha quality | matting benchmark or synthetic alpha | SAD, MSE, boundary F1 |
| Inpainting | masked Places/background completion | LPIPS, FID/KID, human preference |
| Intrinsics | IIW | WHDR |
| End-to-end | generated synthetic scenes | recomposition PSNR/SSIM, layer count, order accuracy |

## Synthetic benchmark

Real photos don't come with a z-order answer key, so the synthetic generator in `scripts/make_synthetic_dataset.py` does the heavy lifting for quantitative evaluation. Each generated scene comes with known RGBA layers, a known near-to-far order, and known hidden regions. That gives the project a ground-truth target that no real photo can provide.

## Qualitative figures

For the curated real-image qualitative set, each example should show:

- the input;
- segmentation overlay;
- depth map;
- layer contact sheet;
- completed background;
- the recomposition;
- a parallax GIF frame or another edit demo;
- the albedo/shading split.

That's a lot of panels per image, but it's the only way to make the components legible at a glance.

## Ablation table

| Variant | Segmenter | Depth | Inpaint | Intrinsics | Expected lesson |
|---|---|---|---|---|---|
| A | SLIC | luminance | Telea | Retinex | baseline |
| B | Mask2Former | luminance | Telea | Retinex | semantics help |
| C | Mask2Former | Depth Anything V2 | Telea | Retinex | depth improves order |
| D | Grounded SAM2 | Depth Pro | LaMa | Retinex | masks + metric depth |
| E | Grounded SAM2 | ensemble | LaMa | Marigold-IID external | strongest |
