# Benchmarking Plan

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

Use `scripts/make_synthetic_dataset.py` to generate scenes with known RGBA layers and near-to-far order. This gives the full project a ground truth target, which real photos do not have.

## Qualitative figures

For each selected image show:

- input;
- segmentation overlay;
- depth map;
- layer contact sheet;
- completed background;
- recomposition;
- parallax GIF frame or edit;
- albedo/shading split.

## Ablation table

| Variant | Segmenter | Depth | Inpaint | Intrinsics | Expected lesson |
|---|---|---|---|---|---|
| A | SLIC | luminance | Telea | Retinex | baseline |
| B | Mask2Former | luminance | Telea | Retinex | semantics help |
| C | Mask2Former | Depth Anything V2 | Telea | Retinex | depth improves order |
| D | Grounded SAM2 | Depth Pro | LaMa | Retinex | masks + metric depth |
| E | Grounded SAM2 | ensemble | LaMa | Marigold-IID external | strongest |
