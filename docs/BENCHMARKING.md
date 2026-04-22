# Benchmarking Plan

Here's the shape of what needs measuring, and where the data for each track comes from:

| Goal | Dataset | Metric |
|---|---|---|
| Visible semantic grouping | COCO Panoptic, ADE20K | coarse-group mIoU, thing/stuff mIoU |
| Open-vocab detection | LVIS, ODinW, custom prompts | AP, recall@K |
| Depth | NYU Depth V2, DIODE, KITTI | AbsRel, RMSE, delta thresholds |
| Layer ordering | synthetic composites, NYU Depth V2 | pairwise order accuracy, occlusion-edge F1 |
| Alpha quality | synthetic alpha, generated layered scenes | SAD, MSE, boundary F1 |
| Inpainting | masked Places/background completion, synthetic hidden regions | LPIPS, FID/KID, human preference |
| Intrinsics | IIW | WHDR |
| End-to-end | generated synthetic scenes | recomposition PSNR/SSIM, layer count, order accuracy, edit success |

## Synthetic benchmark

Real photos don't come with a z-order answer key, so the synthetic generator in `scripts/make_synthetic_dataset.py` does the heavy lifting for quantitative evaluation. The richer `layerbench_pp` export format adds:

- visible masks;
- amodal masks;
- alpha mattes;
- `layers_effects_rgba/` when effects are enabled;
- `intrinsics/albedo.png` and `intrinsics/shading.png`;
- `occlusion_graph.json` and `scene_metadata.json`.

That gives the project a ground-truth target for order, hidden support, effects, and intrinsic structure that no real photo can provide.

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

For the updated repo state, add one explicit recursive-peeling storyboard:

- original image;
- first selected layer;
- residual after inpaint;
- second selected layer;
- final completed background;
- any exported associated-effect layer.

That's a lot of panels per image, but it's the only way to make the components legible at a glance.

## Ablation table

| Variant | Segmenter | Depth | Inpaint | Intrinsics | Expected lesson |
|---|---|---|---|---|---|
| A | SLIC | luminance | Telea | Retinex | baseline |
| B | Mask2Former | luminance | Telea | Retinex | semantics help |
| C | Mask2Former | Depth Anything V2 | Telea | Retinex | depth improves order |
| D | Grounded SAM2 | Depth Pro | LaMa | Retinex | masks + metric depth |
| E | Grounded SAM2 | ensemble | LaMa | Marigold-IID external | strongest one-shot native recipe |
| F | Grounded SAM2 + peel | Depth Pro | iterative Telea/LaMa | Retinex | recursive peeling and effect layers |
