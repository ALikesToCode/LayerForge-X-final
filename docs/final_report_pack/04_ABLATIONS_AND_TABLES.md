# Appendix B. Extended Tables and Ablations

This appendix collects the extended quantitative tables, ablation templates, and failure-analysis material that support the main report.

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

## Frontier candidate-bank review

The repo now also contains the measured five-image frontier comparison at `runs/frontier_review/frontier_summary.json`.

| Method | Images | Mean PSNR | Mean SSIM | Mean self-eval score | Best-image wins |
|---|---:|---:|---:|---:|---:|
| LF native | 5 | 37.6688 | 0.9708 | 0.6283 | 4 |
| LF peel | 5 | 27.0988 | 0.9096 | 0.4783 | 0 |
| Qwen raw 4 | 5 | 29.0757 | 0.8850 | 0.2541 | 0 |
| Q+G preserve 4 | 5 | 28.5539 | 0.8638 | 0.5259 | 0 |
| Q+G reorder 4 | 5 | 28.5397 | 0.8637 | 0.5251 | 1 |

Interpretation:

- `LayerForge native` is now the strongest overall candidate-bank row by the repo's explicit self-evaluation score and wins `4/5` measured images once anti-triviality penalties are enabled;
- the hardened selector no longer lets `LayerForge peeling` win the truck image simply because the recursive removal path is visually dramatic;
- `Qwen + graph reorder` now wins the cat image, showing that imported generative stacks can still beat the native path on specific compact scenes;
- `Qwen raw` remains the compact frontier generative baseline, but it is no longer the best overall editable representation once structure and editability are scored explicitly.

## Editability suite snapshot

The frontier review is now paired with an editability suite so recomposition fidelity is not the only score that matters.

| Method | Remove response ↑ | Move response ↑ | Recolor response ↑ | Edit success ↑ | Non-edit preservation ↑ | Background hole ratio ↓ |
|---|---:|---:|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 | 0.9999 | 0.4860 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 | 1.0000 | 0.5433 |
| Qwen raw 4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 | 1.0000 | 1.0000 |
| Q+G preserve 4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 | 0.9887 | 0.1420 |
| Q+G reorder 4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 | 0.9886 | 0.1427 |

Interpretation:

- the editability suite is the anti-triviality guardrail for the frontier selector;
- `Qwen raw (4)` is the obvious example of why recomposition alone is insufficient, because its remove/move/recolor responses are almost zero while its background-hole ratio is effectively `1.0`;
- the hybrid rows currently post the strongest edit-success scores because imported generative stacks plus explicit LayerForge graph metadata are still easy to move, recolor, and remove cleanly.

## Promptable extraction benchmark snapshot

The prompt-conditioned extraction path is now measured instead of being only a CLI affordance.

| Prompt type | Queries | Target hit rate | Mean target IoU | Mean alpha MAE |
|---|---:|---:|---:|---:|
| text | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + point | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + box | 10 | 1.0000 | 0.3776 | 0.1503 |
| point | 10 | 0.0000 | 0.8654 | 0.0222 |
| box | 10 | 0.0000 | 0.8654 | 0.0222 |

Interpretation:

- text-bearing prompts now hit the intended semantic target on the measured synthetic set;
- point-only and box-only prompts still lock onto a neighboring region with high overlap but wrong semantics;
- the benchmark is therefore useful because it distinguishes semantic hit rate from overlap and alpha quality.

## Transparent benchmark snapshot

The transparent / alpha-composited recovery path now has a measured synthetic benchmark instead of only a qualitative smoke demo.

| Metric | Mean |
|---|---:|
| Transparent alpha MAE | 0.1131 |
| Background PSNR | 25.9863 |
| Background SSIM | 0.9541 |
| Recompose PSNR | 56.0066 |
| Recompose SSIM | 0.9996 |

Interpretation:

- transparent recomposition is a sanity check here; alpha error and clean-background quality are the primary transparent metrics;
- this path should be presented as an approximate transparent-layer recovery mode, not a claim of state-of-the-art generative transparent decomposition;
- the current prototype is strongest on flare-like overlays and weakest on the semi-transparent panel variant;
- despite that, it is now a measured component and belongs in the report as a frontier-aligned extension.

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
