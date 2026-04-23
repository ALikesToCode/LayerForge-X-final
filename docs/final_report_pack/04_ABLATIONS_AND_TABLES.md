# Appendix B. Extended Tables and Ablations

This appendix collects the extended quantitative tables, measured ablations, and failure-taxonomy material that support the main report.

## B.1 Completed runs snapshot

The rows below are measured runs rather than placeholders:

| Variant | Segmentation | Depth | Ordering | Split | Mean best IoU | PLOA | Recompose PSNR |
|---|---|---|---|---|---:|---:|---:|
| A1 | classical | geometric luminance | boundary | synthetic fast | 0.1549 | 0.1667 | 19.1360 |
| A2 | classical | geometric luminance | boundary | synth test | 0.1549 | 0.1667 | 19.1589 |
| A3 | classical | geometric luminance | learned ranker | synth test | 0.1549 | 0.1667 | 19.4138 |

Interpretation:

- `A2 → A3` provides the measured learned-ordering result;
- the dominant bottleneck remains proposal quality, because the fast classical segmenter still produces roughly `65` predicted layers for `5` ground-truth layers;
- the most credible qualitative path is therefore the real-image `grounded_sam2 + depth_pro` system rather than further tuning of the deterministic baseline.

## B.2 Frontier candidate-bank review

The repository contains a measured five-image frontier comparison in `runs/frontier_review/frontier_summary.json`.

| Method | Images | Mean PSNR | Mean SSIM | Mean self-eval score | Best-image wins |
|---|---:|---:|---:|---:|---:|
| LF native | 5 | 37.6688 | 0.9708 | 0.6283 | 4 |
| LF peel | 5 | 27.0988 | 0.9096 | 0.4783 | 0 |
| Qwen raw 4 | 5 | 29.0757 | 0.8850 | 0.2541 | 0 |
| Q+G preserve 4 | 5 | 28.5539 | 0.8638 | 0.5259 | 0 |
| Q+G reorder 4 | 5 | 28.5397 | 0.8637 | 0.5251 | 1 |

Interpretation:

- `LayerForge native` is the highest-scoring candidate-bank row under the explicit self-evaluation score and wins `4/5` measured images once anti-triviality penalties are enabled;
- `Qwen + graph reorder` wins the cat scene, showing that imported generative stacks can still outperform the native path on specific compact images;
- `Qwen raw` remains the compact generative baseline, but it is no longer the strongest editable representation once structure and editability are scored explicitly.

## B.3 Editability suite

The frontier review is paired with an editability suite so that recomposition fidelity is not the only selection signal.

| Method | Remove response ↑ | Move response ↑ | Recolor response ↑ | Edit success ↑ | Non-edit preservation ↑ | Background hole ratio ↓ |
|---|---:|---:|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 | 0.9999 | 0.4860 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 | 1.0000 | 0.5433 |
| Qwen raw 4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 | 1.0000 | 1.0000 |
| Q+G preserve 4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 | 0.9887 | 0.1420 |
| Q+G reorder 4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 | 0.9886 | 0.1427 |

Interpretation:

- the editability suite acts as the anti-triviality guardrail for the frontier selector;
- `Qwen raw (4)` demonstrates why recomposition alone is insufficient, because remove/move/recolor responses are near zero while the background-hole ratio is effectively `1.0`;
- the hybrid rows currently post the strongest edit-success scores because imported generative stacks combined with explicit LayerForge graph metadata remain easy to move, recolor, and remove cleanly.

## B.4 Promptable extraction benchmark

The prompt-conditioned extraction path is measured rather than treated only as a CLI feature.

| Prompt type | Queries | Target hit rate | Mean target IoU | Mean alpha MAE |
|---|---:|---:|---:|---:|
| text | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + point | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + box | 10 | 1.0000 | 0.3776 | 0.1503 |
| point | 10 | 0.0000 | 0.8654 | 0.0222 |
| box | 10 | 0.0000 | 0.8654 | 0.0222 |

Interpretation:

- text-bearing prompts identify the intended semantic target on the measured synthetic set;
- point-only and box-only prompts still achieve high overlap while missing the semantic target;
- the present bottleneck is semantic routing rather than matte stability.

## B.5 Transparent benchmark

The transparent or alpha-composited recovery path has a measured synthetic benchmark rather than only a qualitative smoke demonstration.

| Metric | Mean |
|---|---:|
| Transparent alpha MAE | 0.1131 |
| Background PSNR | 25.9863 |
| Background SSIM | 0.9541 |
| Recompose PSNR | 56.0066 |
| Recompose SSIM | 0.9996 |

Interpretation:

- transparent recomposition is a sanity check; alpha error and clean-background quality are the primary transparent metrics;
- the current path should be described as approximate transparent-layer recovery rather than state-of-the-art generative transparent decomposition;
- the prototype is strongest on flare-like overlays and weakest on the semi-transparent panel variant.

## B.6 Main ablation matrix

| Variant | Segmentation | Depth | Ordering | Alpha | Amodal | Inpaint | Intrinsic | Purpose |
|---|---|---|---|---|---|---|---|---|
| A | SLIC/classical | luminance | global median | hard | no | no | no | weak baseline |
| B | Mask2Former | none | area/heuristic | hard | no | no | no | semantic-only baseline |
| C | Mask2Former | Depth Anything V2 | global median | hard | no | no | no | depth-only test |
| D | Mask2Former | Depth Anything V2 | boundary graph | hard | no | no | no | graph-ordering test |
| E | Grounded-SAM2 | Depth Anything V2 | boundary graph | soft | no | no | no | promptable masks plus soft alpha |
| F | Grounded-SAM2 | Depth Pro / MoGe | boundary graph | soft | heuristic | OpenCV | no | amodal plus completion |
| G | Grounded-SAM2 | ensemble | learned edge ranker | soft/matting | amodal | LaMa | no | strong non-intrinsic system |
| H | full | ensemble | learned edge ranker | soft/matting | amodal | LaMa | Retinex / Marigold-IID | full LayerForge-X |
| I | full + peel | ensemble | graph-guided peeling | soft/matting | amodal | iterative completion | Retinex / Marigold-IID | recursive peeling variant |

## B.7 Literature comparison

| Method family | Semantic layers | Depth order | Amodal hidden parts | Soft alpha | Inpainting | Intrinsics | Single image | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LDI | no | yes | partial | no | no | no | yes/varies | rendering representation |
| 3D photo inpainting | no/limited | yes | yes | no | yes | no | RGB-D/depth | parallax focus |
| Panoptic segmentation | yes | no | no | no | no | no | yes | visible masks only |
| Grounded-SAM | yes/open vocab | no | no | no | no | no | yes | promptable visible masks |
| Matting | foreground only | no | no | yes | no | no | yes | strong alpha boundaries |
| Amodal segmentation | object masks | limited | yes | no | sometimes | no | yes | hidden shape, not full layer stack |
| LayerDecomp-style | foreground/background | partial | yes | yes | yes | no | yes | strong generative editing baseline |
| Qwen-Image-Layered-style | yes | implicit | yes | yes | yes | no/limited | yes | end-to-end generative RGBA layers |
| LayerForge-X | yes | explicit graph | yes | yes | yes | optional | yes | modular and benchmarkable |

## B.8 Dataset coverage

| Dataset | Used for | Ground truth available | Metrics |
|---|---|---|---|
| Synthetic-LayerBench / layerbench_pp | full pipeline | RGBA layers, z-order, masks, clean background, optional albedo/shading, optional effects | PLOA, PSNR, SSIM, LPIPS, alpha MAE, amodal IoU |
| COCO Panoptic | visible semantic grouping | panoptic masks | coarse-group mIoU, thing/stuff mIoU |
| ADE20K | scene/stuff parsing | semantic masks | coarse-group mIoU, pixel accuracy |
| NYU Depth V2 | indoor depth order | RGB-D, labels | AbsRel, RMSE, PLOA |
| DIODE | indoor/outdoor depth | RGB-D | AbsRel, RMSE, PLOA |
| KINS / COCOA | amodal segmentation | amodal masks | modal IoU, amodal IoU, invisible IoU |
| IIW | intrinsic decomposition | reflectance comparisons | WHDR |
| Real curated set | qualitative editing | no full ground truth | visual comparison, preference judgments |

## B.9 Failure taxonomy

| Failure | Cause | Example | Fix / future work |
|---|---|---|---|
| Wrong semantic grouping | segmenter misses object or merges regions | chair merged with table | stronger prompts or panoptic model |
| Wrong depth order | monocular depth ambiguity | mirror, window, poster | boundary ranker plus uncertainty |
| Jagged edge | hard mask or weak matting | hair or fur | stronger matting backend |
| Missing shadow or effect | object-only mask | person moved without shadow | associated-effect layer |
| Bad inpainting | large unseen region | removed foreground person | stronger completion backend |
| Bad amodal shape | heavy occlusion | hidden vehicle side | amodal model |
| Intrinsic artifacts | single-image ambiguity | texture mistaken as shading | stronger IID model |
| Too many layers | oversegmentation | fragmented background | graph merging |
| Too few layers | undersegmentation | person and bicycle merged | prompt refinement |

## B.10 Intrinsic export snapshot

The optional intrinsic path is implemented as an approximate appearance-factor export rather than as a standalone intrinsic-image benchmark. The measured truck winner currently provides both global and per-layer albedo/shading artifacts.

| Run | Intrinsic method | Global albedo | Global shading | Per-layer albedo exports | Per-layer shading exports | Representative layer |
|---|---|---|---|---:|---:|---|
| `truck_candidate_search_v2/best` | `retinex` | yes | yes | 20 | 20 | `007_vehicle_car` |

Interpretation:

- the exported albedo and shading layers support recolouring-style demonstrations and inspection of appearance factors on top of the primary layer graph;
- the factorization remains approximate and is therefore treated as a stretch component rather than as a headline benchmark claim;
- the corresponding visual evidence is documented in `docs/figures/intrinsic_layer_demo.png`.
