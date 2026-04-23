This appendix collects the extended quantitative tables, measured ablations, and failure-taxonomy material that support the main report.

## B.1 Completed runs snapshot

The rows below report measured runs. All three use the deterministic classical segmenter with geometric-luminance depth; the changing factor is the ordering rule and the held-out split.

| Variant | Split | Ordering | Mean best IoU | PLOA | Recompose PSNR |
|---|---|---|---:|---:|---:|
| A1 | synthetic fast | boundary | 0.1549 | 0.1667 | 19.1360 |
| A2 | synth test | boundary | 0.1549 | 0.1667 | 19.1589 |
| A3 | synth test | learned ranker | 0.1549 | 0.1667 | 19.4138 |

Interpretation:

- `A2 → A3` provides the measured learned-ordering result;
- the dominant bottleneck remains proposal quality, because the fast classical segmenter still produces roughly `65` predicted layers for `5` ground-truth layers;
- the most credible qualitative path is therefore the real-image `grounded_sam2 + depth_pro` system rather than further tuning of the deterministic baseline.

## B.2 Frontier candidate-bank review

The five-image frontier comparison was measured locally, and the submission archive ships the copied summary in `report_artifacts/metrics_snapshots/frontier_review_summary.json`.

| Row | PSNR | SSIM | Self-eval | Wins |
|---|---:|---:|---:|---:|
| LF native | 37.6688 | 0.9708 | 0.6981 | 4 |
| LF peel | 27.0988 | 0.9096 | 0.5314 | 0 |
| Q raw4 | 29.0757 | 0.8850 | 0.2824 | 0 |
| Q+G-P4 | 28.5539 | 0.8638 | 0.5843 | 0 |
| Q+G-R4 | 28.5397 | 0.8637 | 0.5834 | 1 |

Interpretation:

- `LayerForge native` is the highest-scoring candidate-bank row under the explicit self-evaluation score and wins `4/5` measured images once anti-triviality penalties are enabled;
- `Qwen + graph reorder` wins the cat scene, showing that imported generative stacks can still outperform the native path on specific compact images;
- `Qwen raw` remains the compact generative baseline, but it is no longer the strongest editable representation once structure and editability are scored explicitly.

## B.3 Editability suite

The frontier review is paired with an editability suite so that recomposition fidelity is not the only selection signal.

Response metrics:

| Row | Remove ↑ | Move ↑ | Recolor ↑ | Edit success ↑ |
|---|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 |
| Q raw4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 |
| Q+G-P4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 |
| Q+G-R4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 |

Stability metrics:

| Row | Non-edit preserve ↑ | Hole ratio ↓ |
|---|---:|---:|
| LF native | 0.9999 | 0.4860 |
| LF peel | 1.0000 | 0.5433 |
| Q raw4 | 1.0000 | 1.0000 |
| Q+G-P4 | 0.9887 | 0.1420 |
| Q+G-R4 | 0.9886 | 0.1427 |

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

### B.6.1 Core ladder

| Variant | Segmentation | Depth | Ordering | Key additions | Purpose |
|---|---|---|---|---|---|
| A | classical | luminance | global median | hard alpha only | weak baseline |
| B | Mask2Former | none | area heuristic | semantic masks | semantic-only baseline |
| C | Mask2Former | Depth Anything V2 | global median | depth cue | depth-only test |
| D | Mask2Former | Depth Anything V2 | boundary graph | local ordering graph | graph-ordering test |
| E | Grounded-SAM2 | Depth Anything V2 | boundary graph | promptable masks + soft alpha | promptable decomposition |
| F | Grounded-SAM2 | Depth Pro / MoGe | boundary graph | heuristic amodal + OpenCV completion | amodal plus completion |

### B.6.2 Full-system extensions

| Variant | Backbone | Ordering | Completion | Intrinsics | Purpose |
|---|---|---|---|---|---|
| G | Grounded-SAM2 + ensemble depth | learned edge ranker | LaMa | no | strongest non-intrinsic native system |
| H | full native stack | learned edge ranker | LaMa | Retinex / Marigold-IID | full LayerForge-X |
| I | full native stack + peel | graph-guided peeling | iterative completion | Retinex / Marigold-IID | recursive peeling variant |

## B.7 Literature comparison

### B.7.1 Representation-oriented baselines

| Method family | Semantic layers | Explicit ordering | Hidden support | Notes |
|---|---:|---:|---:|---|
| LDI | no | yes | partial | rendering representation |
| 3D photo inpainting | limited | yes | yes | parallax-oriented RGB-D editing |
| Panoptic segmentation | yes | no | no | visible masks only |
| Grounded-SAM | yes | no | no | promptable visible masks |
| Amodal segmentation | object masks | limited | yes | hidden shape without full editable stack |

### B.7.2 Layered editing baselines

| Method family | Soft alpha | Completion | Intrinsics | Single image | Notes |
|---|---:|---:|---:|---:|---|
| Matting | yes | no | no | yes | strong alpha boundaries, no scene graph |
| LayerDecomp-style | yes | yes | no | yes | generative editing baseline |
| Qwen-Image-Layered-style | yes | yes | limited | yes | end-to-end generative RGBA layers |
| LayerForge-X | yes | yes | optional | yes | modular and benchmarkable graph representation |

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

| Run | Method | Global factors | Per-layer exports | Representative layer |
|---|---|---|---|---|
| `truck_candidate_search_v2/best` | `retinex` | albedo + shading | `20` albedo and `20` shading layers | `007_vehicle_car` |

Interpretation:

- the exported albedo and shading layers support recolouring-style demonstrations and inspection of appearance factors on top of the primary layer graph;
- the factorization remains approximate and is therefore treated as a stretch component rather than as a headline benchmark claim;
- the corresponding visual evidence is documented in `docs/figures/intrinsic_layer_demo.png`.
