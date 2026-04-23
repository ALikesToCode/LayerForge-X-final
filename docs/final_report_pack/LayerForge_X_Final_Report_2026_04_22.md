# LayerForge-X Final Report

## Abstract

Single-image editing systems increasingly need structured scene representations rather than a flat RGB bitmap or a folder of visible cutouts. LayerForge-X addresses that need by exporting a **Depth-Aware Amodal Layer Graph (DALG)**: ordered RGBA layers with semantic grouping, soft alpha, occlusion metadata, optional amodal support, background completion, intrinsic appearance factors, and editability-oriented diagnostics. The system combines native decomposition, Qwen/external RGBA enrichment, recursive peeling, promptable extraction, transparent-layer recovery, and explicit self-evaluation so different candidate representations can be compared under one graph contract. This report focuses on the measured behavior of those components, the benchmark protocol used to evaluate them, and the practical limits that still separate the current implementation from fully generative layered scene understanding.

## 1. Introduction

The core goal is not just to decompose an image, but to convert it into an **editable scene asset**. That requires more than segmentation. A useful representation needs explicit near-to-far ordering, soft alpha boundaries, at least heuristic amodal support, some notion of hidden/background completion, and export surfaces that support real editing workflows. LayerForge-X therefore treats the scene graph as the canonical object and regards PNG stacks, debug artifacts, and design-manifest exports as projections of that graph.

## 2. Contributions

This report makes six concrete claims:

1. LayerForge-X implements a depth-aware amodal layer graph rather than a simple bag of masks.
2. The repo includes a fair Qwen comparison with preserve/reorder hybrid modes and a common evaluation frame.
3. Recursive peeling is implemented as a measured alternative path rather than only a conceptual extension.
4. The evaluation stack now includes anti-trivial editability metrics, not only recomposition fidelity.
5. Promptable extraction and transparent decomposition are both implemented as measured benchmarked components.
6. The system exports a canonical DALG manifest and a product-facing design-manifest projection suitable for future API/editor integration.

\newpage

<!-- include: 01 -->

\newpage

<!-- include: 03 -->

\newpage

<!-- include: 02 -->

## 5. Results

### Hero figures

#### Native, hybrid, and graph-aware reconstruction

![Truck recomposition comparison](../figures/truck_recomposition_comparison.png){ width=100% }

#### Frontier candidate-bank selection

![Frontier review](../figures/frontier_review.png){ width=100% }

#### Promptable extraction benchmark

![Prompt extraction benchmark](../figures/prompt_extract_benchmark.png){ width=100% }

#### Transparent decomposition benchmark

![Transparent benchmark](../figures/transparent_benchmark.png){ width=100% }

#### Associated-effect prototype

![Associated-effect demo](../figures/effects_layer_demo.png){ width=100% }

### Main measured summary

Abbreviations in the tables below: `LF` = LayerForge, `Q+G` = Qwen plus LayerForge graph enrichment.

#### Five-image Qwen raw versus hybrid review

| Method | Images | Graph | Mean PSNR | Mean SSIM |
|---|---:|---|---:|---:|
| LF native | 5 | yes | 27.3438 | 0.9464 |
| Qwen raw 4 | 5 | no | 29.0757 | 0.8850 |
| Q+G preserve 4 | 5 | yes | 28.5539 | 0.8638 |
| Q+G reorder 4 | 5 | yes | 28.5397 | 0.8637 |

#### Associated-effect demo

| Artifact | Effect detected | Predicted effect px | Ground-truth effect px | Effect IoU |
|---|---|---:|---:|---:|
| `runs/effects_groundtruth_demo_cutting_edge` | yes | 4853 | 13750 | 0.3529 |

#### Five-image frontier candidate-bank review

| Method | Images | Mean PSNR | Mean SSIM | Mean self-eval | Best-image wins |
|---|---:|---:|---:|---:|---:|
| LF native | 5 | 37.6688 | 0.9708 | 0.6283 | 4 |
| LF peel | 5 | 27.0988 | 0.9096 | 0.4783 | 0 |
| Qwen raw 4 | 5 | 29.0757 | 0.8850 | 0.2541 | 0 |
| Q+G preserve 4 | 5 | 28.5539 | 0.8638 | 0.5259 | 0 |
| Q+G reorder 4 | 5 | 28.5397 | 0.8637 | 0.5251 | 1 |

#### Five-image editability suite

| Method | Remove | Move | Recolor | Edit success | Non-edit preserve | Hole ratio |
|---|---:|---:|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 | 0.9999 | 0.4860 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 | 1.0000 | 0.5433 |
| Qwen raw 4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 | 1.0000 | 1.0000 |
| Q+G preserve 4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 | 0.9887 | 0.1420 |
| Q+G reorder 4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 | 0.9886 | 0.1427 |

#### Promptable extraction benchmark

| Prompt type | Queries | Hit rate | Mean IoU | Mean alpha MAE |
|---|---:|---:|---:|---:|
| text | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + point | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + box | 10 | 1.0000 | 0.3776 | 0.1503 |
| point | 10 | 0.0000 | 0.8654 | 0.0222 |
| box | 10 | 0.0000 | 0.8654 | 0.0222 |

#### Transparent benchmark

| Metric | Mean |
|---|---:|
| Transparent alpha MAE | 0.1131 |
| Background PSNR | 25.9863 |
| Background SSIM | 0.9541 |
| Recompose PSNR | 56.0066 |
| Recompose SSIM | 0.9996 |

### Interpretation

- Raw Qwen remains the stronger compact pure-PSNR baseline on the measured five-image sweep.
- Native LayerForge posts the strongest mean SSIM on the same images, at the cost of a larger average stack.
- The measured frontier candidate bank selects `LF native` for `4/5` images, with `Q+G reorder 4` winning the cat scene.
- The `Q+G preserve 4` row is the fair metadata-first hybrid comparison because it keeps the **best external visual order** while adding graph structure, amodal masks, ordering metadata, and intrinsic artifacts.
- The editability suite is the anti-triviality guardrail for the selector, which is why raw Qwen's object-removal response remains near zero despite reasonable recomposition scores.
- Promptable extraction is now a measured component instead of only a CLI feature. Text-bearing prompts currently carry the semantic routing load, while point-only and box-only prompts still need better disambiguation.
- Transparent recomposition is reported as a sanity check; alpha error and clean-background quality are the primary transparent-layer metrics.
- The associated-effect path now has a real exported demo artifact with a materially improved clean-reference rerun, but it must still be framed as an early heuristic rather than a solved component.

## 6. Discussion

The strongest reading of the current results is not that LayerForge-X universally beats generative decomposers on raw pixels. The stronger claim is that it turns native, generative, and recursive decompositions into one explicit editable graph representation with auditable metrics and exportable structure. Qwen remains the right generative RGBA baseline. LayerForge-X remains strongest when framed as a graph-aware, benchmarkable, editability-oriented complement to that frontier.

## 7. Limitations

Failure taxonomy and future-work framing are documented in [04_ABLATIONS_AND_TABLES.md](04_ABLATIONS_AND_TABLES.md) and [02_BENCHMARKING_PROTOCOL.md](02_BENCHMARKING_PROTOCOL.md). The report should explicitly keep:

- wrong semantic grouping;
- wrong depth order;
- jagged alpha boundaries;
- missing shadow/effect layers;
- bad inpainting in large unseen regions;
- bad amodal continuation under heavy occlusion;
- intrinsic split errors;
- point-only and box-only prompt-routing ambiguity;
- transparent-layer recovery that is still approximate rather than generative.

## 8. Conclusion

LayerForge-X is now best understood as a self-evaluating layer-representation system rather than a simple decomposition script. It can produce native graph layers, enrich frontier RGBA layers, run recursive peeling, measure editability, benchmark prompt extraction, approximate transparent recovery, and export a canonical DALG manifest. That combination is the core project contribution.

## Appendix A: artifact map

Primary source files for the report narrative:

- [../RESULTS_SUMMARY_2026_04_19.md](../RESULTS_SUMMARY_2026_04_19.md)
- [../QWEN_IMAGE_LAYERED_COMPARISON.md](../QWEN_IMAGE_LAYERED_COMPARISON.md)
- [../REPORT_TABLES.md](../REPORT_TABLES.md)
- [../FIGURES.md](../FIGURES.md)
- [../NEXT_REVIEW_CHECKLIST_2026_04_22.md](../NEXT_REVIEW_CHECKLIST_2026_04_22.md)
- [../PRODUCT_ARCHITECTURE_AND_LAUNCH.md](../PRODUCT_ARCHITECTURE_AND_LAUNCH.md)
- [../api/openapi.yaml](../api/openapi.yaml)
- [../../report_artifacts/command_log.md](../../report_artifacts/command_log.md)

## Appendix B: extended tables and ablations

\newpage

<!-- include: 04 -->
