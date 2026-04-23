# LayerForge-X Final Report

# Abstract

Single-image editing systems increasingly need structured scene representations rather than a flat RGB bitmap or a folder of visible cutouts. LayerForge-X addresses that need by exporting a **Depth-Aware Amodal Layer Graph (DALG)**: ordered RGBA layers with semantic grouping, soft alpha, occlusion metadata, optional amodal support, background completion, intrinsic appearance factors, and editability-oriented diagnostics. The system combines native decomposition, Qwen/external RGBA enrichment, recursive peeling, promptable extraction, transparent-layer recovery, and explicit self-evaluation so different candidate representations can be compared under one graph contract. This report focuses on the measured behavior of those components, the benchmark protocol used to evaluate them, and the practical limits that still separate the current implementation from fully generative layered scene understanding.

# 1. Introduction

The core goal is not just to decompose an image, but to convert it into an **editable scene asset**. That requires more than segmentation. A useful representation needs explicit near-to-far ordering, soft alpha boundaries, at least heuristic amodal support, some notion of hidden/background completion, and export surfaces that support real editing workflows. LayerForge-X therefore treats the scene graph as the canonical object and regards PNG stacks, debug artifacts, and design-manifest exports as projections of that graph.

# 2. Contributions

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

# 6. Results

### Hero figures

#### Native, hybrid, and graph-aware reconstruction

![Truck recomposition comparison](../figures/truck_recomposition_comparison.png){ width=100% }

#### Frontier candidate-bank selection

![Frontier review](../figures/frontier_review.png){ width=100% }

\newpage

#### Promptable extraction benchmark

![Prompt extraction benchmark](../figures/prompt_extract_benchmark.png){ width=100% }

#### Transparent decomposition benchmark

![Transparent benchmark](../figures/transparent_benchmark.png){ width=100% }

#### Associated-effect prototype

![Associated-effect demo](../figures/effects_layer_demo.png){ width=100% }

#### Intrinsic export demo

![Intrinsic layer demo](../figures/intrinsic_layer_demo.png){ width=100% }

### Main measured summary

Abbreviations in the tables below: `LF` = LayerForge, `Q raw4` = four-layer raw Qwen, `Q+G-P4` = four-layer Qwen plus LayerForge graph enrichment with the best external visual order preserved, and `Q+G-R4` = the same imported four-layer stack exported in graph order.

#### Five-image Qwen raw versus hybrid review

All rows below are five-image means. `Q+G-P4` keeps the best external Qwen stack, while `Q+G-R4` exports the same imported layers in graph order.

| Row | Mean PSNR | Mean SSIM |
|---|---:|---:|
| LF native | 27.3438 | 0.9464 |
| Q raw4 | 29.0757 | 0.8850 |
| Q+G-P4 | 28.5539 | 0.8638 |
| Q+G-R4 | 28.5397 | 0.8637 |

#### Associated-effect demo

| Metric | Value |
|---|---:|
| Predicted effect pixels | 4853 |
| Ground-truth effect pixels | 13750 |
| Effect IoU | 0.3529 |

#### Five-image frontier candidate-bank review

All rows below are five-image means.

| Row | PSNR | SSIM | Self-eval | Wins |
|---|---:|---:|---:|---:|
| LF native | 37.6688 | 0.9708 | 0.6981 | 4 |
| LF peel | 27.0988 | 0.9096 | 0.5314 | 0 |
| Q raw4 | 29.0757 | 0.8850 | 0.2824 | 0 |
| Q+G-P4 | 28.5539 | 0.8638 | 0.5843 | 0 |
| Q+G-R4 | 28.5397 | 0.8637 | 0.5834 | 1 |

#### Five-image editability suite

Response metrics:

| Row | Remove | Move | Recolor | Edit success |
|---|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 |
| Q raw4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 |
| Q+G-P4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 |
| Q+G-R4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 |

Stability metrics:

| Row | Non-edit preserve | Hole ratio |
|---|---:|---:|
| LF native | 0.9999 | 0.4860 |
| LF peel | 1.0000 | 0.5433 |
| Q raw4 | 1.0000 | 1.0000 |
| Q+G-P4 | 0.9887 | 0.1420 |
| Q+G-R4 | 0.9886 | 0.1427 |

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
- The `Q+G preserve 4` row is the most direct metadata-first hybrid comparison because it keeps the **best external visual order** while adding graph structure, amodal masks, ordering metadata, and intrinsic artifacts.
- The editability suite is the anti-triviality guardrail for the selector, which is why raw Qwen's object-removal response remains near zero despite reasonable recomposition scores.
- Promptable extraction is now a measured component instead of only a CLI feature. Text-bearing prompts currently carry the semantic routing load, while point-only and box-only prompts still need better disambiguation.
- Transparent recomposition is reported as a sanity check; alpha error and clean-background quality are the primary transparent-layer metrics.
- The associated-effect path now has a real exported demo artifact with a materially improved clean-reference rerun, but it must still be framed as an early heuristic rather than a solved component.
- The intrinsic export path is present as a Retinex-style stretch module. The new intrinsic demo figure should be read as evidence of exported appearance factors for recolouring-style edits, not as a state-of-the-art intrinsic benchmark.

# 7. Discussion

The current results do not support the claim that LayerForge-X universally exceeds generative decomposers on raw pixel fidelity. The defensible claim is narrower and more important: native, generative, and recursive decompositions are normalized into a single editable graph representation with auditable metrics and exportable structure. Qwen remains the appropriate generative RGBA baseline, while LayerForge-X is most compelling as a graph-aware, benchmarkable, editability-oriented complement to that frontier.

# 8. Limitations

Failure taxonomy and evaluation details are summarized in Appendix B. The main current limitations are:

- wrong semantic grouping;
- wrong depth order;
- jagged alpha boundaries;
- missing shadow/effect layers;
- bad inpainting in large unseen regions;
- bad amodal continuation under heavy occlusion;
- intrinsic split errors;
- point-only and box-only prompt-routing ambiguity;
- transparent-layer recovery that is still approximate rather than generative.

# 9. Conclusion

LayerForge-X is best interpreted as a self-evaluating layer-representation system rather than a simple decomposition script. It produces native graph layers, enriches frontier RGBA layers, runs recursive peeling, measures editability, benchmarks prompt extraction, approximates transparent recovery, and exports a canonical DALG manifest. That combination defines the central project contribution.

<!-- include: 05 -->

# Appendix A: Artifact Map

Submission source-of-truth files:

- [../../PROJECT_MANIFEST.json](../../PROJECT_MANIFEST.json)
- [../../report_artifacts/README.md](../../report_artifacts/README.md)
- [../../report_artifacts/metrics_snapshots/](../../report_artifacts/metrics_snapshots)
- [../../report_artifacts/figure_sources/figure_manifest.json](../../report_artifacts/figure_sources/figure_manifest.json)
- [../../report_artifacts/command_log.md](../../report_artifacts/command_log.md)
- [../RESULTS_SUMMARY_CURRENT.md](../RESULTS_SUMMARY_CURRENT.md)
- [../QWEN_IMAGE_LAYERED_COMPARISON.md](../QWEN_IMAGE_LAYERED_COMPARISON.md)
- [../REPORT_TABLES.md](../REPORT_TABLES.md)
- [../FIGURES.md](../FIGURES.md)
- [../PRODUCT_ARCHITECTURE_AND_LAUNCH.md](../PRODUCT_ARCHITECTURE_AND_LAUNCH.md)
- [../api/openapi.yaml](../api/openapi.yaml)

# Appendix B: Extended Tables and Ablations

\newpage

<!-- include: 04 -->

# Appendix C: Command Log

The full command log is shipped in [../../report_artifacts/command_log.md](../../report_artifacts/command_log.md). The final archive refresh used the following command families:

```bash
./.venv/bin/pytest -q
python scripts/export_report_artifacts.py
python scripts/generate_report_figures.py
python scripts/build_report_docx.py
python scripts/build_site_data.py
python scripts/make_submission_zip.py
```

Those commands connect the final report outputs to the compact JSON summaries and generated figures that are shipped in the evidence pack.

# Appendix D: Extra Literature Notes

The literature used in this report clusters into five recurring themes:

- layered rendering and depth-aware image representations;
- panoptic, amodal, and promptable segmentation;
- single-image intrinsic decomposition and matting;
- generative layered decomposition systems such as Qwen-Image-Layered and LayerDecomp-style methods;
- editing-oriented scene representations that prioritize reusable structure rather than one-off reconstruction.

Extended repository notes remain available in `docs/LITERATURE_REVIEW.md` and `docs/REFERENCES.md`, but the report body and the references section are the intended citation surface for submission.
