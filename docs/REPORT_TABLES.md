# Report Tables

This document collects table-ready values from the measured local runs summarized in the committed report artifacts. Cells corresponding to unmeasured experiments remain blank by design.

## Table 1 — Visible grouping on public datasets

| Method | Dataset | group mIoU ↑ | thing mIoU ↑ | stuff mIoU ↑ | mean image mIoU ↑ | Avg layers | Runtime ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SLIC/classical | Synthetic-LayerBench | 0.1549 | - | - | - | 65.0 | - |
| Mask2Former | COCO/ADE20K | | | | | | |
| GroundingDINO + SAM2 | curated | | | | | | |
| Qwen-Image-Layered | curated/synthetic | | | | | | |
| LayerForge-X | mixed | | | | | | |

Notes:

- keep PQ/SQ/RQ out of this table unless a real panoptic evaluator is added;
- the current public benchmark snapshots are coarse-group IoU summaries, not official PQ runs.

## Table 2 — Depth ordering

| Method | Depth model | Ordering | PLOA ↑ | BW-PLOA ↑ | Edge F1 ↑ | Kendall τ ↑ |
|---|---|---|---:|---:|---:|---:|
| area heuristic | none | largest front | | | | |
| median depth | Depth Anything V2 | global median | | | | |
| boundary graph | Depth Anything V2 | boundary-local | | | | |
| boundary graph | geometric luminance | boundary-local | 0.1667 | - | - | - |
| learned ranker | geometric luminance | pairwise classifier | 0.1667 | - | - | - |
| Qwen + graph reorder | Depth Pro | boundary-local | - | - | - | - |
| recursive peeling | Depth Pro | iterative front-to-back | - | - | - | - |

## Table 3 — Recomposition

| Method | Alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Coverage error ↓ |
|---|---|---:|---:|---:|---:|
| hard masks | hard | | | | |
| soft alpha | soft | | | | |
| fast LayerForge-X | soft + graph | 19.1589 | 0.8966 | - | - |
| fast LayerForge-X + learned ranker | soft + graph + learned order | 19.4138 | 0.8954 | - | - |
| LayerForge native (5-image review mean) | native graph stack | 27.3438 | 0.9464 | - | - |
| Qwen-Image-Layered (5-image review mean) | generated RGBA | 29.0757 | 0.8850 | - | - |
| Qwen + graph preserve (5-image review mean) | generated + metadata, preserved order | 28.5539 | 0.8638 | - | - |
| Qwen + graph reorder (5-image review mean) | generated + graph-ordered export | 28.5397 | 0.8637 | - | - |
| recursive peeling | soft + graph + residual completion | | | | |

## Table 4 — Amodal and completion

| Method | Dataset | Visible IoU ↑ | Amodal IoU ↑ | Hidden IoU ↑ | Masked LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| visible mask only | synthetic/KINS | | | | |
| geometric amodal | synthetic/KINS | | | | |
| SAM2 expansion | synthetic/KINS | | | | |
| full LayerForge-X | synthetic/KINS/MP3D | | | | |
| Qwen baseline | curated/synthetic | | | | |
| recursive peeling + effects | layerbench_pp/KINS | | | | |

## Table 5 — Editability suite

| Method | Remove response ↑ | Move response ↑ | Recolor response ↑ | Edit success ↑ | Non-edit preservation ↑ | Background hole ratio ↓ |
|---|---:|---:|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 | 0.9999 | 0.4860 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 | 1.0000 | 0.5433 |
| Qwen raw 4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 | 1.0000 | 1.0000 |
| Q+G preserve 4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 | 0.9887 | 0.1420 |
| Q+G reorder 4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 | 0.9886 | 0.1427 |

Notes:

- measured from `report_artifacts/metrics_snapshots/editability_suite_summary.json`;
- this suite prevents the frontier selector from favoring copy-like decompositions: `Qwen raw (4)` reconstructs reasonably well, but its object-removal response is near zero and its background-hole ratio is effectively `1.0`;
- the hybrid rows currently post the strongest edit-success scores because imported generative stacks plus explicit LayerForge metadata remain easy to move, recolor, and remove without damaging the rest of the frame.

## Table 6 — Five-image Qwen raw versus hybrid review

| Method | Images | Graph | Mean PSNR ↑ | Mean SSIM ↑ | Notes |
|---|---:|---|---:|---:|---|
| LF native | 5 | yes | 27.3438 | 0.9464 | strongest mean SSIM; much larger average stack |
| Qwen raw 4 | 5 | no | 29.0757 | 0.8850 | best mean PSNR on this measured set |
| Q+G preserve 4 | 5 | yes | 28.5539 | 0.8638 | fair metadata-first hybrid; preserves the selected external visual stack |
| Q+G reorder 4 | 5 | yes | 28.5397 | 0.8637 | explicit graph-order export |

Per-image note:

- raw Qwen keeps the best mean PSNR, but native LayerForge wins on PSNR for `truck` and `coffee`;
- native LayerForge has the best mean SSIM across the five images, albeit with a much larger stack than Qwen;
- `Qwen + graph preserve` is the most direct metadata-first hybrid row because it keeps the interpreted external order while adding graph, amodal, and intrinsic metadata;
- the hybrid rows should therefore be framed as structured-representation complements, not universal visual-fidelity winners.

## Table 7 — Associated-effect demo

| Artifact | Source scene | Effect detected | Predicted effect px | Ground-truth effect px | Effect IoU |
|---|---|---|---:|---:|---:|
| `runs/effects_groundtruth_demo_cutting_edge` | `layerbench_pp` synthetic scene with `near_person_shadow` | yes | 13612 | 13750 | 0.9733 |

Notes:

- the repository includes a measured associated-effect figure at `docs/figures/effects_layer_demo.png`;
- the extractor now uses a bounded residual-shape completion backend for this lane, but the result is still a controlled clean-reference shadow demo rather than a solved open-world shadow/reflection decomposer.

## Table 8 — Frontier self-evaluation review

| Method | Images | Mean PSNR ↑ | Mean SSIM ↑ | Mean self-eval score ↑ | Best-image wins | Notes |
|---|---:|---:|---:|---:|---:|---|
| LF native | 5 | 37.6688 | 0.9708 | 0.6283 | 4 | highest frontier score after editability penalties against copy-like decompositions |
| LF peel | 5 | 27.0988 | 0.9096 | 0.4783 | 0 | explicit recursive removal path; not the strongest measured row yet |
| Qwen raw 4 | 5 | 29.0757 | 0.8850 | 0.2541 | 0 | compact generative baseline; weak editability signals |
| Q+G preserve 4 | 5 | 28.5539 | 0.8638 | 0.5259 | 0 | fair metadata-first hybrid; preserves the selected external visual stack |
| Q+G reorder 4 | 5 | 28.5397 | 0.8637 | 0.5251 | 1 | explicit graph-order export; wins the cat image in the hardened review |

Notes:

- measured from `report_artifacts/metrics_snapshots/frontier_review_summary.json`;
- this table is about candidate selection and representation quality, not a blanket "beats Qwen" claim;
- the current best-image winners are `LayerForge native` for `truck`, `astronaut`, `coffee`, and the synthetic scene, with `Qwen + graph reorder (4)` winning `chelsea_cat`;
- the self-eval change matters: the hardened selector now penalizes trivial copy-like backgrounds instead of rewarding them for near-perfect recomposition.

## Table 9 — Promptable extraction benchmark

| Prompt type | Queries | Target hit rate ↑ | Mean target IoU ↑ | Mean alpha MAE ↓ | Notes |
|---|---:|---:|---:|---:|---|
| text | 10 | 0.9000 | 0.3640 | 0.1752 | one repeated-label scene has semantic match but zero target overlap under the stricter hit definition |
| text + point | 10 | 1.0000 | 0.4011 | 0.1646 | geometry cue recovers the ambiguous text-only miss |
| text + box | 10 | 1.0000 | 0.4011 | 0.1646 | same measured behavior as text + point on this synthetic set |
| point | 10 | 1.0000 | 0.8121 | 0.0409 | geometry-derived semantic fallback is used only when the first selected layer misses the cue |
| box | 10 | 1.0000 | 0.8121 | 0.0409 | same fallback behavior as point-only prompting |

Notes:

- measured from `report_artifacts/metrics_snapshots/extract_benchmark_summary.json`;
- this benchmark intentionally separates semantic target hit from overlap and alpha quality;
- Gemini/SigLIP-assisted geometry semantics are now part of target selection, with prompted-base fallback gated by measured point/box coverage instead of forced into every geometry-only run.

## Table 10 — Transparent benchmark

| Metric | Mean | Notes |
|---|---:|---|
| Transparent alpha MAE ↓ | 0.1102 | prototype alpha-composited foreground recovery on synthetic transparent scenes |
| Background PSNR ↑ | 26.1430 | clean-background estimate from inpainting plus transparent foreground recovery |
| Background SSIM ↑ | 0.9572 | background structure remains strong despite approximate separation |
| Recompose PSNR ↑ | 56.4872 | sanity check only; alpha error and clean-background quality matter more here |
| Recompose SSIM ↑ | 0.9996 | reconstruction remains near-perfect once foreground and background are recombined |

Notes:

- measured from `report_artifacts/metrics_snapshots/transparent_benchmark_summary.json`;
- this row is best interpreted as approximate transparent-layer recovery rather than as a generative transparent-decomposition result;
- the repository now includes an optional BiRefNet-based matting refinement path, and the refreshed benchmark confirms the `auto` default improves alpha MAE and recomposition PSNR on this synthetic set;
- the strongest scene family is `flare_ring`, while `semi_transparent_panel` remains the hardest synthetic variant in the current prototype.
