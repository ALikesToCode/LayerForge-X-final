# Report Tables

These tables are partly filled with the runs that have already been executed in this repo. Leave the remaining cells blank in the report until those experiments are actually run.

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

## Table 5 — Editing utility

| Method | Object removal | Object move | Recolor | Parallax | Non-edit preservation ↓ | Runtime ↓ |
|---|---:|---:|---:|---:|---:|---:|
| mask only | | | | | | |
| full LayerForge-X | | | | | | |
| Qwen-Image-Layered | | | | | | |
| Qwen + graph | | | | | | |
| recursive peeling + effects | | | | | | |

## Table 6 — Five-image Qwen raw versus hybrid review

| Method | Images | Graph | Mean PSNR ↑ | Mean SSIM ↑ | Notes |
|---|---:|---|---:|---:|---|
| LayerForge native | 5 | yes | 27.3438 | 0.9464 | strongest mean SSIM; much larger average stack |
| Qwen raw (4) | 5 | no | 29.0757 | 0.8850 | best mean PSNR on this measured set |
| Qwen + LayerForge graph preserve (4) | 5 | yes | 28.5539 | 0.8638 | fair metadata-first hybrid; preserves Qwen visual order |
| Qwen + LayerForge graph reorder (4) | 5 | yes | 28.5397 | 0.8637 | explicit graph-order export |

Per-image note:

- raw Qwen keeps the best mean PSNR, but native LayerForge wins on PSNR for `truck` and `coffee`;
- native LayerForge has the best mean SSIM across the five images, albeit with a much larger stack than Qwen;
- `Qwen + graph preserve` is the fairest hybrid row because it keeps the interpreted external order while adding graph, amodal, and intrinsic metadata;
- the hybrid rows should therefore be framed as structured-representation complements, not universal visual-fidelity winners.

## Table 7 — Associated-effect demo

| Artifact | Source scene | Effect detected | Predicted effect px | Ground-truth effect px | Effect IoU |
|---|---|---|---:|---:|---:|
| `runs/effects_groundtruth_demo_cutting_edge` | `layerbench_pp` synthetic scene with `near_person_shadow` | yes | 4853 | 13750 | 0.3529 |

Notes:

- the repo now contains a real associated-effect demo figure at `docs/figures/effects_layer_demo.png`;
- the extractor is still an early heuristic prototype, but the clean-reference rerun is now materially stronger than the first draft and good enough to discuss without overselling it as solved shadow decomposition.
