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
| Qwen + graph | Depth Pro | boundary-local | - | - | - | - |
| recursive peeling | Depth Pro | iterative front-to-back | - | - | - | - |

## Table 3 — Recomposition

| Method | Alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Coverage error ↓ |
|---|---|---:|---:|---:|---:|
| hard masks | hard | | | | |
| soft alpha | soft | | | | |
| fast LayerForge-X | soft + graph | 19.1589 | 0.8966 | - | - |
| fast LayerForge-X + learned ranker | soft + graph + learned order | 19.4138 | 0.8954 | - | - |
| Qwen-Image-Layered (5-image review mean) | generated RGBA | 29.0757 | 0.8850 | - | - |
| Qwen + graph (5-image review mean) | generated + ordered | 28.5408 | 0.8637 | - | - |
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
| Qwen raw (4) | 5 | no | 29.0757 | 0.8850 | best pixel fidelity on this measured set |
| Qwen + LayerForge graph (4) | 5 | yes | 28.5408 | 0.8637 | explicit graph, amodal masks, intrinsic layers, depth ordering |

Per-image note:

- raw Qwen wins PSNR on all five images in the current sweep;
- the hybrid improves SSIM only on `astronaut`;
- the hybrid row should therefore be framed as a structured-representation complement, not a universal visual-fidelity winner.

## Table 7 — Associated-effect demo

| Artifact | Source scene | Effect detected | Predicted effect px | Ground-truth effect px | Effect IoU |
|---|---|---|---:|---:|---:|
| `runs/effects_groundtruth_demo_cutting_edge` | `layerbench_pp` synthetic scene with `near_person_shadow` | yes | 411 | 13750 | 0.0006 |

Notes:

- the repo now contains a real associated-effect demo figure at `docs/figures/effects_layer_demo.png`;
- the extractor fires, but the current heuristic remains weak and should be described as an early demo rather than a solved effect-layer method.
