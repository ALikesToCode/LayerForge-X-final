# Report Tables

These tables are partly filled with the runs that have already been executed in this repo. Leave the remaining cells blank in the report until those experiments are actually run.

## Table 1 — Segmentation

| Method | Dataset | mIoU ↑ | PQ ↑ | SQ ↑ | RQ ↑ | Avg layers | Runtime ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SLIC/classical | Synthetic-LayerBench | 0.1549 | - | - | - | 65.0 | - |
| Mask2Former | COCO/ADE20K | | | | | | |
| GroundingDINO + SAM2 | curated | | | | | | |
| Qwen-Image-Layered | curated/synthetic | | | | | | |
| LayerForge-X | mixed | | | | | | |

## Table 2 — Depth ordering

| Method | Depth model | Ordering | PLOA ↑ | BW-PLOA ↑ | Edge F1 ↑ | Kendall τ ↑ |
|---|---|---|---:|---:|---:|---:|
| area heuristic | none | largest front | | | | |
| median depth | Depth Anything V2 | global median | | | | |
| boundary graph | Depth Anything V2 | boundary-local | | | | |
| boundary graph | geometric luminance | boundary-local | 0.1667 | - | - | - |
| learned ranker | geometric luminance | pairwise classifier | 0.1667 | - | - | - |
| Qwen + graph | Depth Pro | boundary-local | - | - | - | - |

## Table 3 — Recomposition

| Method | Alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Coverage error ↓ |
|---|---|---:|---:|---:|---:|
| hard masks | hard | | | | |
| soft alpha | soft | | | | |
| fast LayerForge-X | soft + graph | 19.1589 | 0.8966 | - | - |
| fast LayerForge-X + learned ranker | soft + graph + learned order | 19.4138 | 0.8954 | - | - |
| Qwen-Image-Layered | generated RGBA | 26.7874 | 0.7723 | - | - |
| Qwen + graph | generated + ordered | 27.4612 | 0.7953 | - | - |

## Table 4 — Amodal and completion

| Method | Dataset | Visible IoU ↑ | Amodal IoU ↑ | Hidden IoU ↑ | Masked LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| visible mask only | synthetic/KINS | | | | |
| geometric amodal | synthetic/KINS | | | | |
| SAM2 expansion | synthetic/KINS | | | | |
| full LayerForge-X | synthetic/KINS/MP3D | | | | |
| Qwen baseline | curated/synthetic | | | | |

## Table 5 — Editing utility

| Method | Object removal | Object move | Recolor | Parallax | Non-edit preservation ↓ | Runtime ↓ |
|---|---:|---:|---:|---:|---:|---:|
| mask only | | | | | | |
| full LayerForge-X | | | | | | |
| Qwen-Image-Layered | | | | | | |
| Qwen + graph | | | | | | |
