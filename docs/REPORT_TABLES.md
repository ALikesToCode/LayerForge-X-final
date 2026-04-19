# Report Tables

Fill these after running experiments.

## Table 1 — Segmentation

| Method | Dataset | mIoU ↑ | PQ ↑ | SQ ↑ | RQ ↑ | Avg layers | Runtime ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SLIC/classical | Synthetic-LayerBench | | | | | | |
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
| boundary graph | Depth Pro | boundary-local | | | | |
| learned ranker | ensemble | pairwise classifier | | | | |
| Qwen + graph | Depth Pro | boundary-local | | | | |

## Table 3 — Recomposition

| Method | Alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Coverage error ↓ |
|---|---|---:|---:|---:|---:|
| hard masks | hard | | | | |
| soft alpha | soft | | | | |
| full LayerForge-X | soft + graph | | | | |
| Qwen-Image-Layered | generated RGBA | | | | |
| Qwen + graph | generated + ordered | | | | |

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
