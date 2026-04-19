# Novelty and Ablations

## Main novelty claims

### Claim 1 — Depth-Aware Amodal Layer Graph

LayerForge-X represents a decomposed image as a graph, not only a stack. Nodes store semantic RGBA layers; edges store depth/occlusion relations.

### Claim 2 — Boundary-weighted occlusion ordering

Layer ordering uses local depth evidence near shared boundaries rather than only global mean/median depth.

### Claim 3 — Layer enrichment beyond RGBA

Each layer is enriched with semantic group, depth statistics, amodal support, soft alpha, completed background metadata, and optional albedo/shading.

### Claim 4 — Qwen-aware evaluation

Qwen-Image-Layered is treated as a modern generative baseline and optional proposal source.

### Claim 5 — Multi-axis benchmark

The project evaluates segmentation, ordering, graph quality, recomposition, amodal masks, intrinsic decomposition, editing utility, and runtime.

## Ablation table

| Variant | Segmentation | Depth | Ordering | Alpha | Amodal | Inpaint | Intrinsic | Purpose |
|---:|---|---|---|---|---|---|---|---|
| A | SLIC/classical | luminance | global median | hard | no | no | no | weak baseline |
| B | Mask2Former | none | area heuristic | hard | no | no | no | segmentation-only |
| C | Mask2Former | Depth Anything V2 | global median | hard | no | no | no | tests depth |
| D | Mask2Former | Depth Anything V2 | boundary graph | hard | no | no | no | tests graph ordering |
| E | GroundingDINO + SAM2 | Depth Anything V2 | boundary graph | soft | no | no | no | open vocab + alpha |
| F | GroundingDINO + SAM2 | Depth Pro/MoGe | boundary graph | soft | heuristic | OpenCV | no | amodal + completion |
| G | GroundingDINO + SAM2 | ensemble | learned ranker | soft/matting | yes | LaMa | no | learned ordering |
| H | full | ensemble | learned graph | soft/matting | yes | LaMa | yes | final method |
| Q | Qwen-Image-Layered | implicit | none/manual | generated | implicit | generated | no | frontier baseline |
| Q+G | Qwen + LayerForge graph | depth model | boundary graph | generated | implicit | generated | yes | hybrid |

## Interpretation

```text
A → B: segmentation quality matters
B → C: depth improves ordering
C → D: boundary graph improves occlusion reasoning
D → E: open-vocabulary masks and soft alpha improve editing
E → F: amodal support and inpainting improve object removal/movement
F → G: learned pairwise ordering improves consistency
G → H: intrinsic split enables appearance edits
Q/Q+G: compares against frontier generative layer decomposition
```

## Claims to avoid

Do not write:

```text
We solve single-image layer decomposition.
```

Write:

```text
We provide a practical and inspectable approximation to single-image layer decomposition.
```

Do not write:

```text
We recover true hidden object appearance.
```

Write:

```text
We synthesize plausible hidden/background content for editing and evaluate it when ground truth is available.
```

Do not write:

```text
We beat Qwen-Image-Layered.
```

Write:

```text
We compare against Qwen-Image-Layered and show complementary strengths in explicit ordering, graph metadata, and component-level evaluation.
```
