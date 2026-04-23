# Novelty and Ablations

## Main novelty claims

The points below summarize the elements that are distinct within LayerForge-X and defensible as project-level contributions.

### Claim 1 — Depth-Aware Amodal Layer Graph

The output is a graph rather than only a layer stack. Nodes carry semantic RGBA layers together with structured metadata, while edges encode depth and occlusion relations with confidence scores. A flat folder of PNG files does not preserve that information.

### Claim 2 — Boundary-aware ordering with an optional learned ranker

Layer ordering uses local depth evidence near shared boundaries instead of only global mean or median depth. The repository also includes a lightweight learned pairwise ranker trained on synthetic scenes. This provides a direct heuristic-versus-learned ordering comparison without recasting the full system as an end-to-end trained decomposer.

### Claim 3 — Layer enrichment beyond RGBA

Each layer ships with a semantic group, depth statistics, amodal support, soft alpha, optional background-completion metadata, and optional albedo/shading — not just a cutout.

### Claim 4 — Qwen-aware evaluation

Qwen-Image-Layered is treated as a modern generative baseline *and* an optional proposal source. The `enrich-qwen` command operationalises the second of those.

### Claim 5 — Multi-axis benchmark

Evaluation covers segmentation, ordering, graph quality, recomposition, amodal masks, intrinsic decomposition, and editing utility. Each axis addresses a different failure mode and therefore contributes distinct evidence. Runtime remains an optional future selector signal rather than an active component of the archived five-image frontier summary.

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

## Completed ablations in this repo

The full table above is the evaluation plan. The completed runs currently checked into the repo are the subset below:

| Variant | Segmentation | Depth | Ordering | Split | Mean best IoU | PLOA | Recompose PSNR |
|---:|---|---|---|---|---:|---:|---:|
| A1 | classical | geometric luminance | boundary | synthetic fast | 0.1549 | 0.1667 | 19.1360 |
| A2 | classical | geometric luminance | boundary | synth test | 0.1549 | 0.1667 | 19.1589 |
| A3 | classical | geometric luminance | learned ranker | synth test | 0.1549 | 0.1667 | 19.4138 |

The key measured result is `A2 → A3`: the learned ranker improves recomposition PSNR by about `+0.255` dB on held-out scenes, but the current ordering metric does not improve because the proposal stage still returns about `65` predicted layers for `5` ground-truth layers.

## Interpretation

Read the step-by-step diffs as mini-experiments:

```text
A → B: segmentation quality matters
B → C: depth improves ordering
C → D: boundary graph improves occlusion reasoning
D → E: open-vocabulary masks and soft alpha improve editing
E → F: amodal support and inpainting improve object removal/movement
F → G: learned pairwise ordering should improve consistency when the proposal quality is good enough
G → H: intrinsic split enables appearance edits
Q/Q+G: compares against frontier generative layer decomposition
```

## Claim boundaries

The phrasing below keeps the project claims aligned with the measured evidence and avoids overstating the current implementation.

Avoid:

```text
We solve single-image layer decomposition.
```

Prefer:

```text
We provide a practical and inspectable approximation to single-image layer decomposition.
```

Avoid:

```text
We recover true hidden object appearance.
```

Prefer:

```text
We synthesize plausible hidden/background content for editing and evaluate it when ground truth is available.
```

Avoid:

```text
We beat Qwen-Image-Layered.
```

Prefer:

```text
We compare against Qwen-Image-Layered and show complementary strengths in explicit ordering, graph metadata, and component-level evaluation.
```
