# Defense Summary

## Concise project description

LayerForge-X converts a single RGB image into an interpretable layered scene representation rather than a flat set of masks. Each output layer is RGBA and carries semantic grouping, explicit near-to-far ordering, depth statistics, amodal support, and optional albedo/shading. The central contribution is not a new foundation model; it is the Depth-Aware Amodal Layer Graph together with a benchmarking protocol that measures segmentation, ordering, recomposition, and editability.

## Primary contributions

- the output is a graph-structured layer representation, not only exported masks;
- ordering is explicit and depth-aware rather than implicit;
- amodal masks, background completion, and intrinsic appearance are attached to the same layer objects;
- the repository supports a hybrid setting where external RGBA decompositions such as Qwen can be enriched with explicit graph metadata;
- the project includes a synthetic benchmark and a learned ordering ablation rather than relying only on qualitative demonstrations.

## Conservative claim boundaries

- the system should not be described as solving single-image layer decomposition in general;
- hidden content is estimated or plausibly completed, not recovered exactly;
- the repository does not claim to outperform Qwen-Image-Layered as a general-purpose generative decomposer;
- the full pipeline is not end-to-end trained.

## Comparative positioning against Qwen

Qwen-Image-Layered is the strongest recent open baseline for generating semantically disentangled RGBA layers from one image. LayerForge-X differs in emphasis: it turns decomposition into an inspectable scene representation with explicit ordering, occlusion edges, amodal masks, background-completion metadata, intrinsic layers, and evaluation hooks. Qwen is therefore best framed as a frontier baseline and an optional proposal source.

## Interpretation notes

### Synthetic benchmark scale

The deterministic fallback uses classical segmentation so it can run anywhere and serve as a controlled reference point. That baseline over-segments strongly, producing roughly 65 predicted layers for 5 ground-truth layers on the synthetic benchmark. Mean IoU and pairwise ordering accuracy are therefore bottlenecked by proposal quality. The learned ranker still improves recomposition PSNR on the held-out split, indicating that ordering contributes, while stronger proposals remain the main avenue for improvement.

### Learned component

The learned component is intentionally lightweight. It is a pairwise near/far ranker trained on synthetic scenes. For matched layer pairs it uses depth, geometry, and boundary features and predicts which layer should be in front. This strengthens the experimental design without requiring a large end-to-end training setup.

### Why depth is necessary

Segmentation identifies which pixels belong together, but not which semantic region should appear in front when layers overlap or must be recomposed. A layered representation without depth-aware order is fragile for editing, parallax, and occlusion reasoning. Depth converts a collection of masks into a coherent stack.

### Why graph edges matter

Graph edges store pairwise occlusion evidence. That gives the representation structure: which layers touch, which layer is in front, and how confident that relation is. Failures are easier to inspect and explain with graph metadata than with an unordered folder of PNGs.

## Representative artifacts

- `runs/demo_grounded_depthpro_final/debug/ordered_layer_contact_sheet.png`
- `runs/demo_grounded_depthpro_final/debug/segmentation_overlay.png`
- `runs/demo_grounded_depthpro_final/debug/layer_graph.json`
- `docs/RESULTS_SUMMARY_CURRENT.md`

## Closing statement

The main value of the project is not unrestricted layer generation. Its value is making single-image layer decomposition explicit, inspectable, and benchmarkable.
