# Viva Notes

## 30-second pitch

LayerForge-X converts a single RGB image into an interpretable layered scene representation rather than a flat set of masks. Each output layer is RGBA and carries semantic grouping, explicit near-to-far ordering, depth statistics, amodal support, and optional albedo/shading. The core contribution is not a new foundation model; it is the Depth-Aware Amodal Layer Graph plus a benchmarking protocol that measures segmentation, ordering, recomposition, and editability.

## What is actually novel here?

- the output is a graph-structured layer representation, not just exported masks
- ordering is explicit and depth-aware rather than implicit
- amodal masks, background completion, and intrinsic appearance are attached to the same layer objects
- the repo supports a hybrid setting where external RGBA decompositions such as Qwen can be enriched with explicit graph metadata
- the project includes a synthetic benchmark and a learned ordering ablation instead of only qualitative demos

## What should you avoid claiming?

- do not say the system "solves" single-image layer decomposition
- do not say hidden content is recovered exactly; it is estimated or plausibly completed
- do not claim to beat Qwen-Image-Layered in general image-layer generation
- do not imply the entire pipeline is end-to-end trained

## Best honest positioning against Qwen

Qwen-Image-Layered is the strongest recent open baseline for generating semantically disentangled RGBA layers from one image. LayerForge-X is different in emphasis: it turns decomposition into an inspectable scene representation with explicit ordering, occlusion edges, amodal masks, background-completion metadata, intrinsic layers, and evaluation hooks. Qwen is better framed as a frontier baseline and an optional proposal source, not as something to ignore.

## If asked why the synthetic numbers are low

The fast deterministic baseline deliberately uses classical segmentation so it can run anywhere and serve as a controlled reference point. That baseline over-segments badly, producing about 65 predicted layers for 5 ground-truth layers on the synthetic benchmark. Because of that, mean IoU and pairwise ordering accuracy are bottlenecked by proposal quality. The learned ranker still improves recomposition PSNR on the held-out split, which shows ordering matters, but the strongest next gain will come from stronger proposals such as Grounded-SAM2 or Qwen layers.

## If asked what the learned component is

The learned component is intentionally lightweight. It is a pairwise near/far ranker trained on synthetic scenes. For matched layer pairs it uses depth, geometry, and boundary features and predicts which layer should be in front. This makes the project experimentally stronger without requiring a huge end-to-end training setup.

## If asked why depth is needed when segmentation already exists

Segmentation tells you what pixels belong together, but not which semantic region is in front when layers overlap or need to be recomposed. A layered representation without depth-aware order is fragile for editing, parallax, and occlusion reasoning. Depth is what turns a bag of masks into a usable stack.

## If asked why graph edges matter

The graph edges carry pairwise occlusion evidence. That gives the representation structure: which layers touch, which one is in front, and how confident that relation is. It is much easier to inspect failures and explain behavior with graph metadata than with a folder of PNGs.

## Strongest demo artifacts to show

- `runs/demo_grounded_depthpro_final/debug/ordered_layer_contact_sheet.png`
- `runs/demo_grounded_depthpro_final/debug/segmentation_overlay.png`
- `runs/demo_grounded_depthpro_final/debug/layer_graph.json`
- `docs/RESULTS_SUMMARY_2026_04_19.md`

## Best closing line

The project’s main value is not that it is the best possible layer generator. The value is that it makes single-image layer decomposition explicit, inspectable, and benchmarkable, which is exactly what a course project should optimise for.
