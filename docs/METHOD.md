# Method

## Problem

One RGB image goes in. A stack of semantic RGBA layers comes out — layers that can be recomposed, reordered, edited, recoloured, relit, or animated. That's the whole pitch.

## DALG: Depth-Aware Amodal Layer Graph

Each node in the graph carries more than a cutout. Concretely, a node stores:

- the visible mask and a soft alpha;
- a semantic label and a higher-level semantic group;
- three depth statistics — median depth, a near percentile, a far percentile;
- an amodal mask estimating the full extent including hidden parts;
- an RGBA albedo layer;
- an RGBA shading layer;
- and occlusion links to other nodes.

## Pipeline

1. **Segment proposal.** Pick a segmenter: SLIC for the deterministic fallback, Mask2Former for closed-set panoptic segmentation, or GroundingDINO + SAM2 for open-vocabulary prompts.
2. **Depth estimation.** One of: the geometric luminance baseline, Depth Anything V2, Depth Pro, Marigold, or an ensemble.
3. **Overlap resolution.** Where proposed masks overlap, pixels are assigned to the nearer or higher-priority mask.
4. **Stuff-plane splitting.** Large regions like sky, road, wall, or background get split into depth quantile planes so they don't dominate the ordering.
5. **Alpha refinement.** Binary masks are converted to soft alpha, with depth edges used to harden boundaries where the depth map says there's genuine discontinuity.
6. **Ordering and occlusion graph.** Adjacent regions with distinct depth near their shared boundary become front-to-back edges. The default fast path uses this boundary-local evidence directly; the learned variant trains a lightweight pairwise ranker on synthetic scenes and uses the predicted near/far probabilities to sort the layer stack.
7. **Amodal completion.** Masks are expanded conservatively through closing, hole filling, and hull-limited dilation.
8. **Background completion.** Removable foreground is inpainted to create an edit-ready background layer.
9. **Intrinsic split.** Albedo and shading are computed globally and then masked per layer.
10. **Export.** Ordered stack, grouped stack, graph JSON, metrics, and debug panels.

## Learned pairwise ordering

The optional learned ordering module is deliberately lightweight:

- training data comes from the synthetic LayerBench scenes already used for benchmarking;
- predicted segments are matched to ground-truth layers by IoU;
- matched layer pairs generate feature vectors built from depth statistics, mask overlap/boundary evidence, and region geometry;
- a small logistic model is fit in NumPy and saved as a JSON ranker;
- inference uses the pairwise scores to produce a global near-to-far order.

This keeps the learned component easy to reproduce on a single GPU workstation while still giving the report a real "classical heuristic vs learned ordering" ablation.

## Defensible novelty

The honest place to plant the novelty flag is the representation and the fusion, not any individual module:

> A single-image editable layer graph that combines semantic grouping, monocular geometry, amodal extent, soft alpha, completed background, and intrinsic appearance factors in a single inspectable output.

I am very explicitly *not* claiming to have invented SAM, depth estimation, or intrinsic decomposition. The contribution is that they get integrated into a coherent, benchmarkable layered representation, and that the representation is the graph rather than a bag of PNGs.

## Current benchmark readout

The completed benchmark runs in the repo support the following measured claim:

- on a held-out synthetic split, the learned pairwise ranker improves recomposition PSNR from `19.1589` to `19.4138` while leaving mean best IoU (`0.1549`) and pairwise layer-order accuracy (`0.1667`) unchanged.

That pattern is plausible: ordering gets slightly better for compositing, but the headline order metric is bottlenecked by over-segmentation in the deterministic classical proposal stage. The right way to present that is as evidence that the ordering module helps, but that better proposals are still the main lever.
