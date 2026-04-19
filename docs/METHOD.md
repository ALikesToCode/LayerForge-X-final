# Method

## Problem

Input: one RGB image. Output: a stack of semantic RGBA layers that can be recomposed, reordered, edited, recolored, relit, or animated.

## DALG: Depth-Aware Amodal Layer Graph

Each layer node stores:

- visible mask and soft alpha;
- semantic label and group;
- depth median, near percentile, far percentile;
- amodal mask;
- albedo RGBA;
- shading RGBA;
- occlusion links to other nodes.

## Pipeline

1. **Segment proposal:** SLIC, Mask2Former, or GroundingDINO + SAM2.
2. **Depth estimation:** geometric baseline, Depth Anything V2, Depth Pro, Marigold, or ensemble.
3. **Overlap resolution:** pixels are assigned to nearer/higher-priority masks.
4. **Stuff-plane splitting:** sky/road/wall/background regions are split into depth quantile planes.
5. **Alpha refinement:** binary masks are converted to soft alpha; depth edges harden boundaries.
6. **Occlusion graph:** adjacent regions with distinct depth form front-to-back edges.
7. **Amodal completion:** masks are conservatively expanded through closing, hole filling, and hull-limited dilation.
8. **Background completion:** removable foreground is inpainted to create an edit-ready background layer.
9. **Intrinsic split:** albedo and shading are saved per layer.
10. **Export:** ordered stack, grouped stack, graph JSON, metrics, and debug panels.

## Defensible novelty

The novelty is the representation and fusion layer:

> A single-image editable layer graph that combines semantic grouping, monocular geometry, amodal extent, soft alpha, completed background, and intrinsic appearance factors.

Do not claim to invent SAM, depth estimation, or intrinsic decomposition. Claim that you integrate them into a coherent, benchmarkable layered representation.
