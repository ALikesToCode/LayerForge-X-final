# Novelty and Method: LayerForge-X

## Core idea

LayerForge-X should be described as a **Depth-Aware Amodal Layer Graph**, abbreviated **DALG**.

The output is not merely:

```text
image → segmentation masks → PNG files
```

The output is:

```text
image → graph of editable scene layers
```

Each graph node is a layer. Each graph edge is an ordering/occlusion relation. The renderer converts the graph into a depth-ordered RGBA stack.

---

# 1. Representation: Depth-Aware Amodal Layer Graph

Define the graph:

```text
G = (V, E)
```

Each node `v_i ∈ V` represents one semantic layer:

```text
v_i = {
    label_i,                 semantic label
    group_i,                 person / animal / vehicle / furniture / stuff / background
    M_i^vis,                 visible/modal binary mask
    A_i,                     soft alpha matte
    M_i^amo,                 amodal/full mask
    C_i,                     visible RGBA color layer
    H_i,                     hidden/completed content, optional
    D_i,                     depth statistics
    R_i, S_i,                albedo and shading, optional
    bbox_i, area_i,          geometry metadata
    u_i                      uncertainty score
}
```

Each edge `e_ij ∈ E` represents one relation:

```text
v_i occludes v_j
v_i is in front of v_j
v_i is behind v_j
v_i is adjacent to v_j
v_i belongs with v_j as an associated effect
```

Final RGBA layers are produced by topologically sorting the graph from near to far or far to near depending on rendering convention.

## Why this is novel enough for a project

Most baseline systems export independent masks. DALG explicitly stores:

1. semantic identity,
2. depth order,
3. soft alpha,
4. amodal extent,
5. completion information,
6. intrinsic appearance,
7. graph relations and confidence.

That is a richer, inspectable scene representation.

---

# 2. Novel component A: Boundary-Weighted Occlusion Graph

## Problem

Sorting layers by global mean or median depth fails often. Example: a large floor region may have near pixels at the bottom and far pixels at the top. A person standing on the floor may have a median depth that incorrectly ranks the whole floor against the person.

## Proposed solution

Infer pairwise order using local evidence near shared boundaries.

For each pair of adjacent masks `(i, j)`:

1. Find boundary pixels of `M_i` and `M_j`.
2. Find shared/contact boundary region.
3. Sample predicted depth near that boundary.
4. Compute robust local depth statistics.
5. Add a directed edge from near layer to far layer with confidence.

## Formula

Let `B_ij` be pixels near the boundary between layers `i` and `j`.

```text
z_i^B = median(D[p] for p in dilate(M_i) ∩ B_ij)
z_j^B = median(D[p] for p in dilate(M_j) ∩ B_ij)
```

If smaller depth means nearer:

```text
score(i in front of j) = sigmoid((z_j^B - z_i^B) / sigma)
```

Add edge:

```text
if score > theta:
    i → j
else if score < 1 - theta:
    j → i
else:
    ambiguous edge
```

Confidence:

```text
w_ij = |z_i^B - z_j^B| * shared_boundary_length(i,j) * alpha_boundary_confidence
```

## Cycle resolution

Depth predictions can create cycles:

```text
person in front of chair
chair in front of table
table in front of person
```

Resolve cycles by finding an order that maximizes satisfied weighted edges:

```text
argmax_order sum_{i,j} w_ij * 1[order_i before order_j]
```

Practical approximation:

```text
rank_i = weighted_out_degree_i - weighted_in_degree_i
sort by rank_i
```

or use iterative removal of the lowest-confidence cycle edge.

## Claim

> We replace global depth sorting with a boundary-weighted occlusion graph, improving ordering for large background stuff regions and partially overlapping objects.

This is a real algorithmic contribution.

---

# 3. Novel component B: Lightweight Layer Order Ranker

This is the easiest way to make the project look genuinely experimental.

## Idea

Train a small model to predict whether layer `i` is in front of layer `j`.

It does not require training a huge neural net. A logistic regression, random forest, or gradient-boosted tree is enough.

## Input features

For every candidate pair `(i, j)`:

```text
Δ median depth
Δ boundary median depth
Δ minimum depth
Δ vertical centroid
Δ area
bbox overlap
shared boundary length
semantic pair features
T-junction heuristic count
alpha boundary confidence
is_thing_i, is_thing_j
is_stuff_i, is_stuff_j
```

## Label

From synthetic data or RGB-D data:

```text
y_ij = 1 if i is nearer than j else 0
```

## Training data

- Synthetic-LayerBench gives exact z-order.
- NYU Depth V2 / DIODE give approximate depth order from ground-truth depth.

## Model

Recommended:

```text
LogisticRegression(class_weight='balanced')
```

or:

```text
RandomForestClassifier(n_estimators=200, max_depth=8)
```

## Evaluation

Compare:

```text
Global median depth sorting
Boundary depth sorting
Learned pairwise ranker
```

Metrics:

```text
PLOA
BW-PLOA
Occlusion Edge F1
Kendall tau
```

## Claim

> We introduce a lightweight pairwise layer-order ranker trained on synthetic layered composites and/or RGB-D data, improving occlusion graph consistency over global depth statistics.

This is compact, doable, and credible.

---

# 4. Novel component C: Visible-vs-amodal dual masks

## Problem

A visible mask only contains observed pixels. Editing needs hidden extents.

## Proposed representation

For every object layer, store two masks:

```text
M_visible: what is currently visible
M_amodal: estimated full object extent
```

The hidden region is:

```text
M_hidden = M_amodal - M_visible
```

Export both:

```text
layer_007_visible_rgba.png
layer_007_amodal_mask.png
layer_007_hidden_completion_rgba.png
```

## Fallback method

If no amodal model is used:

1. Detect occlusion boundaries using nearby layers and depth discontinuities.
2. Expand the mask only behind closer occluders.
3. Use class-dependent expansion limits.
4. Smooth the amodal mask.
5. Inpaint hidden color inside `M_hidden`.

## Strong method

Use SAMEO / amodal SAM-style backend when available.

## Claim

> We distinguish visible support from estimated amodal support, allowing the exported layers to represent both observed content and plausible hidden continuation.

---

# 5. Novel component D: Associated-effect layers

## Problem

Moving an object without its shadow or reflection looks fake. Standard segmentation ignores object-associated effects.

## Proposed representation

For selected foreground objects, estimate an effect mask:

```text
M_effect_i = soft region near object boundary with low-frequency intensity/color change
```

Examples:

```text
person_shadow
car_reflection
smoke_or_transparency
contact_shadow
```

Represent as either:

```text
same object layer with extended alpha
```

or:

```text
separate associated effect layer linked by graph edge
```

Graph relation:

```text
person_core --associated_effect--> person_shadow
```

## Simple heuristic

For shadows:

1. Find pixels near the bottom/contact region of object mask.
2. Search for connected darkened regions relative to local background.
3. Restrict by direction and distance from object.
4. Export as low-alpha effect layer.

## Claim

> We add optional associated-effect layers so that object edits can preserve shadows or other local visual effects.

This should be framed as exploratory unless results are strong.

---

# 6. Novel component E: Layer-local intrinsic decomposition

## Problem

Intrinsic image methods operate on full images, but editing often happens per layer.

## Proposed method

Run intrinsic decomposition globally to avoid mask-boundary artifacts:

```text
I ≈ A * S + residual
```

Then apply layer alpha:

```text
A_i = A ⊙ alpha_i
S_i = S ⊙ alpha_i
```

For each layer, enforce visible recomposition:

```text
I_i ≈ A_i * S_i
```

Export:

```text
layer_i_rgba.png
layer_i_albedo_rgba.png
layer_i_shading_rgba.png
```

## Claim

> We expose per-layer albedo and shading controls, enabling recoloring and shading edits while preserving the original layer alpha.

This is useful, but keep it as a stretch contribution.

---

# 7. Full pipeline

## Step 1: Layer proposal

Use one of:

```text
classical components
Mask2Former panoptic
GroundingDINO + SAM2
Florence-2 + SAM2
```

Output:

```text
visible masks + labels + confidence
```

## Step 2: Semantic merging

Merge fragments into semantic groups:

```text
person
animal
vehicle
furniture
plant
background-stuff
text/graphic
effect/unknown
```

## Step 3: Depth / geometry

Use one or an ensemble:

```text
Depth Anything V2
Depth Pro
Marigold
MoGe
```

Output:

```text
depth map
optional normals / point map
confidence map
```

## Step 4: Soft alpha

Refine masks using:

```text
mask confidence
image gradients
boundary blur
matting backend if available
```

## Step 5: Boundary-weighted occlusion graph

Build graph edges using local depth evidence around shared boundaries.

## Step 6: Amodal expansion

Estimate full object masks and hidden regions.

## Step 7: Completion

Inpaint background and hidden regions:

```text
OpenCV Telea fallback
LaMa backend
Diffusion inpainting backend
```

## Step 8: Intrinsic split

Run Retinex fallback or Marigold-IID-style model.

## Step 9: Export

Export:

```text
ordered individual RGBA layers
grouped semantic RGBA layers
visible and amodal masks
background completion
layer graph JSON
albedo/shading RGBA layers
parallax preview
metrics report
```

---

# 8. Main contributions section

Use this in the report.

## Contributions

1. **Depth-Aware Amodal Layer Graph.** We formulate single-image layer extraction as a graph representation in which nodes store semantic RGBA layers, visible and amodal masks, depth statistics, soft alpha, completion state, and optional intrinsic appearance, while edges encode occlusion and depth relations.

2. **Boundary-weighted layer ordering.** We infer occlusion order using boundary-local depth evidence rather than only global mean or median depth, improving cases involving large stuff regions, slanted surfaces, and partially overlapping objects.

3. **Promptable semantic layer extraction.** The system supports both closed-set panoptic segmentation and open-vocabulary grounded segmentation, enabling user-controllable extraction of layers such as “person,” “window,” “left chair,” or “red car.”

4. **Amodal and completion-aware editing.** The system separates visible masks from estimated amodal masks and uses inpainting to synthesize hidden or background regions, supporting object removal, movement, and parallax.

5. **Multi-axis benchmark.** We evaluate not only segmentation, but also depth-order accuracy, recomposition fidelity, amodal completion, intrinsic decomposition, and editability.

---

# 9. Safe novelty claims

These are strong and defensible.

```text
We propose a modular framework for converting a single RGB image into a depth-aware amodal layer graph rather than a set of independent visible segmentation masks.
```

```text
We introduce boundary-weighted occlusion graph construction for more reliable near-to-far layer ordering than global depth statistics.
```

```text
We evaluate layered image decomposition across segmentation, depth ordering, recomposition, completion, intrinsic decomposition, and editing tasks.
```

```text
We provide a synthetic layered benchmark with ground-truth RGBA layers, depth order, modal/amodal masks, and clean backgrounds for controlled ablation.
```

---

# 10. Claims to avoid

Do not write:

```text
Our method solves single-image layered decomposition.
```

Do write:

```text
Our method provides a practical and inspectable approximation to single-image layered decomposition.
```

Do not write:

```text
Our method recovers true hidden object appearance.
```

Do write:

```text
Our method synthesizes plausible hidden/background content for editing; quantitative evaluation is performed where ground truth is available.
```

Do not write:

```text
Our method produces physically correct albedo and shading.
```

Do write:

```text
Our intrinsic split is intended as an editable appearance factorization and is evaluated as an approximation.
```

---

# 11. Best title and abstract

## Title

**LayerForge-X: Depth-Aware Amodal Layer Graphs from a Single RGB Image**

## Abstract

Single RGB images collapse object identity, occlusion, transparency, illumination, and depth into a single raster canvas, making local editing and parallax manipulation difficult. We present LayerForge-X, a modular system that converts one RGB image into a depth-aware amodal layer graph. Each graph node stores a semantic RGBA layer with visible mask, soft alpha matte, estimated amodal extent, depth statistics, optional completed content, and optional intrinsic albedo/shading factors. Graph edges encode occlusion and near-to-far ordering inferred from boundary-local monocular depth evidence. The graph is exported as ordered RGBA layers, semantic group layers, completed background, intrinsic appearance layers, and editing previews. We evaluate the representation using panoptic segmentation metrics, pairwise depth-order accuracy, recomposition fidelity, amodal mask/completion metrics, and editing demonstrations including object removal, movement, parallax, and recoloring. The results show that combining semantic proposals, monocular geometry, soft alpha refinement, amodal reasoning, and completion yields more editable and interpretable layers than visible-mask baselines.

---

# 12. Method section skeleton

## 4.1 Problem definition

Given an input image:

```text
I ∈ [0,1]^{H×W×3}
```

infer `K` layers:

```text
L_k = (C_k, A_k, y_k, z_k, M_k^vis, M_k^amo)
```

where:

```text
C_k ∈ [0,1]^{H×W×3}
A_k ∈ [0,1]^{H×W}
y_k is semantic label
z_k is depth/order score
M_k^vis is visible mask
M_k^amo is amodal mask
```

The layers should satisfy approximate recomposition:

```text
I ≈ Render(L_1, ..., L_K, order)
```

## 4.2 Layer proposal

Describe panoptic/open-vocabulary segmentation.

## 4.3 Depth and geometry

Describe monocular depth estimation and normalization.

## 4.4 Occlusion graph

Describe boundary-weighted edge construction.

## 4.5 Alpha refinement

Describe hard masks, feathering, gradient-aware alpha, matting backend.

## 4.6 Amodal masks and completion

Describe visible/amodal distinction and inpainting.

## 4.7 Intrinsic decomposition

Describe albedo/shading export.

## 4.8 Rendering and export

Describe alpha compositing and file outputs.
