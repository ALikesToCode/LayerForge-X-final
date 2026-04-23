# 4. Method

## 4.1 Core idea

LayerForge-X is organized around a **Depth-Aware Amodal Layer Graph (DALG)**. The system does not treat decomposition as a one-time conversion from an image into a folder of masks. Instead, it models the scene as a graph of editable layer objects. Each node stores one layer together with semantic, geometric, and appearance metadata; each edge stores an ordering or occlusion relation. The renderer traverses this graph and emits a depth-ordered RGBA stack, design manifest, metrics, and editability diagnostics.

## 4.2 Representation

Let the graph be

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

Each edge `e_ij ∈ E` represents one relation between two nodes:

```text
v_i occludes v_j
v_i is in front of v_j
v_i is behind v_j
v_i is adjacent to v_j
v_i is linked to v_j as an associated effect
```

The exported RGBA stack is obtained from a topological sort of the graph, either in near-to-far or far-to-near order depending on the rendering path.

## 4.3 Boundary-weighted occlusion graph

### Motivation

Global mean or median depth is often insufficient for ordering. Large floor or wall regions can span both near and far pixels, which makes global statistics unreliable relative to nearby objects.

### Construction

For each adjacent pair of masks `(i, j)`:

1. find the boundary pixels of `M_i` and `M_j`;
2. isolate the shared or contact boundary region;
3. sample predicted depth near that boundary;
4. compute robust local depth statistics;
5. add a directed edge from the near layer to the far layer when the evidence is strong enough.

Let `B_ij` denote pixels near the boundary between layers `i` and `j`:

```text
z_i^B = median(D[p] for p in dilate(M_i) ∩ B_ij)
z_j^B = median(D[p] for p in dilate(M_j) ∩ B_ij)
```

With the convention that smaller depth means nearer:

```text
score(i in front of j) = sigmoid((z_j^B - z_i^B) / sigma)
```

Edge decision:

```text
if score > theta:
    i → j
else if score < 1 - theta:
    j → i
else:
    ambiguous edge
```

Confidence weight:

```text
w_ij = |z_i^B - z_j^B| * shared_boundary_length(i,j) * alpha_boundary_confidence
```

### Cycle handling

Monocular depth predictions are not guaranteed to be globally consistent and may induce cycles. LayerForge-X resolves these cases by deriving a weighted rank from graph structure:

```text
rank_i = weighted_out_degree_i - weighted_in_degree_i
```

Layers are then sorted by this score, or low-confidence cycle edges are removed until the graph becomes acyclic.

## 4.4 Lightweight layer-order ranker

The repository also includes an optional learned ordering component implemented in `src/layerforge/ranker.py`. The model is intentionally lightweight: a logistic pairwise near/far ranker trained in NumPy on synthetic LayerBench scenes.

Input features for each candidate pair `(i, j)` include:

```text
Δ median depth
Δ boundary median depth
Δ minimum depth
Δ vertical centroid
Δ area
bbox overlap
shared boundary length
semantic pair indicators
T-junction heuristic count
alpha boundary confidence
is_thing_i, is_thing_j
is_stuff_i, is_stuff_j
```

Labels are constructed from synthetic z-order:

```text
y_ij = 1 if i is nearer than j else 0
```

The current measured ablation uses this ranker to compare global median sorting, boundary-depth sorting, and learned pairwise ordering on held-out synthetic scenes.

## 4.5 Visible and amodal dual masks

For every object-like layer, LayerForge-X stores both visible and amodal support:

```text
M_visible: what is observed
M_amodal: estimated full object extent
M_hidden = M_amodal - M_visible
```

This separation is important because visible masks alone are insufficient for object movement, occlusion reasoning, and removal-based editing. The repository exports visible masks, amodal masks, and hidden-region completions separately when those artifacts are available.

## 4.6 Frontier candidate bank and self-evaluation

Once the repository contained native LayerForge runs, recursive peeling, and Qwen-based hybrids, the central question became representation selection rather than single-pipeline identification:

```text
Which decomposition should be trusted for a given image?
```

LayerForge-X therefore evaluates a **frontier candidate bank** and selects the strongest editable representation per image using explicit metrics. The current candidate families are:

```text
LayerForge native
LayerForge peeling
Qwen raw
Qwen + graph preserve
Qwen + graph reorder
```

The selector is implemented in:

- `src/layerforge/proposals.py`
- `src/layerforge/self_eval.py`
- `scripts/run_frontier_comparison.py`

The self-evaluation score is intentionally explicit. It combines:

```text
recomposition fidelity
edit-preservation penalties against copy-like decompositions
semantic separation
alpha quality
graph confidence
```

In the committed five-image frontier summary, the active weighted score is

```text
score = 0.20 * recomposition_fidelity
      + 0.25 * edit_preservation
      + 0.20 * semantic_separation
      + 0.10 * alpha_quality
      + 0.15 * graph_confidence
```

The implementation still supports an optional runtime term for future reruns with fresh timings, but the shipped frontier summary was rescored from cached runs and therefore keeps runtime inactive. This formulation still turns the repository into a self-evaluating layer-representation system rather than a single fixed pipeline.

## 4.7 Recursive peeling

One-shot decomposition forces ordering, hidden support, and background completion into a single pass. LayerForge-X adds a second path based on **graph-guided recursive peeling**:

```text
I_0 = input RGB
for t in 1..T:
    propose layers on I_{t-1}
    choose the frontmost editable entity from the current graph
    export RGBA_t
    inpaint the residual canvas to obtain I_t
repeat until only background remains
```

Each iteration stores:

```text
iteration_t/input.png
iteration_t/selected_mask.png
iteration_t/selected_layer.png
iteration_t/residual_inpainted.png
```

This formulation makes the next layer and the residual canvas explicit rather than implicit.

## 4.8 Associated-effect layers

Standard segmentation omits shadows, reflections, and similar local residual effects. LayerForge-X therefore includes an optional associated-effect path that estimates a low-alpha effect region near the selected foreground object.

The representation can be modeled either as an extended object alpha or as a separate effect layer linked by a graph edge:

```text
person_core --associated_effect--> person_shadow
```

The current repository uses a lightweight heuristic:

1. identify pixels near the bottom or contact region of the object mask;
2. search for connected darkened or structured residual regions relative to a local background estimate;
3. restrict the expansion by distance and direction relative to the object;
4. export the result as a low-alpha effect layer.

This component is presented as a prototype rather than a solved visual-effects decomposition system.

## 4.9 Layer-local intrinsic decomposition

Intrinsic image methods typically operate on full images, while editing operates on layers. LayerForge-X therefore runs intrinsic decomposition globally and masks the result afterward:

```text
I ≈ A * S + residual
```

For each layer:

```text
A_i = A ⊙ alpha_i
S_i = S ⊙ alpha_i
```

with per-layer consistency:

```text
I_i ≈ A_i * S_i
```

The repository exports:

```text
layer_i_rgba.png
layer_i_albedo_rgba.png
layer_i_shading_rgba.png
```

These layers support recoloring and simple shading edits, while the report treats the factorization as an approximation rather than a physically exact intrinsic decomposition.

## 4.10 Full pipeline

### Step 1: Layer proposal

One of:

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

### Step 2: Semantic merging

Fragments are collapsed into higher-level semantic groups:

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

### Step 3: Depth and geometry

One or more of:

```text
Depth Anything V2
Depth Pro
Marigold
MoGe
```

Outputs:

```text
depth map
optional normals / point map
confidence map
```

### Step 4: Soft alpha

Masks are refined using:

```text
mask confidence
image gradients
boundary blur
matting backend if available
```

### Step 5: Boundary-weighted occlusion graph

Graph edges are built from local depth evidence around shared boundaries.

### Step 6: Amodal expansion

Full object masks and hidden regions are estimated conservatively.

### Step 7: Completion and residual update

Background and hidden regions are completed using one of:

```text
OpenCV Telea fallback
LaMa backend
Diffusion inpainting backend
```

### Step 8: Intrinsic split

The system runs the Retinex fallback or a stronger intrinsic backend when available.

### Step 9: Export

LayerForge-X writes:

```text
ordered individual RGBA layers
grouped semantic RGBA layers
visible and amodal masks
background completion
layer graph JSON
albedo/shading RGBA layers
iteration artifacts for recursive peeling
optional associated-effect RGBA layers
parallax preview
metrics report
canonical DALG manifest
```

## 4.11 Problem definition

Given an input image

```text
I ∈ [0,1]^{H×W×3}
```

the objective is to infer `K` layers

```text
L_k = (C_k, A_k, y_k, z_k, M_k^vis, M_k^amo)
```

where

```text
C_k ∈ [0,1]^{H×W×3}
A_k ∈ [0,1]^{H×W}
y_k is a semantic label
z_k is a depth/order score
M_k^vis is the visible mask
M_k^amo is the amodal mask
```

The layers should satisfy approximate recomposition:

```text
I ≈ Render(L_1, ..., L_K, order)
```
