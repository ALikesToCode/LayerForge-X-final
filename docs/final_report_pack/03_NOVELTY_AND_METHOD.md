# Novelty and Method: LayerForge-X

## Core idea

LayerForge-X is best described as a **Depth-Aware Amodal Layer Graph**, or **DALG** when that phrase gets tiring.

The output isn't:

```text
image → segmentation masks → PNG files
```

It's:

```text
image → graph of editable scene layers
```

Each graph node is a layer. Each graph edge is an ordering or occlusion relation. The renderer walks the graph and produces a depth-ordered RGBA stack on demand.

---

# 1. Representation: Depth-Aware Amodal Layer Graph

The graph itself:

```text
G = (V, E)
```

Each node `v_i ∈ V` is one semantic layer:

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

Each edge `e_ij ∈ E` encodes one relation between two nodes:

```text
v_i occludes v_j
v_i is in front of v_j
v_i is behind v_j
v_i is adjacent to v_j
v_i belongs with v_j as an associated effect
```

Final RGBA layers come out of a topological sort of the graph, in either near-to-far or far-to-near order depending on what the renderer wants.

## Why this is novel enough for a project

Most baselines export independent masks and call it a day. DALG explicitly stores seven things a segmentation mask doesn't:

1. semantic identity,
2. depth order,
3. soft alpha,
4. amodal extent,
5. completion information,
6. intrinsic appearance,
7. graph relations and confidence.

That combination is what makes the output a scene representation rather than a flat output.

---

# 2. Novel component A: Boundary-Weighted Occlusion Graph

## Problem

Sorting layers by global mean or median depth is where most first-attempt pipelines quietly fall over. The failure mode is easy to describe: a large floor region might have near pixels at the bottom of the image and far pixels at the top. Its median depth ends up somewhere in the middle, and that median can rank the floor incorrectly against a person actually standing on it.

## Proposed solution

Instead of sorting by global statistics, infer pairwise ordering from local depth near shared boundaries.

For each adjacent pair of masks `(i, j)`:

1. Find the boundary pixels of `M_i` and `M_j`.
2. Find the shared / contact boundary region.
3. Sample predicted depth near that boundary.
4. Compute robust local depth statistics.
5. Add a directed edge from the near layer to the far layer, annotated with confidence.

## Formula

Let `B_ij` be pixels near the boundary between layers `i` and `j`:

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

## Cycle resolution

Depth predictions aren't internally consistent, and they do produce cycles:

```text
person in front of chair
chair in front of table
table in front of person
```

The resolution objective is the obvious one — find an order that maximises satisfied weighted edges:

```text
argmax_order sum_{i,j} w_ij * 1[order_i before order_j]
```

In practice the approximation used is much cheaper:

```text
rank_i = weighted_out_degree_i - weighted_in_degree_i
sort by rank_i
```

Alternatively, iteratively drop the lowest-confidence cycle edge until the graph becomes a DAG.

## Claim

> We replace global depth sorting with a boundary-weighted occlusion graph, improving ordering for large background stuff regions and partially overlapping objects.

This is a genuine, small algorithmic contribution — not a world-shaking one, but it's real and it can be isolated in the ablations.

---

# 3. Novel component B: Lightweight Layer Order Ranker

This is the easiest way to make the project feel *experimental* rather than just engineering, and it is now implemented in the repo as `src/layerforge/ranker.py`.

## Idea

Train a small model to predict whether layer `i` is in front of layer `j`. No giant neural network needed — the implemented version uses a lightweight logistic model trained in NumPy on the synthetic benchmark scenes.

## Input features

For each candidate pair `(i, j)`:

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

From synthetic composites or RGB-D data:

```text
y_ij = 1 if i is nearer than j else 0
```

## Training data

The implemented source is:

- Synthetic-LayerBench, which gives exact z-order by construction.

RGB-D supervision remains future work, not a claimed completed result.

## Model

Implemented option:

```text
lightweight logistic model with pairwise depth / geometry / boundary features
```

## Evaluation

The completed ablation in the repo compares:

```text
Global median depth sorting
Boundary depth sorting
Learned pairwise ranker
```

Using:

```text
mean best IoU
PLOA
recomposition PSNR / SSIM
```

## Claim

> We introduce a lightweight pairwise layer-order ranker trained on synthetic layered composites, and on the held-out synthetic split it improves recomposition PSNR over the boundary-only ordering baseline.

Compact, doable, and — compared to "we trained a massive model" — easy to defend.

---

# 4. Novel component C: Visible-vs-amodal dual masks

## Problem

A visible (modal) mask only contains observed pixels. Editing needs the hidden extent too.

## Proposed representation

For every object layer, store two masks:

```text
M_visible: what is currently visible
M_amodal: estimated full object extent
```

The hidden region is the difference:

```text
M_hidden = M_amodal - M_visible
```

And export all three as separate files:

```text
layer_007_visible_rgba.png
layer_007_amodal_mask.png
layer_007_hidden_completion_rgba.png
```

---

# 5. Novel component D: Frontier candidate bank and self-evaluation

## Problem

Once the repo included native LayerForge runs, recursive peeling, and fair Qwen hybrids, the next question was no longer "which single pipeline is the method?" The real question became:

```text
which decomposition should be trusted for this image?
```

That is a different problem from segmentation or ordering alone. Some images benefit from a compact generative RGBA stack. Others benefit from the larger but more explicit native graph. Others are better handled by recursive peeling because the iterative residuals expose hidden structure.

## Proposed solution

Instead of forcing one path, LayerForge-X now runs a **frontier candidate bank** and selects the best editable representation per image using measured evidence.

Current candidate families in the repo:

```text
LayerForge native
LayerForge peeling
Qwen raw
Qwen + graph preserve
Qwen + graph reorder
```

The implementation lives in:

- `src/layerforge/proposals.py`
- `src/layerforge/self_eval.py`
- `scripts/run_frontier_comparison.py`

## Candidate scoring

Each candidate is scored after it actually runs. The current self-evaluation stage is intentionally explicit rather than pretending to be a learned black box.

The score combines:

```text
recomposition fidelity
edit-preservation and anti-trivial copy penalties
semantic separation
alpha quality
graph confidence
runtime
```

The implemented weighted score is:

```text
score = 0.20 * recomposition_fidelity
      + 0.25 * edit_preservation
      + 0.20 * semantic_separation
      + 0.10 * alpha_quality
      + 0.15 * graph_confidence
      + 0.10 * runtime
```

where:

- `recomposition_fidelity` is computed from normalised PSNR and SSIM;
- `edit_preservation` rewards remove/move/recolor responses that affect the target layer while preserving the rest of the frame;
- `semantic_separation` rewards foreground coverage without collapsing everything into one dominant layer;
- `alpha_quality` favours soft, non-binary alpha mattes when the data supports them;
- `graph_confidence` rewards explicit graph output, ordered layers, and occlusion edges;
- `runtime` is a light preference for cheaper candidates when quality is otherwise close.

## Why this matters

This changes the repo from:

```text
one pipeline with a few toggles
```

to:

```text
a self-evaluating layered-representation system
```

That is a more current framing for the project because the frontier is no longer about one monolithic decomposition method. It is about combining strong proposal sources and selecting the best editable scene representation with evidence.

## Claim

> We extend LayerForge-X with a frontier candidate bank and a self-evaluation stage that compares native, generative, hybrid, and recursive decompositions, then selects the most useful editable representation per image using measured fidelity, editability, semantic separation, graph confidence, alpha quality, and runtime signals.

## Fallback method

If no amodal model is used at all, there's a usable geometric fallback:

1. Detect occlusion boundaries using nearby layers and depth discontinuities.
2. Expand the mask only behind closer occluders.
3. Apply class-dependent expansion limits (e.g. don't let a person's amodal mask grow indefinitely).
4. Smooth the resulting amodal mask.
5. Inpaint hidden colour inside `M_hidden`.

## Strong method

Plug in SAMEO or another amodal-SAM-style backend when compute and weights are available.

## Claim

> We distinguish visible support from estimated amodal support, allowing the exported layers to represent both observed content and plausible hidden continuation.

---

# 5. Novel component D: Graph-guided recursive semantic peeling

## Problem

One-shot decomposition freezes the full scene in place before any hidden content has been revealed. That forces ordering, hidden support, and background completion to compete inside a single pass.

## Proposed method

Instead of extracting every layer once, iterate:

```text
I_0 = input RGB
for t in 1..T:
    propose layers on I_{t-1}
    choose the frontmost editable entity from the current graph
    export RGBA_t
    inpaint the residual canvas to obtain I_t
repeat until only background remains
```

At each iteration, store:

```text
iteration_t/input.png
iteration_t/selected_mask.png
iteration_t/selected_layer.png
iteration_t/residual_inpainted.png
```

## Why it matters

This reformulates the pipeline from:

```text
segment once -> sort once -> export once
```

to:

```text
extract front layer -> reveal residual -> repeat
```

That makes the next layer and the completed background explicit rather than implied.

## Claim

> We add a graph-guided recursive peeling mode that repeatedly extracts the current frontmost editable layer, inpaints the residual canvas, and updates the ordered scene representation.

This is the strongest "frontier" addition in the current repo state because it changes the decomposition formulation rather than only tuning a component.

---

# 6. Novel component E: Associated-effect layers

## Problem

Moving an object without its shadow or reflection looks fake. Standard segmentation ignores these effects entirely.

## Proposed representation

For selected foreground objects, estimate an effect mask:

```text
M_effect_i = soft region near object boundary with low-frequency intensity/color change
```

Common examples:

```text
person_shadow
car_reflection
smoke_or_transparency
contact_shadow
```

Two possible representations:

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

A basic shadow heuristic that works surprisingly often:

1. Find pixels near the bottom or contact region of the object mask.
2. Look for connected darkened regions relative to local background.
3. Restrict the expansion by direction and distance from the object.
4. Export as a low-alpha effect layer.

## Claim

> We add optional associated-effect layers so that object edits can preserve shadows or other local visual effects.

This one is best framed as exploratory unless the results turn out to be very clean.

---

# 7. Novel component F: Layer-local intrinsic decomposition

## Problem

Intrinsic image methods generally operate on full images, but editing happens per layer. A naive approach — run intrinsic separately per masked layer — creates obvious artifacts at mask boundaries.

## Proposed method

Run intrinsic decomposition globally first, then mask afterwards:

```text
I ≈ A * S + residual
```

Apply each layer's alpha:

```text
A_i = A ⊙ alpha_i
S_i = S ⊙ alpha_i
```

And enforce visible-recomposition consistency per layer:

```text
I_i ≈ A_i * S_i
```

Export alongside the normal RGBA:

```text
layer_i_rgba.png
layer_i_albedo_rgba.png
layer_i_shading_rgba.png
```

## Claim

> We expose per-layer albedo and shading controls, enabling recoloring and shading edits while preserving the original layer alpha.

Useful, but stretch — keep it in the report as a bonus contribution rather than a central claim.

---

# 8. Full pipeline

## Step 1: Layer proposal

Choose one:

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

Collapse fragments into higher-level semantic groups:

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

Pick one, or ensemble several:

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

Refine the masks using:

```text
mask confidence
image gradients
boundary blur
matting backend if available
```

## Step 5: Boundary-weighted occlusion graph

Build graph edges from local depth evidence around shared boundaries (see §2 above).

## Step 6: Amodal expansion

Estimate full object masks and identify hidden regions.

## Step 7: Completion / recursive residual update

Inpaint background and hidden regions using one of:

```text
OpenCV Telea fallback
LaMa backend
Diffusion inpainting backend
```

## Step 8: Intrinsic split

Run the Retinex fallback or a Marigold-IID-style model.

## Step 9: Export

Write out:

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
```

---

# 9. Main contributions section

A clean version of the contributions bullet list suitable for the report:

## Contributions

1. **Depth-Aware Amodal Layer Graph.** We formulate single-image layer extraction as a graph representation in which nodes store semantic RGBA layers, visible and amodal masks, depth statistics, soft alpha, completion state, and optional intrinsic appearance, while edges encode occlusion and depth relations.

2. **Boundary-weighted layer ordering.** We infer occlusion order using boundary-local depth evidence rather than only global mean or median depth, improving cases involving large stuff regions, slanted surfaces, and partially overlapping objects.

3. **Promptable semantic layer extraction.** The system supports both closed-set panoptic segmentation and open-vocabulary grounded segmentation, enabling user-controllable extraction of layers such as "person," "window," "left chair," or "red car."

4. **Recursive peeling and completion-aware editing.** The system supports both one-shot layer export and graph-guided recursive peeling, where a frontmost layer is extracted, the residual scene is completed, and the graph is updated for the next iteration.

5. **Amodal and effect-aware layers.** The system separates visible masks from estimated amodal masks and can optionally export associated-effect layers so that edits preserve shadows or other local visual effects.

6. **Multi-axis benchmark.** We evaluate not only segmentation, but also depth-order accuracy, recomposition fidelity, amodal completion, intrinsic decomposition, and editability.

---

# 10. Safe novelty claims

These are the phrasings I'd recommend — strong enough to mean something, soft enough to survive scrutiny:

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

# 11. Claims to avoid

And the phrasings to steer clear of, each paired with a defensible substitute:

Don't write:

```text
Our method solves single-image layered decomposition.
```

Do write:

```text
Our method provides a practical and inspectable approximation to single-image layered decomposition.
```

Don't write:

```text
Our method recovers true hidden object appearance.
```

Do write:

```text
Our method synthesizes plausible hidden/background content for editing; quantitative evaluation is performed where ground truth is available.
```

Don't write:

```text
Our method produces physically correct albedo and shading.
```

Do write:

```text
Our intrinsic split is intended as an editable appearance factorization and is evaluated as an approximation.
```

---

# 12. Best title and abstract

## Title

**LayerForge-X: Depth-Aware Amodal Layer Graphs from a Single RGB Image**

## Abstract

Single RGB images collapse object identity, occlusion, transparency, illumination, and depth into a single raster canvas, making local editing and parallax manipulation difficult. We present LayerForge-X, a modular system that converts one RGB image into a depth-aware amodal layer graph. Each graph node stores a semantic RGBA layer with a visible mask, a soft alpha matte, an estimated amodal extent, depth statistics, optional completed content, and optional intrinsic albedo and shading factors. Graph edges encode occlusion and near-to-far ordering inferred from boundary-local monocular depth evidence. The graph is exported as ordered RGBA layers, semantic group layers, completed background, intrinsic appearance layers, and editing previews. We evaluate the representation using visible-grouping metrics on public datasets, pairwise depth-order accuracy, recomposition fidelity, amodal mask and completion metrics, and editing demonstrations including object removal, movement, parallax, and recolouring. The results indicate that combining semantic proposals, monocular geometry, soft alpha refinement, amodal reasoning, and completion yields more editable and interpretable layers than visible-mask baselines.

---

# 13. Method section skeleton

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

Panoptic segmentation and open-vocabulary alternatives; pick per-config.

## 4.3 Depth and geometry

Monocular depth estimation and its normalisation.

## 4.4 Occlusion graph

Boundary-weighted edge construction; the key algorithmic section.

## 4.5 Alpha refinement

Hard masks, feathering, gradient-aware alpha, and an optional matting backend.

## 4.6 Amodal masks and completion

Visible / amodal distinction, hidden-region definition, and inpainting.

## 4.7 Recursive peeling and completion

Iterative extraction of frontmost layers with residual inpainting.

## 4.8 Intrinsic decomposition

Albedo and shading export, per layer.

## 4.9 Rendering and export

Alpha compositing and the set of output files.
