# LayerForge-X Final Report

## Abstract

Single-image editing systems increasingly need structured scene representations rather than a flat RGB bitmap or a folder of visible cutouts. LayerForge-X addresses that need by exporting a **Depth-Aware Amodal Layer Graph (DALG)**: ordered RGBA layers with semantic grouping, soft alpha, occlusion metadata, optional amodal support, background completion, intrinsic appearance factors, and editability-oriented diagnostics. The system combines native decomposition, Qwen/external RGBA enrichment, recursive peeling, promptable extraction, transparent-layer recovery, and explicit self-evaluation so different candidate representations can be compared under one graph contract. This report focuses on the measured behavior of those components, the benchmark protocol used to evaluate them, and the practical limits that still separate the current implementation from fully generative layered scene understanding.

## 1. Introduction

The core goal is not just to decompose an image, but to convert it into an **editable scene asset**. That requires more than segmentation. A useful representation needs explicit near-to-far ordering, soft alpha boundaries, at least heuristic amodal support, some notion of hidden/background completion, and export surfaces that support real editing workflows. LayerForge-X therefore treats the scene graph as the canonical object and regards PNG stacks, debug artifacts, and design-manifest exports as projections of that graph.

## 2. Contributions

This report makes six concrete claims:

1. LayerForge-X implements a depth-aware amodal layer graph rather than a simple bag of masks.
2. The repo includes a fair Qwen comparison with preserve/reorder hybrid modes and a common evaluation frame.
3. Recursive peeling is implemented as a measured alternative path rather than only a conceptual extension.
4. The evaluation stack now includes anti-trivial editability metrics, not only recomposition fidelity.
5. Promptable extraction and transparent decomposition are both implemented as measured benchmarked components.
6. The system exports a canonical DALG manifest and a product-facing design-manifest projection suitable for future API/editor integration.

\newpage

# 3. Related Work

## Project framing

The project looks at **single-image layered scene decomposition**: given one RGB bitmap, infer a set of re-composable RGBA layers that are semantically meaningful, ordered by depth and occlusion, and — as a stretch goal — decomposed into albedo and shading. The fundamental difficulty is that a raster image collapses a lot of scene factors into one 2D array: object identity, transparency, shadows, reflections, illumination, camera projection, and occlusion. Any strong layered representation therefore has to borrow from several different sub-fields at once: scene understanding, monocular geometry, matting, completion, and appearance decomposition.

LayerForge-X positions itself as a **Depth-Aware Amodal Layer Graph (DALG)** rather than a plain segmentation exporter. Every layer is a graph node that carries a visible mask, a soft alpha, a semantic label, depth statistics, an estimated amodal extent, completed hidden or background content, and optional intrinsic appearance. Graph edges encode near/far and occludes/occluded-by relations. The intent is that the output is useful for editing, parallax, object removal, relighting-style operations, and general downstream analysis — not just "a folder of PNGs."

---

## 1. Layered depth and image-based rendering

The oldest directly relevant representation is the **Layered Depth Image (LDI)**. Shade et al. introduced LDIs as an image-based rendering structure where a single camera view can store **multiple samples along one line of sight** instead of just the first visible surface. That's the right historical starting point because the present project also needs more than a flat visible mask: it needs depth ordering and some notion of hidden-region reasoning.

LDIs work well for novel-view rendering because disocclusions can be handled more gracefully than with a flat depth map. They were not, however, designed to be editable semantic layers. A classical LDI doesn't really tell you which layer is "person," "chair," "sky," or "road," and it doesn't separate albedo from shading or split foreground effects from object cores.

**Use in this project:** DALG borrows the idea of storing layered samples along viewing rays, but makes the layers semantic, alpha-composited, and editable.

---

## 2. 3D photography and depth inpainting

Shih et al.'s **3D Photography using Context-Aware Layered Depth Inpainting** converts an RGB-D image into a multi-layer representation aimed at parallax rendering. Their system uses an LDI with explicit pixel connectivity and performs colour-and-depth inpainting in occluded regions. The paper is directly relevant because it stitches together layered depth, inpainting, and interactive view synthesis in one pipeline — exactly the kind of integration this project is trying to do, just for a different target task.

The catch is that 3D Photography assumes RGB-D input or externally supplied depth, and the target is novel-view synthesis rather than semantic object editing. It doesn't output object-level semantic RGBA layers like "people," "vehicles," "furniture," and "background stuff."

**Use in this project:** LayerForge-X treats context-aware hidden-region completion as *one stage* inside a semantic layer graph. The output can drive parallax, but its primary purpose is object editing.

---

## 3. Scene decomposition, object layers, and completed occluded content

A few older works sit closer to the full scene-decomposition goal. Dhamo et al.'s **Object-Driven Multi-Layer Scene Decomposition From a Single Image** builds an LDI from a single RGB image and uses object information to infer occluded intermediate layers. Zheng et al.'s **Layer-by-Layer Completed Scene Decomposition** studies decomposition into individual objects, occlusion relationships, amodal masks, and content completion — a list that could honestly serve as the target for this project.

These works matter because they make the case that correct layering requires more than visible segmentation. Occluded regions are part of the representation. If a person is standing in front of a car, a practical layer representation should estimate the hidden continuation of the car, at least enough to support removal or parallax.

**Use in this project:** LayerForge-X follows this line but modernises the toolchain: foundation segmentation models, current monocular geometry, promptable / open-vocabulary controls, and explicit evaluation of recomposition and editability.

---

## 4. Video layer decomposition and omnimattes

Layered decomposition has also been studied heavily in video. **Omnimatte** decomposes a video into object-associated RGBA layers that can include not only the object but also related visual effects — shadows, smoke, reflections. That's conceptually important, because a clean object cutout is frequently not enough for editing: moving a person without their shadow immediately looks fake.

Generative Omnimatte and a handful of follow-ups push this idea further with stronger generative priors. But video methods have a free lunch that single-image methods don't: motion and temporal consistency. Without optical flow or multiple frames, the task is genuinely more ambiguous.

**Use in this project:** LayerForge-X can optionally export associated-effect layers (object core, shadow / reflection / effect region, clean background). Even a simple shadow-layer heuristic makes the output read as serious layer decomposition rather than plain segmentation.

---

## 5. Modern generative layer decomposition

This is the corner of the field that makes the project topic especially timely. **LayerDecomp** targets image layer decomposition with visual effects, producing a clean background and a transparent foreground while preserving effects like shadows and reflections. **DiffDecompose** studies layer-wise decomposition of alpha-composited images, particularly transparent and semi-transparent cases. **Qwen-Image-Layered** proposes an end-to-end diffusion model that decomposes one RGB image into multiple semantically disentangled RGBA layers. **Referring Layer Decomposition** frames the task as prompt-conditioned RGBA layer extraction.

All four sit very close to the project statement, and the report should cite them specifically to show awareness of the frontier. The important boundary to draw is that LayerForge-X should not claim to beat these large generative systems on raw image quality — it almost certainly can't, and claiming otherwise would be unconvincing. The claim is a different one: a transparent, modular, geometry-aware layer graph that can be benchmarked component-by-component instead of only eyeballed.

**Use in this project:** These works define the current research frontier. LayerForge-X positions itself as a practical, inspectable alternative that fuses segmentation, geometry, amodal reasoning, alpha refinement, inpainting, and intrinsic decomposition.

---

## 6. Panoptic and open-vocabulary segmentation

Layered decomposition needs both object and stuff regions. Panoptic segmentation is a natural fit because it unifies instance-level things (people, animals, vehicles) with amorphous stuff (sky, road, wall, grass, water). The panoptic segmentation paper also gave us the unified **Panoptic Quality (PQ)** metric.

**Mask2Former** is the closed-set baseline of choice — it's a universal segmentation architecture that handles semantic, instance, and panoptic segmentation with a single head. Its obvious limitation is that it's bounded by the training label vocabulary.

Open-vocabulary models cover that gap. **GroundingDINO** detects objects specified by free-form text (either category names or referring expressions), and **SAM / SAM2** produces promptable segmentation masks. Stitching these together allows prompts like "left chair," "red car," "window," or "foreground person" — exactly the kind of control a layer editor actually wants.

**Use in this project:** The report should compare closed-set panoptic segmentation and open-vocabulary grounded segmentation side by side. The second is more exciting, but the first has the advantage of being trivial to benchmark with standard PQ / mIoU.

---

## 7. Monocular depth and geometry

Depth ordering is central, and the obvious first heuristic — sort by average object depth — is often wrong. Large regions such as walls, floors, tables, and roads span a wide depth range, so their "average depth" doesn't correspond to where they actually sit relative to neighbours.

Modern monocular geometry models provide strong priors that make depth-based ordering viable at all. **Depth Anything V2** improves robustness and detail over older monocular depth models using synthetic labelled data plus large-scale pseudo-labelled real data. **Depth Pro** produces sharp metric depth from a single image without camera intrinsics. **Marigold** repurposes diffusion priors for affine-invariant depth estimation. **MoGe** goes further and predicts a fuller monocular geometry package — point maps, depth, normals, camera field of view.

**Use in this project:** Rather than sorting layers by global median depth, LayerForge-X infers pairwise ordering from *boundary-local* depth evidence. That's more robust when objects overlap and when background regions cover both near and far pixels.

---

## 8. Amodal segmentation and occlusion reasoning

Visible (modal) masks describe only what can be seen. Amodal segmentation estimates the full object extent, including the invisible occluded parts. **KINS** is a key amodal instance segmentation benchmark, and newer foundation-model-based work such as **SAMEO** adapts Segment Anything-style mask decoders to occluded objects.

Amodal reasoning matters because editing demands it. If a user removes a foreground object, the background has to be completed. If a user moves a partially occluded object, its hidden parts may need to be hallucinated. This is inherently ambiguous, so completion outputs should be treated as *plausible* rather than ground-truth-accurate on real images.

**Use in this project:** LayerForge-X reports modal visible masks separately from amodal masks. That avoids over-claiming and makes the representation honest.

---

## 9. Alpha matting and edge quality

Hard segmentation masks make jagged cutouts. Real layers need soft alpha around hair, fur, glass, motion blur, antialiased vector edges, smoke, and any semi-transparent object. Matting methods estimate a fractional alpha matte directly.

**Matting Anything** is a relevant reference because it combines SAM features with a lightweight mask-to-matte module and supports visual or linguistic prompts — which plays well with this project's open-vocabulary layer extraction direction.

**Use in this project:** Even if the implementation uses a simple boundary-feathering fallback, the report should explicitly compare hard alpha to soft alpha. The visual difference is usually obvious even at thumbnail size.

---

## 10. Inpainting and hidden-region completion

Editing layered content requires plausible content behind removed or moved objects. **LaMa** is the reliable baseline here: it was explicitly designed for large masks and high-resolution generalisation, using Fourier convolutions and large-mask training.

Inpainting should not be framed as "ground truth recovery" on real images. For a real photograph the method cannot know what's behind the object; it can only produce a plausible completion. For synthetic composites where the hidden background is genuinely known, inpainting is quantifiable.

**Use in this project:** Use synthetic data to score background completion with PSNR / SSIM / LPIPS inside removed-object regions. Use real images for qualitative object-removal and parallax demos.

---

## 11. Intrinsic images: albedo and shading

Intrinsic image decomposition splits an image into reflectance / albedo and illumination / shading. The problem is badly under-constrained from a single image — one image, two unknowns per pixel. **Intrinsic Images in the Wild (IIW)** introduced a large in-the-wild benchmark using human reflectance judgments and WHDR. Recent diffusion-based intrinsic methods (Marigold-IID and friends) give stronger modern baselines.

For this project, intrinsic decomposition should be framed as a stretch module rather than the core contribution. A Retinex-style fallback is fine, but the report should be honest about the fact that physical correctness is limited.

**Use in this project:** Export per-layer albedo and shading as useful editing approximations. Evaluate with IIW/WHDR when possible; otherwise fall back to synthetic scenes with known albedo and shading.

---

## Gap summary

A compact view of what each prior area covers, where it stops, and how LayerForge-X uses it:

| Prior area | What it solves | What it misses for this project | How LayerForge-X uses it |
|---|---|---|---|
| LDI / image-based rendering | Multi-depth samples for novel views | No semantic/editable object layers | Use layered depth idea as representation backbone |
| 3D photo inpainting | Parallax and hidden-region completion | Usually not semantic object/stuff editing | Add semantic graph nodes and RGBA export |
| Panoptic segmentation | Things + stuff parsing | No depth, alpha, hidden content | Provides layer proposals |
| Open-vocabulary segmentation | User-specified object masks | No ordering/completion by itself | Enables promptable layer extraction |
| Monocular depth | Per-pixel relative/metric depth | No object graph or masks | Supplies geometry for ordering |
| Amodal segmentation | Full object extent under occlusion | Does not complete appearance alone | Supplies hidden masks |
| Matting | Soft alpha boundaries | Usually foreground/background only | Refines layer alpha |
| Inpainting | Plausible missing content | No semantic/depth ordering | Completes background/hidden regions |
| Intrinsic images | Albedo/shading factors | Highly ambiguous; not layer-aware | Optional per-layer appearance split |
| Generative layer decomposition | End-to-end RGBA layers | Often black-box and hard to benchmark component-wise | Used as frontier comparison and motivation |

---

## Report-ready related-work paragraph

This is the paragraph-length version suitable for dropping straight into the report:

Single-image layered scene decomposition sits at the intersection of image-based rendering, scene parsing, monocular geometry, amodal perception, matting, inpainting, and intrinsic image decomposition. Classical Layered Depth Images show why a single visible surface per pixel is insufficient for view synthesis, since disoccluded content requires multiple depth and colour samples along camera rays. Modern 3D-photo methods extend this with learned colour-and-depth inpainting, but primarily target parallax rather than semantic object editing. Panoptic segmentation provides a natural source of object and stuff proposals, while open-vocabulary detectors and promptable segmenters allow user-specified layer extraction beyond fixed label sets. Recent monocular depth and geometry models improve the reliability of depth ordering, but depth alone does not produce editable layers. Amodal segmentation and inpainting address the invisible portions of occluded objects and backgrounds, while matting improves layer boundaries. Recent generative layer-decomposition systems demonstrate the importance of RGBA layers for editing, but their end-to-end nature makes component-wise analysis difficult. Our work therefore proposes an inspectable Depth-Aware Amodal Layer Graph that combines semantic masks, depth ordering, soft alpha, amodal extent, completion, and optional intrinsic decomposition into a re-composable representation.


\newpage

# 4. Method

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


\newpage

# 5. Experiments and Evaluation Protocol

## Goal

The project shouldn't be evaluated with one vague "looks good" comparison. A layered representation has several measurable properties, each of which can fail independently:

1. Are the layer regions semantically correct?
2. Is the near-to-far depth and occlusion order correct?
3. Do the layers recompose back into the original image?
4. Are alpha boundaries usable for editing?
5. Are hidden and background regions completed plausibly?
6. Does the intrinsic split behave sensibly?
7. Does the representation actually support edits better than baselines?

Because any of those can be wrong while the others look fine, the benchmark runs on multiple tracks.

For the current repository state, treat `PROJECT_MANIFEST.json`, `report_artifacts/metrics_snapshots/*.json`, and `report_artifacts/command_log.md` as the source of truth for reported numbers. `docs/RESULTS_SUMMARY_CURRENT.md` is the human-readable bridge to those artifacts.

---

# Track A: Modal semantic / panoptic segmentation

## Datasets

- **COCO Panoptic** for common objects and stuff.
- **ADE20K** for dense scene parsing and a broader set of stuff classes.
- Optional: a small hand-labelled project test set for whichever domains the demos lean on.

## Metrics

### Panoptic Quality

When ground-truth panoptic annotations are available, report PQ:

```text
PQ = sum_{(p,g) in TP} IoU(p,g) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Also report the two components:

```text
SQ = segmentation quality over matched segments
RQ = recognition quality
```

### Semantic mIoU

For semantic-only labels, the standard per-class IoU averaged over classes:

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU = mean_c IoU_c
```

## Baselines

| Baseline | Purpose |
|---|---|
| SLIC / classical connected components | Low-level non-semantic baseline |
| Mask2Former panoptic | strong closed-set panoptic baseline |
| GroundingDINO + SAM2 | open-vocabulary promptable baseline |
| Florence-2 + SAM2, optional | prompt-conditioned alternative |

## Report table

| Method | Dataset | group mIoU ↑ | thing mIoU ↑ | stuff mIoU ↑ | mean image mIoU ↑ | Avg layers | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | COCO-val subset | | | | | | |
| Mask2Former | COCO-val subset | | | | | | |
| Grounded-SAM2 | curated prompts | | | | | | |
| LayerForge-X | mixed | | | | | | |

For the current repo state, note the distinction clearly:

- the implemented COCO and ADE20K evaluators are **coarse-group IoU** benchmarks rather than official PQ pipelines;
- PQ/SQ/RQ stay as the report-template target if a full panoptic evaluator is added later;
- do not relabel the current JSON summaries as PQ.

---

# Track B: Depth-order and occlusion-graph quality

## Why this matters

The output is a stack, so order is as important as mask quality. Get the ordering wrong and recomposition breaks, parallax looks incoherent, and the whole representation stops being useful even if each individual mask is perfect. Global average-depth sorting fails when objects are large, slanted, or span multiple depth planes — exactly the cases most scenes contain — so evaluation has to measure pairwise ordering, not just a global ranking.

## Datasets

- **Synthetic-LayerBench**: generated composites with known z-order.
- **NYU Depth V2**: indoor RGB-D scenes with object and instance labels.
- **DIODE**: indoor / outdoor RGB-D for generalisation.
- Optional: KITTI for outdoor road scenes if vehicles and roads are a focus.

## Ground-truth pair construction

For each image, define a ground-truth depth value per layer using median ground-truth depth inside the visible mask:

```text
z_i = median(Depth_GT[p] for p in visible_mask_i)
```

Then, for each candidate pair `(i, j)`, include it only if:

```text
|z_i - z_j| > tau_depth
```

That threshold rejects near-ties, which otherwise get penalised as "wrong" even when the order is genuinely ambiguous.

## Metrics

### Pairwise Layer Order Accuracy (PLOA)

```text
PLOA = (# correctly ordered valid pairs) / (# valid pairs)
```

A pair is correct if:

```text
sign(pred_depth_i - pred_depth_j) == sign(gt_depth_i - gt_depth_j)
```

Stick to one depth convention consistently: here, smaller depth means nearer when using metric depth.

### Boundary-Weighted PLOA (BW-PLOA)

Weight pairs by shared boundary length or adjacency confidence, so pairs that actually touch count more than pairs sitting in different corners of the image:

```text
BW-PLOA = sum_{i,j} w_ij * correct_ij / sum_{i,j} w_ij
```

Recommended weight:

```text
w_ij = shared_boundary_length(i,j) * min(area_i, area_j)^0.5
```

### Occlusion Edge F1

Build a ground-truth occlusion graph from synthetic z-order or RGB-D boundary reasoning. Then compare predicted graph edges as a set:

```text
Precision = correct_pred_edges / pred_edges
Recall    = correct_pred_edges / gt_edges
F1        = 2PR / (P + R)
```

### Kendall tau / inversion count

For images with a total ground-truth layer order, also report Kendall tau or the normalised inversion count — it's a cleaner single number than PLOA when the order is fully defined.

## Report table

| Method | Depth source | Ordering rule | PLOA ↑ | BW-PLOA ↑ | Occlusion F1 ↑ | Kendall τ ↑ |
|---|---|---|---:|---:|---:|---:|
| No depth | none | layer area / heuristic | | | | |
| Luminance depth | grayscale | global median | | | | |
| Depth Anything V2 | monocular | global median | | | | |
| Depth Pro | metric mono | global median | | | | |
| LayerForge-X | ensemble | boundary graph | | | | |

---

# Track C: RGBA recomposition fidelity

## Why this matters

If the exported layers are correct, alpha-compositing them in predicted order should recover the input image closely. This track is essentially a sanity check on the representation as a whole.

## Rendering equation

For layers composited far-to-near:

```text
C_out = alpha_over(L_1, L_2, ..., L_K)
```

Equivalently: start from the farthest layer and alpha-over each nearer layer on top.

## Metrics

### Pixel reconstruction

```text
MAE  = mean(|I - I_hat|)
MSE  = mean((I - I_hat)^2)
PSNR = 20 log10(MAX_I / sqrt(MSE))
```

### Structural similarity

SSIM or MS-SSIM.

### Perceptual similarity

LPIPS if it's available. Lower is better.

### Alpha coverage error

Compare the summed alpha against the valid image area. For opaque natural images, summed alpha should cover the whole scene:

```text
coverage_error = mean(|clip(sum_k alpha_k, 0, 1) - 1|)
```

## Report table

| Method | Hard/soft alpha | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Alpha coverage err ↓ | Edge artifacts ↓ |
|---|---|---:|---:|---:|---:|---:|
| Hard masks | hard | | | | | |
| Feathered masks | soft | | | | | |
| Depth-aware alpha | soft | | | | | |
| Matting backend | soft | | | | | |

---

# Track D: Amodal mask and hidden-region completion

## Datasets

- **Synthetic-LayerBench**: exact full masks and hidden pixels are known.
- **KINS**: driving-scene amodal instance segmentation.
- **COCOA / COCO-A**: amodal object annotations where available.
- **MP3D-Amodal**, if accessible, for real indoor amodal masks from 3D data.

## Metrics

### Modal IoU — visible mask quality

```text
IoU_visible = |M_pred_visible ∩ M_gt_visible| / |M_pred_visible ∪ M_gt_visible|
```

### Amodal IoU — full object extent quality

```text
IoU_amodal = |M_pred_amodal ∩ M_gt_amodal| / |M_pred_amodal ∪ M_gt_amodal|
```

### Invisible-region IoU

The hardest and most informative of the three, because it isolates just the hidden portion:

```text
M_invisible = M_amodal - M_visible
IoU_invisible = IoU(M_pred_invisible, M_gt_invisible)
```

### Background-completion quality

On synthetic composites where the clean background is known:

```text
PSNR_masked, SSIM_masked, LPIPS_masked
```

These are computed only inside the removed or hidden region.

## Report table

| Method | Amodal module | Inpaint module | Visible IoU ↑ | Amodal IoU ↑ | Invisible IoU ↑ | Masked LPIPS ↓ |
|---|---|---|---:|---:|---:|---:|
| Visible masks only | none | none | | | | |
| Heuristic expansion | shape prior | OpenCV | | | | |
| SAMEO-style | amodal model | OpenCV | | | | |
| Full LayerForge-X | amodal + depth | LaMa/diffusion | | | | |

---

# Track E: Intrinsic albedo/shading split

## Datasets

- **IIW** for reflectance-order judgments and WHDR.
- **Synthetic-LayerBench-Intrinsic** with known albedo and shading.
- Optional: MIT Intrinsic Images if a small controlled set is enough.

## Metrics

### WHDR on IIW

WHDR measures whether predicted reflectance comparisons agree with human judgments. Lower is better.

### Synthetic intrinsic metrics

If ground-truth albedo `A` and shading `S` are available:

```text
MSE_albedo
MSE_shading
Scale-invariant MSE
Layer-local color constancy error
```

### Recomposition consistency

Per layer:

```text
I_layer ≈ A_layer * S_layer + residual_layer
```

Scored as:

```text
intrinsic_recompose_error = mean(|I_layer - A_layer * S_layer| inside alpha > 0)
```

## Report table

| Method | WHDR ↓ | Albedo MSE ↓ | Shading MSE ↓ | Recompose error ↓ | Notes |
|---|---:|---:|---:|---:|---|
| Retinex fallback | | | | | fast but approximate |
| Marigold-IID | | | | | stronger external backend |
| LayerForge-X per-layer | | | | | mask-aware export |

---

# Track F: Editability evaluation

## Why this matters

The whole point of layers is editing. The most compelling final section should demonstrate that the representation supports practical operations, not just that it scores well on segmentation benchmarks.

## Edits to evaluate

1. **Object removal** — remove one foreground layer and show the completed background.
2. **Object translation** — move one layer sideways and see whether background holes stay plausible.
3. **Parallax preview** — shift layers according to depth to simulate viewpoint change.
4. **Depth-of-field edit** — blur far layers more than near layers.
5. **Albedo recolour** — recolour an object through its albedo while preserving shading.
6. **Relighting-lite** — scale or alter the shading layer.

## Metrics

### Non-edited region preservation

Outside the edit mask, the image should remain unchanged:

```text
preservation_MAE = mean(|I_original - I_edited| outside affected_region)
```

### Hole artifact ratio

After movement or removal, count transparent or invalid pixels:

```text
hole_ratio = invalid_pixels / image_pixels
```

### User preference study

A small study is plenty:

- 10 to 20 images.
- 3 methods: baseline, depth-aware only, full method.
- Question: "Which edit looks more plausible?"
- Report preference percentage.

Even a dozen people can surface systematic differences.

## Report table

| Method | Removal preference ↑ | Move preference ↑ | Parallax artifacts ↓ | Preservation MAE ↓ | Notes |
|---|---:|---:|---:|---:|---|
| Hard segmentation | | | | | jagged edges |
| Soft alpha only | | | | | better boundaries |
| Depth-aware + inpaint | | | | | fewer holes |
| Full LayerForge-X | | | | | best editability |

---

# Synthetic-LayerBench design

This is the easiest path to strong, defensible numbers, precisely because ground truth is known by construction.

## Data generation

Composite scenes from known layers:

```text
background B
for each layer k from far to near:
    choose object sprite / shape / cutout
    assign z_k
    assign semantic class
    assign alpha matte
    optionally assign albedo and shading
    composite using alpha-over
save:
    final RGB image
    each GT RGBA layer
    modal mask
    amodal/full mask
    alpha matte
    depth order
    clean background
    albedo and shading if available
    optional associated-effect layer
```

## Domains

At least three domains to avoid overfitting the benchmark to one look:

| Domain | Why it matters |
|---|---|
| Flat/vector graphics | crisp shapes, clear order |
| Photographic cutouts | realistic textures and boundaries |
| Stylized/anime | line art and cel shading |

## Recommended split

```text
train/dev for order ranker: 300 images
validation: 100 images
test: 100 images
```

If time is genuinely short:

```text
30 synthetic images + 20 real qualitative images
```

Even that is a big improvement over hand-picked demos alone.

## Rich synthetic export now implemented

The repo now supports:

```bash
python scripts/make_synthetic_dataset.py \
  --output data/synthetic_layerbench_pp \
  --count 20 \
  --output-format layerbench_pp \
  --with-effects
```

Per scene, `layerbench_pp` writes:

```text
image.png
layers_near_to_far/
visible_masks/
amodal_masks/
alpha_mattes/
layers_effects_rgba/
intrinsics/albedo.png
intrinsics/shading.png
depth.png
depth.npy
occlusion_graph.json
scene_metadata.json
```

That format is the right one to use for recursive-peeling and effect-layer evaluation because it preserves both the visible scene and the hidden/effect supervision.

---

# Ablation protocol

Run a controlled set where one component changes at a time. The reason to do this rather than one giant comparison is simple: only this kind of diff can attribute credit to an individual component.

| ID | Segmentation | Depth | Ordering | Alpha | Amodal | Inpaint | Intrinsics |
|---|---|---|---|---|---|---|---|
| A | SLIC | luminance | global median | hard | off | off | off |
| B | Mask2Former | none | heuristic | hard | off | off | off |
| C | Mask2Former | Depth Anything V2 | global median | hard | off | off | off |
| D | Mask2Former | Depth Anything V2 | boundary graph | hard | off | off | off |
| E | Grounded-SAM2 | Depth Anything V2 | boundary graph | soft | off | off | off |
| F | Grounded-SAM2 | Depth Pro/MoGe | boundary graph | soft | heuristic | OpenCV | off |
| G | Grounded-SAM2 | ensemble | learned edge ranker | soft/matting | amodal | LaMa | off |
| H | full | ensemble | learned edge ranker | soft/matting | amodal | LaMa | Retinex/Marigold-IID |
| I | full + peel | ensemble | graph-guided peeling | soft/matting | amodal | iterative completion | Retinex/Marigold-IID |

## Expected interpretation

- A → B measures the semantic segmentation benefit.
- B → C measures the depth benefit.
- C → D measures the boundary graph benefit on top of depth.
- D → E measures open-vocabulary plus soft alpha.
- E → F measures amodal and inpaint.
- G → H measures intrinsic split usefulness.
- H → I measures whether recursive peeling improves editability or hidden-region completion beyond the one-shot stack.

---

# Minimum result set for best marks

Figures that should be in the final report, in roughly this order:

1. Input image.
2. Semantic overlay.
3. Depth map.
4. Layer graph visualization.
5. Ordered RGBA contact sheet.
6. Hard-mask baseline vs soft-alpha result.
7. Global-depth ordering vs boundary-graph ordering.
8. Visible mask vs amodal mask.
9. Object removal with background completion.
10. Recursive peeling storyboard.
11. Parallax GIF or frame strip.
12. Albedo/shading layer visualisation.
13. Failure cases.

Tables:

1. Literature comparison table.
2. Benchmark/dataset table.
3. Ablation metrics table.
4. Runtime/memory table.
5. Failure-case taxonomy.

---

# Failure-case taxonomy

Failures are part of the contribution, not a thing to hide. Classifying them makes the report read as mature rather than salesy:

| Failure | Cause | Example | Fix / future work |
|---|---|---|---|
| Wrong semantic grouping | segmenter misses object or merges stuff | chair merged with table | better prompts / panoptic model |
| Wrong depth order | monocular depth ambiguity | mirror/window/flat poster | boundary ranker + uncertainty |
| Jagged edge | hard mask or bad matting | hair/fur | matting backend |
| Missing shadow/effect | object-only mask | person moved without shadow | associated-effect layer |
| Bad inpainting | large unseen region | removed foreground person | diffusion/LaMa inpaint |
| Bad amodal shape | heavy occlusion | hidden car side | amodal model/SAMEO |
| Intrinsic artifacts | single-image ambiguity | texture mistaken as shading | stronger IID model |
| Too many layers | oversegmentation | background split into fragments | graph merging |
| Too few layers | undersegmentation | person + bicycle merged | prompt refinement |

---

# Recommended final benchmark narrative

If the report needs one paragraph summarising the whole evaluation, this is the one:

> We evaluate LayerForge-X across four axes: segmentation quality, layer-order correctness, recomposition fidelity, and editability. Standard panoptic metrics measure visible semantic grouping, while a synthetic layer benchmark and RGB-D datasets measure pairwise depth-order accuracy. Recomposition metrics verify that the exported RGBA stack preserves the original image. Finally, object removal, object movement, parallax, and intrinsic recolouring demonstrate that the representation is genuinely useful for editing rather than being a segmentation visualisation in disguise.


## 6. Results

### Hero figures

#### Native, hybrid, and graph-aware reconstruction

![Truck recomposition comparison](../figures/truck_recomposition_comparison.png){ width=100% }

#### Frontier candidate-bank selection

![Frontier review](../figures/frontier_review.png){ width=100% }

#### Promptable extraction benchmark

![Prompt extraction benchmark](../figures/prompt_extract_benchmark.png){ width=100% }

#### Transparent decomposition benchmark

![Transparent benchmark](../figures/transparent_benchmark.png){ width=100% }

#### Associated-effect prototype

![Associated-effect demo](../figures/effects_layer_demo.png){ width=100% }

### Main measured summary

Abbreviations in the tables below: `LF` = LayerForge, `Q+G` = Qwen plus LayerForge graph enrichment.

#### Five-image Qwen raw versus hybrid review

| Method | Images | Graph | Mean PSNR | Mean SSIM |
|---|---:|---|---:|---:|
| LF native | 5 | yes | 27.3438 | 0.9464 |
| Qwen raw 4 | 5 | no | 29.0757 | 0.8850 |
| Q+G preserve 4 | 5 | yes | 28.5539 | 0.8638 |
| Q+G reorder 4 | 5 | yes | 28.5397 | 0.8637 |

#### Associated-effect demo

| Artifact | Effect detected | Predicted effect px | Ground-truth effect px | Effect IoU |
|---|---|---:|---:|---:|
| `runs/effects_groundtruth_demo_cutting_edge` | yes | 4853 | 13750 | 0.3529 |

#### Five-image frontier candidate-bank review

| Method | Images | Mean PSNR | Mean SSIM | Mean self-eval | Best-image wins |
|---|---:|---:|---:|---:|---:|
| LF native | 5 | 37.6688 | 0.9708 | 0.6283 | 4 |
| LF peel | 5 | 27.0988 | 0.9096 | 0.4783 | 0 |
| Qwen raw 4 | 5 | 29.0757 | 0.8850 | 0.2541 | 0 |
| Q+G preserve 4 | 5 | 28.5539 | 0.8638 | 0.5259 | 0 |
| Q+G reorder 4 | 5 | 28.5397 | 0.8637 | 0.5251 | 1 |

#### Five-image editability suite

| Method | Remove | Move | Recolor | Edit success | Non-edit preserve | Hole ratio |
|---|---:|---:|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 | 0.9999 | 0.4860 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 | 1.0000 | 0.5433 |
| Qwen raw 4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 | 1.0000 | 1.0000 |
| Q+G preserve 4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 | 0.9887 | 0.1420 |
| Q+G reorder 4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 | 0.9886 | 0.1427 |

#### Promptable extraction benchmark

| Prompt type | Queries | Hit rate | Mean IoU | Mean alpha MAE |
|---|---:|---:|---:|---:|
| text | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + point | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + box | 10 | 1.0000 | 0.3776 | 0.1503 |
| point | 10 | 0.0000 | 0.8654 | 0.0222 |
| box | 10 | 0.0000 | 0.8654 | 0.0222 |

#### Transparent benchmark

| Metric | Mean |
|---|---:|
| Transparent alpha MAE | 0.1131 |
| Background PSNR | 25.9863 |
| Background SSIM | 0.9541 |
| Recompose PSNR | 56.0066 |
| Recompose SSIM | 0.9996 |

### Interpretation

- Raw Qwen remains the stronger compact pure-PSNR baseline on the measured five-image sweep.
- Native LayerForge posts the strongest mean SSIM on the same images, at the cost of a larger average stack.
- The measured frontier candidate bank selects `LF native` for `4/5` images, with `Q+G reorder 4` winning the cat scene.
- The `Q+G preserve 4` row is the fair metadata-first hybrid comparison because it keeps the **best external visual order** while adding graph structure, amodal masks, ordering metadata, and intrinsic artifacts.
- The editability suite is the anti-triviality guardrail for the selector, which is why raw Qwen's object-removal response remains near zero despite reasonable recomposition scores.
- Promptable extraction is now a measured component instead of only a CLI feature. Text-bearing prompts currently carry the semantic routing load, while point-only and box-only prompts still need better disambiguation.
- Transparent recomposition is reported as a sanity check; alpha error and clean-background quality are the primary transparent-layer metrics.
- The associated-effect path now has a real exported demo artifact with a materially improved clean-reference rerun, but it must still be framed as an early heuristic rather than a solved component.

## 7. Discussion

The strongest reading of the current results is not that LayerForge-X universally beats generative decomposers on raw pixels. The stronger claim is that it turns native, generative, and recursive decompositions into one explicit editable graph representation with auditable metrics and exportable structure. Qwen remains the right generative RGBA baseline. LayerForge-X remains strongest when framed as a graph-aware, benchmarkable, editability-oriented complement to that frontier.

## 8. Limitations

Failure taxonomy and future-work framing are documented in [04_ABLATIONS_AND_TABLES.md](04_ABLATIONS_AND_TABLES.md) and [02_BENCHMARKING_PROTOCOL.md](02_BENCHMARKING_PROTOCOL.md). The report should explicitly keep:

- wrong semantic grouping;
- wrong depth order;
- jagged alpha boundaries;
- missing shadow/effect layers;
- bad inpainting in large unseen regions;
- bad amodal continuation under heavy occlusion;
- intrinsic split errors;
- point-only and box-only prompt-routing ambiguity;
- transparent-layer recovery that is still approximate rather than generative.

## 9. Conclusion

LayerForge-X is now best understood as a self-evaluating layer-representation system rather than a simple decomposition script. It can produce native graph layers, enrich frontier RGBA layers, run recursive peeling, measure editability, benchmark prompt extraction, approximate transparent recovery, and export a canonical DALG manifest. That combination is the core project contribution.

# 10. References

1. Shade, J., Gortler, S., He, L., and Szeliski, R. Layered Depth Images. SIGGRAPH 1998.
2. Shih, M.-L., Su, S.-Y., Kopf, J., and Huang, J.-B. 3D Photography using Context-aware Layered Depth Inpainting. CVPR 2020.
3. Kirillov, A. et al. Panoptic Segmentation. CVPR 2019.
4. Cheng, B. et al. Mask2Former for Universal Image Segmentation. CVPR 2022.
5. Liu, S. et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV 2024 / arXiv 2023.
6. Ravi, N. et al. SAM 2: Segment Anything in Images and Videos. 2024.
7. Xiao, B. et al. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. CVPR 2024.
8. Bochkovskiy, A. et al. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second. 2024.
9. Yang, Y. et al. Generative Image Layer Decomposition with Visual Effects. CVPR 2025.
10. DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers. arXiv 2025.
11. Referring Layer Decomposition. arXiv 2026.
12. Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition. 2025.
13. Yao, J. et al. Matte Anything. 2024.
14. Suvorov, R. et al. Resolution-robust Large Mask Inpainting with Fourier Convolutions. WACV 2022.
15. Bell, S. et al. Intrinsic Images in the Wild. SIGGRAPH 2014.
16. Tai, Y. et al. Segment Anything, Even Occluded. 2025.


## Appendix A: artifact map

Submission source-of-truth files:

- [../../PROJECT_MANIFEST.json](../../PROJECT_MANIFEST.json)
- [../../report_artifacts/README.md](../../report_artifacts/README.md)
- [../../report_artifacts/metrics_snapshots/](../../report_artifacts/metrics_snapshots)
- [../../report_artifacts/figure_sources/figure_manifest.json](../../report_artifacts/figure_sources/figure_manifest.json)
- [../../report_artifacts/command_log.md](../../report_artifacts/command_log.md)
- [../RESULTS_SUMMARY_CURRENT.md](../RESULTS_SUMMARY_CURRENT.md)
- [../QWEN_IMAGE_LAYERED_COMPARISON.md](../QWEN_IMAGE_LAYERED_COMPARISON.md)
- [../REPORT_TABLES.md](../REPORT_TABLES.md)
- [../FIGURES.md](../FIGURES.md)
- [../PRODUCT_ARCHITECTURE_AND_LAUNCH.md](../PRODUCT_ARCHITECTURE_AND_LAUNCH.md)
- [../api/openapi.yaml](../api/openapi.yaml)

## Appendix B: extended tables and ablations

\newpage

# Appendix B. Extended Tables and Ablations

This appendix collects the extended quantitative tables, ablation templates, and failure-analysis material that support the main report.

## Completed runs snapshot

The rows below are no longer placeholders; they correspond to runs already present in the repo:

| Variant | Segmentation | Depth | Ordering | Split | Mean best IoU | PLOA | Recompose PSNR |
|---|---|---|---|---|---:|---:|---:|
| A1 | classical | geometric luminance | boundary | synthetic fast | 0.1549 | 0.1667 | 19.1360 |
| A2 | classical | geometric luminance | boundary | synth test | 0.1549 | 0.1667 | 19.1589 |
| A3 | classical | geometric luminance | learned ranker | synth test | 0.1549 | 0.1667 | 19.4138 |

Interpretation:

- `A2 → A3` gives a real learned-ordering result worth reporting.
- the current bottleneck is proposal quality, because the fast classical segmenter still produces about `65` predicted layers for `5` ground-truth layers.
- therefore the strongest next qualitative row is the real-image `grounded_sam2 + depth_pro` system, not more tuning on the classical baseline.

## Frontier candidate-bank review

The repo now also contains the measured five-image frontier comparison at `runs/frontier_review/frontier_summary.json`.

| Method | Images | Mean PSNR | Mean SSIM | Mean self-eval score | Best-image wins |
|---|---:|---:|---:|---:|---:|
| LF native | 5 | 37.6688 | 0.9708 | 0.6283 | 4 |
| LF peel | 5 | 27.0988 | 0.9096 | 0.4783 | 0 |
| Qwen raw 4 | 5 | 29.0757 | 0.8850 | 0.2541 | 0 |
| Q+G preserve 4 | 5 | 28.5539 | 0.8638 | 0.5259 | 0 |
| Q+G reorder 4 | 5 | 28.5397 | 0.8637 | 0.5251 | 1 |

Interpretation:

- `LayerForge native` is now the strongest overall candidate-bank row by the repo's explicit self-evaluation score and wins `4/5` measured images once anti-triviality penalties are enabled;
- the hardened selector no longer lets `LayerForge peeling` win the truck image simply because the recursive removal path is visually dramatic;
- `Qwen + graph reorder` now wins the cat image, showing that imported generative stacks can still beat the native path on specific compact scenes;
- `Qwen raw` remains the compact frontier generative baseline, but it is no longer the best overall editable representation once structure and editability are scored explicitly.

## Editability suite snapshot

The frontier review is now paired with an editability suite so recomposition fidelity is not the only score that matters.

| Method | Remove response ↑ | Move response ↑ | Recolor response ↑ | Edit success ↑ | Non-edit preservation ↑ | Background hole ratio ↓ |
|---|---:|---:|---:|---:|---:|---:|
| LF native | 0.1097 | 0.1011 | 0.1220 | 0.6695 | 0.9999 | 0.4860 |
| LF peel | 0.1019 | 0.0808 | 0.1082 | 0.5865 | 1.0000 | 0.5433 |
| Qwen raw 4 | 0.0002 | 0.0001 | 0.0001 | 0.1506 | 1.0000 | 1.0000 |
| Q+G preserve 4 | 0.2083 | 0.1509 | 0.1421 | 0.8633 | 0.9887 | 0.1420 |
| Q+G reorder 4 | 0.2080 | 0.1491 | 0.1421 | 0.8607 | 0.9886 | 0.1427 |

Interpretation:

- the editability suite is the anti-triviality guardrail for the frontier selector;
- `Qwen raw (4)` is the obvious example of why recomposition alone is insufficient, because its remove/move/recolor responses are almost zero while its background-hole ratio is effectively `1.0`;
- the hybrid rows currently post the strongest edit-success scores because imported generative stacks plus explicit LayerForge graph metadata are still easy to move, recolor, and remove cleanly.

## Promptable extraction benchmark snapshot

The prompt-conditioned extraction path is now measured instead of being only a CLI affordance.

| Prompt type | Queries | Target hit rate | Mean target IoU | Mean alpha MAE |
|---|---:|---:|---:|---:|
| text | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + point | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + box | 10 | 1.0000 | 0.3776 | 0.1503 |
| point | 10 | 0.0000 | 0.8654 | 0.0222 |
| box | 10 | 0.0000 | 0.8654 | 0.0222 |

Interpretation:

- text-bearing prompts now hit the intended semantic target on the measured synthetic set;
- point-only and box-only prompts still lock onto a neighboring region with high overlap but wrong semantics;
- the benchmark is therefore useful because it distinguishes semantic hit rate from overlap and alpha quality.

## Transparent benchmark snapshot

The transparent / alpha-composited recovery path now has a measured synthetic benchmark instead of only a qualitative smoke demo.

| Metric | Mean |
|---|---:|
| Transparent alpha MAE | 0.1131 |
| Background PSNR | 25.9863 |
| Background SSIM | 0.9541 |
| Recompose PSNR | 56.0066 |
| Recompose SSIM | 0.9996 |

Interpretation:

- transparent recomposition is a sanity check here; alpha error and clean-background quality are the primary transparent metrics;
- this path should be presented as an approximate transparent-layer recovery mode, not a claim of state-of-the-art generative transparent decomposition;
- the current prototype is strongest on flare-like overlays and weakest on the semi-transparent panel variant;
- despite that, it is now a measured component and belongs in the report as a frontier-aligned extension.

## Main ablation matrix

The core sweep. Each row changes exactly one axis relative to the next so the contribution of each component is readable off the table:

| Variant | Segmentation | Depth | Ordering | Alpha | Amodal | Inpaint | Intrinsic | Purpose |
|---|---|---|---|---|---|---|---|---|
| A | SLIC/classical | luminance | global median | hard | no | no | no | weak baseline |
| B | Mask2Former | none | area/heuristic | hard | no | no | no | semantic-only baseline |
| C | Mask2Former | Depth Anything V2 | global median | hard | no | no | no | tests depth addition |
| D | Mask2Former | Depth Anything V2 | boundary graph | hard | no | no | no | tests graph ordering |
| E | Grounded-SAM2 | Depth Anything V2 | boundary graph | soft | no | no | no | tests promptable masks + alpha |
| F | Grounded-SAM2 | Depth Pro / MoGe | boundary graph | soft | heuristic | OpenCV | no | tests amodal + completion |
| G | Grounded-SAM2 | ensemble | learned edge ranker | soft/matting | amodal | LaMa | no | strong non-intrinsic system |
| H | full | ensemble | learned edge ranker | soft/matting | amodal | LaMa | Retinex / Marigold-IID | full LayerForge-X |
| I | full + peel | ensemble | graph-guided peeling | soft/matting | amodal | iterative completion | Retinex / Marigold-IID | recursive peeling variant |

---

## Table 1: Literature comparison

A gap analysis across the most relevant families of prior work. The `LayerForge-X` row is intentionally the most densely populated — that's the point:

| Method family | Semantic layers | Depth order | Amodal hidden parts | Soft alpha | Inpainting | Intrinsics | Single image | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LDI | no | yes | partial | no | no | no | yes/varies | rendering representation |
| 3D photo inpainting | no/limited | yes | yes | no | yes | no | RGB-D/depth | parallax focus |
| Panoptic segmentation | yes | no | no | no | no | no | yes | visible masks only |
| Grounded-SAM | yes/open vocab | no | no | no | no | no | yes | promptable visible masks |
| Matting | foreground only | no | no | yes | no | no | yes | excellent alpha boundaries |
| Amodal segmentation | object masks | limited | yes | no | sometimes | no | yes | hidden shape, not full layer stack |
| LayerDecomp-style | foreground/background | partial | yes | yes | yes | no | yes | strong generative editing baseline |
| Qwen-Image-Layered-style | yes | implicit | yes | yes | yes | no/limited | yes | end-to-end generative RGBA layers |
| LayerForge-X | yes | explicit graph | yes | yes | yes | optional | yes | modular and benchmarkable |

---

## Table 2: Dataset plan

Which dataset gets used for which track, with the available ground truth in each:

| Dataset | Used for | Ground truth available | Metrics |
|---|---|---|---|
| Synthetic-LayerBench / layerbench_pp | full pipeline | RGBA layers, z-order, masks, clean background, optional albedo/shading, optional effects | PLOA, PSNR, SSIM, LPIPS, alpha MAE, amodal IoU |
| COCO Panoptic | visible semantic grouping | panoptic masks | coarse-group mIoU, thing/stuff mIoU |
| ADE20K | scene/stuff parsing | semantic masks | coarse-group mIoU, pixel accuracy |
| NYU Depth V2 | indoor depth order | RGB-D, labels | AbsRel, RMSE, PLOA |
| DIODE | indoor/outdoor depth | RGB-D | AbsRel, RMSE, PLOA |
| KINS / COCOA | amodal segmentation | amodal masks | modal IoU, amodal IoU, invisible IoU |
| IIW | intrinsic decomposition | reflectance comparisons | WHDR |
| Real curated set | qualitative editing | no full GT | visual comparison, user preference |

---

## Table 3: Main quantitative results template

| Method | group mIoU ↑ | PLOA ↑ | BW-PLOA ↑ | Recon PSNR ↑ | Recon SSIM ↑ | LPIPS ↓ | Amodal IoU ↑ | Runtime ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical baseline | | | | | | | | |
| Panoptic only | | | | | | | | |
| Panoptic + depth | | | | | | | | |
| Panoptic + depth + graph | | | | | | | | |
| Open-vocab + depth + graph | | | | | | | | |
| Full LayerForge-X | | | | | | | | |

---

## Table 4: Component ablation template

A leave-one-out table. For each row, the expected-damage column says what I think should break, and the observed-result column is what the numbers actually say. If the observed column doesn't match the expected column, that's interesting and deserves a paragraph in the discussion.

| Removed component | Expected damage | Metric most affected | Observed result |
|---|---|---|---|
| remove semantic segmenter | no meaningful object layers | group mIoU | |
| remove depth | wrong near/far order | PLOA/BW-PLOA | |
| remove boundary graph | large stuff/object order errors | BW-PLOA/Occlusion F1 | |
| remove soft alpha | jagged boundaries | alpha MAE/recomposition edge error | |
| remove amodal masks | no hidden object support | amodal IoU/editing score | |
| remove inpainting | holes after edits | masked LPIPS/hole ratio | |
| remove intrinsic split | no recoloring/shading control | WHDR/edit demo | |
| remove recursive peeling | weaker hidden-region recovery in iterative scenes | edit demo / masked LPIPS | |

---

## Table 5: Failure analysis template

One row per failure example. Talking about failures explicitly is one of the easiest ways to make the report read as mature rather than salesy:

| Image | Failure type | Cause | Visible symptom | Fix/future work |
|---|---|---|---|---|
| image_01 | depth ambiguity | mirror/glass | wrong order | uncertainty + user correction |
| image_02 | mask merge | same-colored objects | two objects in one layer | prompt refinement |
| image_03 | alpha failure | hair/fur | jagged edge | matting backend |
| image_04 | inpaint failure | large hidden background | blurry fill | stronger diffusion inpaint |
| image_05 | amodal failure | extreme occlusion | wrong hidden shape | SAMEO/amodal backend |

---


## Appendix C: command log

- [../../report_artifacts/command_log.md](../../report_artifacts/command_log.md)

## Appendix D: extra literature notes

- [../LITERATURE_REVIEW.md](../LITERATURE_REVIEW.md)
- [../REFERENCES.md](../REFERENCES.md)
