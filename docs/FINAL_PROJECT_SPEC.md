# Final Project Specification

## Title

**LayerForge-X: Interpretable Depth-Aware Amodal Layer Graphs from a Single RGB Image**

## Core idea

A normal RGB image collapses object identity, occlusion, depth, shading, transparency, and background into one raster. LayerForge-X converts that raster into an editable scene-layer representation.

The final representation is a **Depth-Aware Amodal Layer Graph (DALG)**:

```text
G = (V, E)
```

Each node is a layer:

```text
v_i = {
  rgba_i,
  semantic_label_i,
  semantic_group_i,
  visible_mask_i,
  soft_alpha_i,
  amodal_mask_i,
  hidden_mask_i,
  depth_statistics_i,
  depth_bin_i,
  albedo_rgba_i,
  shading_rgba_i,
  bbox_i,
  uncertainty_i
}
```

Each edge stores occlusion / depth evidence:

```text
e_ij = {
  near_id,
  far_id,
  confidence,
  shared_boundary_length,
  local_depth_gap,
  reason
}
```

## Pipeline

```text
Input RGB image
  ↓
candidate layer proposals
  - classical fallback
  - Mask2Former-style panoptic segmentation
  - GroundingDINO + SAM2 open-vocabulary segmentation
  - optional Qwen-Image-Layered imported layers
  ↓
soft alpha refinement
  ↓
monocular depth / geometry estimation
  ↓
boundary-weighted occlusion graph
  ↓
near → far topological ordering
  ↓
amodal support estimation
  ↓
background / hidden-region completion
  ↓
per-layer albedo/shading split
  ↓
RGBA layers + graph JSON + debug visualizations + metrics
```

## Why the graph matters

An unordered folder of transparent PNGs is weak. It does not explain which layer is closer, what occludes what, where hidden support is expected, or whether the output can be evaluated component-wise.

The graph makes the output inspectable:

```json
{
  "rank": 0,
  "name": "000_person_person",
  "label": "person",
  "group": "person",
  "depth_median": 0.21,
  "occludes": [3, 5],
  "occluded_by": [],
  "bbox": [120, 45, 312, 460]
}
```

## Main algorithmic novelty

### Boundary-weighted occlusion ordering

For adjacent layers `i` and `j`, global median depth can fail on large/slanted regions. LayerForge-X instead compares boundary-local depth:

```text
B_ij = local shared boundary support

z_i^B = median depth of layer i near B_ij
z_j^B = median depth of layer j near B_ij
```

If smaller depth means closer:

```text
i is in front of j when z_i^B < z_j^B
```

Confidence:

```text
confidence = |z_i^B - z_j^B| × log(1 + shared_boundary_length)
```

The graph is then topologically sorted. If cycles appear, the weakest edge is removed.

## Final outputs

```text
layers_ordered_rgba/
layers_grouped_rgba/
layers_albedo_rgba/
layers_shading_rgba/
layers_amodal_masks/
debug/depth_gray.png
debug/segmentation_overlay.png
debug/layer_graph.json
debug/background_completion.png
debug/parallax_preview.gif
manifest.json
metrics.json
```

## Project contribution in one paragraph

LayerForge-X decomposes a single RGB image into semantically grouped, depth-ordered, amodal RGBA layers enriched with occlusion relations and intrinsic appearance factors. Unlike direct RGB-to-layer generators, it produces an inspectable graph representation and evaluates layer quality across segmentation, depth ordering, recomposition, amodal completion, intrinsic decomposition, and editing utility.
