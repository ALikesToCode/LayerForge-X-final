# 3. Related Work

## Project framing

LayerForge-X addresses **single-image layered scene decomposition**: given one RGB image, infer a set of recomposable RGBA layers that are semantically meaningful, ordered by depth and occlusion, and, where possible, augmented with intrinsic appearance factors such as albedo and shading. The difficulty is that a raster image collapses object identity, transparency, shadows, reflections, illumination, camera projection, and occlusion into a single 2D array. A useful layered representation therefore draws from several adjacent research areas at once: scene understanding, monocular geometry, matting, completion, and appearance decomposition.

LayerForge-X positions its output as a **Depth-Aware Amodal Layer Graph (DALG)** rather than a plain segmentation export. Each layer is represented as a graph node carrying a visible mask, a soft alpha, a semantic label, depth statistics, an estimated amodal extent, completed hidden or background content, and optional intrinsic appearance factors. Graph edges encode near/far and occlusion relations. The intent is to support editing, parallax, object removal, relighting-style operations, and downstream analysis rather than only visible cutout export.

## 3.1 Layered depth and image-based rendering

The most direct historical precursor is the **Layered Depth Image (LDI)** of Shade et al., which stores multiple samples along a viewing ray instead of only the first visible surface. That formulation is important because the present project also requires more than a flat visible mask: it requires depth ordering and some notion of hidden-region reasoning.

LDIs are effective for view synthesis because disocclusions can be handled more gracefully than with a single depth map. They were not designed as editable semantic layers. Classical LDIs do not identify which layer is “person,” “chair,” “sky,” or “road,” and they do not separate albedo from shading or associate foreground effects with object cores.

**Relevance to LayerForge-X:** DALG borrows the idea of storing layered samples along viewing rays, but makes the layers semantic, alpha-composited, and editable.

## 3.2 3D photography and depth inpainting

Shih et al.’s **3D Photography using Context-Aware Layered Depth Inpainting** converts RGB-D input into a multi-layer representation for parallax rendering. The work is directly relevant because it combines layered depth, inpainting, and interactive view synthesis in a single pipeline. That integration is close in spirit to LayerForge-X, although the target task is different.

The main limitation for the present project is that 3D Photography assumes RGB-D input or externally supplied depth and primarily targets novel-view synthesis rather than semantic object editing. It does not export object-level semantic RGBA layers such as people, vehicles, furniture, or background regions.

**Relevance to LayerForge-X:** LayerForge-X treats hidden-region completion as one stage inside a semantic layer graph. The output can support parallax, but object editing remains the primary goal.

## 3.3 Scene decomposition, object layers, and occluded content

Several older works move closer to the full scene-decomposition target. Dhamo et al.’s **Object-Driven Multi-Layer Scene Decomposition From a Single Image** builds an LDI from one RGB image and uses object reasoning to infer occluded intermediate layers. Zheng et al.’s **Layer-by-Layer Completed Scene Decomposition** studies decomposition into individual objects, occlusion relationships, amodal masks, and content completion.

These works are important because they make clear that correct layering requires more than visible segmentation. Occluded regions are part of the representation. If a person stands in front of a car, an editing-oriented representation should estimate the hidden continuation of the car well enough to support removal or parallax operations.

**Relevance to LayerForge-X:** LayerForge-X follows this line while using a modern toolchain: foundation segmentation models, current monocular geometry, promptable/open-vocabulary controls, and explicit evaluation of recomposition and editability.

## 3.4 Video layer decomposition and Omnimatte-style effects

Layered decomposition has also been studied extensively in video. **Omnimatte** decomposes a video into object-associated RGBA layers that can include related visual effects such as shadows, smoke, and reflections. This is conceptually important because a clean object cutout is often insufficient for editing; moving a person without the associated shadow immediately looks implausible.

Video methods benefit from motion and temporal consistency. Single-image decomposition does not have that advantage, which makes the task substantially more ambiguous.

**Relevance to LayerForge-X:** LayerForge-X can export associated-effect layers such as object cores, shadows, reflections, or local residual layers. Even a lightweight effect-layer path makes the output closer to true layer decomposition than plain segmentation.

## 3.5 Modern generative layer decomposition

Recent work makes the project topic especially timely. **LayerDecomp** targets image layer decomposition with visual effects, producing a clean background and a transparent foreground while preserving effects such as shadows and reflections. **DiffDecompose** studies layer-wise decomposition of alpha-composited images, particularly transparent and semi-transparent cases. **Qwen-Image-Layered** proposes an end-to-end diffusion model that decomposes one RGB image into multiple semantically disentangled RGBA layers. **Referring Layer Decomposition** frames the task as prompt-conditioned RGBA layer extraction.

These works define the current frontier. LayerForge-X does not claim to exceed them on raw generative image quality. Its claim is different: a transparent, modular, geometry-aware layer graph that can be benchmarked component-by-component rather than assessed only by visual inspection.

**Relevance to LayerForge-X:** These systems provide the most direct baselines and motivation for the DALG representation, Qwen enrichment, promptable extraction, and transparent-layer prototype.

## 3.6 Panoptic and open-vocabulary segmentation

Layered decomposition needs both object and background-region proposals. Panoptic segmentation is a natural fit because it unifies instance-level “things” with amorphous “stuff” classes such as sky, road, wall, grass, or water. The original panoptic segmentation work also introduced the unified **Panoptic Quality (PQ)** metric.

**Mask2Former** is a strong closed-set baseline because it handles semantic, instance, and panoptic segmentation with one universal architecture. Its limitation is fixed vocabulary. Open-vocabulary pipelines address that gap. **GroundingDINO** detects objects specified by free-form text, and **SAM/SAM2** provide promptable segmentation masks. Combined, they support prompts such as “left chair,” “red car,” “window,” or “foreground person,” which are highly relevant for editable layer extraction.

**Relevance to LayerForge-X:** Closed-set panoptic segmentation provides a benchmarkable baseline, while open-vocabulary grounded segmentation provides prompt-conditioned extraction for interactive use.

## 3.7 Monocular depth and geometry

Depth ordering is central, and naive global-average sorting often fails. Large regions such as walls, floors, tables, and roads span wide depth ranges, so an average or median depth can mis-rank them relative to nearby objects.

Modern monocular geometry models provide the priors that make depth-based ordering viable. **Depth Anything V2** improves robustness and detail over earlier monocular depth models. **Depth Pro** produces sharp metric depth from a single image without camera intrinsics. **Marigold** repurposes diffusion priors for affine-invariant depth estimation. **MoGe** extends monocular prediction toward a fuller geometry package including point maps and normals.

**Relevance to LayerForge-X:** LayerForge-X uses boundary-local depth evidence rather than global layer statistics when inferring pairwise ordering. That is more robust when objects overlap and when background regions span both near and far pixels.

## 3.8 Amodal segmentation and occlusion reasoning

Visible (modal) masks describe only what can be seen. Amodal segmentation estimates the full object extent, including invisible occluded parts. **KINS** is a key amodal instance segmentation benchmark, and more recent work such as **SAMEO** adapts Segment Anything-style models to occluded objects.

Amodal reasoning matters because editing demands it. If a foreground object is removed, the background must be completed. If a partially occluded object is moved, its hidden parts may need to be estimated. On real images this is inherently ambiguous, so such completions should be treated as plausible rather than exact.

**Relevance to LayerForge-X:** LayerForge-X reports visible masks separately from amodal masks, which keeps the representation explicit about observed versus estimated content.

## 3.9 Alpha matting and edge quality

Hard segmentation masks produce jagged cutouts. Practical layers require soft alpha around hair, fur, glass, motion blur, antialiased edges, smoke, and other semi-transparent structures. Matting methods estimate a fractional alpha matte directly.

**Matting Anything** is especially relevant because it combines SAM features with a lightweight mask-to-matte module and supports visual or linguistic prompts, which aligns well with prompt-conditioned layer extraction.

**Relevance to LayerForge-X:** Even when a simple boundary-feathering fallback is used, the distinction between hard alpha and soft alpha remains important because it strongly affects recomposition quality and editing plausibility.

## 3.10 Inpainting and hidden-region completion

Editing layered content requires plausible content behind removed or moved objects. **LaMa** is the most practical baseline in this category because it was designed for large masks and high-resolution generalization.

Inpainting should not be framed as exact recovery on real images. For photographs, the hidden content is unknown; only plausible completion is possible. For synthetic composites with known clean backgrounds, the problem becomes quantitatively measurable.

**Relevance to LayerForge-X:** Synthetic data supports quantitative completion metrics, while real images support qualitative object-removal and parallax demonstrations.

## 3.11 Intrinsic images: albedo and shading

Intrinsic image decomposition separates an image into reflectance/albedo and illumination/shading. The problem is severely under-constrained from one image. **Intrinsic Images in the Wild (IIW)** introduced a large in-the-wild benchmark based on human reflectance judgments. More recent diffusion-based approaches provide stronger baselines, but the task remains difficult.

Within LayerForge-X, intrinsic decomposition is best treated as a stretch module rather than the core contribution. A Retinex-style fallback is sufficient for an editable approximation, provided the limitations are stated clearly.

**Relevance to LayerForge-X:** Per-layer albedo and shading exports can support recoloring and simple appearance edits, even when the factorization is only approximate.

## 3.12 Gap summary

| Prior area | What it solves | What it misses for this project | How LayerForge-X uses it |
|---|---|---|---|
| LDI / image-based rendering | Multi-depth samples for novel views | No semantic/editable object layers | Uses layered depth as a representational backbone |
| 3D photo inpainting | Parallax and hidden-region completion | Usually not semantic object/stuff editing | Adds semantic graph nodes and RGBA export |
| Panoptic segmentation | Things and stuff parsing | No depth, alpha, or hidden content | Provides layer proposals |
| Open-vocabulary segmentation | User-specified object masks | No ordering or completion by itself | Enables promptable layer extraction |
| Monocular depth | Per-pixel relative/metric depth | No object graph or masks | Supplies geometry for ordering |
| Amodal segmentation | Full object extent under occlusion | Does not complete appearance alone | Supplies hidden masks |
| Matting | Soft alpha boundaries | Usually foreground/background only | Refines per-layer alpha |
| Inpainting | Plausible missing content | No semantic/depth ordering | Completes background and hidden regions |
| Intrinsic images | Albedo/shading factors | Highly ambiguous; not layer-aware | Provides optional appearance factors |
| Generative layer decomposition | End-to-end RGBA layers | Often black-box and hard to benchmark component-wise | Serves as frontier comparison and proposal source |

## Report-ready synthesis

Single-image layered scene decomposition sits at the intersection of image-based rendering, scene parsing, monocular geometry, amodal perception, matting, inpainting, and intrinsic image decomposition. Classical Layered Depth Images show why a single visible surface per pixel is insufficient for view synthesis, since disoccluded content requires multiple depth and color samples along camera rays. Modern 3D-photo methods extend this with learned color-and-depth inpainting, but primarily target parallax rather than semantic object editing. Panoptic segmentation provides a natural source of object and stuff proposals, while open-vocabulary detectors and promptable segmenters allow user-specified layer extraction beyond fixed label sets. Recent monocular depth and geometry models improve the reliability of depth ordering, but depth alone does not produce editable layers. Amodal segmentation and inpainting address invisible portions of occluded objects and backgrounds, while matting improves layer boundaries. Recent generative layer-decomposition systems demonstrate the importance of RGBA layers for editing, but their end-to-end nature makes component-wise analysis difficult. LayerForge-X therefore proposes an inspectable Depth-Aware Amodal Layer Graph that combines semantic masks, depth ordering, soft alpha, amodal extent, completion, and optional intrinsic decomposition into a recomposable representation.
