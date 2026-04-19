# Literature Review: LayerForge-X

## Project framing

The project studies **single-image layered scene decomposition**: given one RGB bitmap, infer a set of re-composable RGBA layers that are semantically meaningful, ordered by depth/occlusion, and optionally decomposed into albedo and shading. The central difficulty is that a raster image collapses many scene factors into one 2D array: object identity, transparency, shadows, reflections, illumination, camera projection, and occlusion. A strong layered representation must therefore combine scene understanding, geometry, matting, completion, and appearance decomposition.

LayerForge-X is positioned as a **Depth-Aware Amodal Layer Graph (DALG)** rather than a plain segmentation exporter. Each layer is a graph node with visible mask, soft alpha, semantic label, depth statistics, estimated amodal extent, completed hidden/background content, and optional intrinsic appearance. Graph edges encode near/far or occludes/occluded-by relations. This makes the output useful for editing, parallax, object removal, relighting-style operations, and downstream analysis.

---

## 1. Layered depth and image-based rendering

The oldest directly relevant representation is the **Layered Depth Image (LDI)**. Shade et al. introduced LDIs as an image-based rendering representation where a single camera view can store **multiple samples along one line of sight**, not just the first visible surface. This is the correct historical starting point because the project also needs more than a flat visible mask: it needs depth ordering and hidden-region reasoning.

LDIs are strong for novel-view rendering because disocclusions can be handled better than with a single depth map. However, classical LDIs are not designed as editable semantic layers. They usually do not tell the user which layer is “person,” “chair,” “sky,” or “road,” and they do not explicitly separate albedo, shading, foreground effects, or object-level amodal masks.

**Use in our project:** DALG borrows the idea of storing layered samples along viewing rays, but it makes the layers semantic, alpha-composited, and editable.

---

## 2. 3D photography and depth inpainting

Shih et al. proposed **3D Photography using Context-Aware Layered Depth Inpainting**, converting an RGB-D image into a multi-layer representation for parallax rendering. Their system uses an LDI with explicit pixel connectivity and performs color-and-depth inpainting in occluded regions. This paper is highly relevant because it connects layered depth, inpainting, and interactive view synthesis.

The limitation for this project is that 3D Photography assumes an RGB-D input or externally supplied depth, and its primary target is novel-view synthesis rather than semantic object editing. It does not directly output semantically grouped RGBA layers such as “people,” “vehicles,” “furniture,” and “background stuff.”

**Use in our project:** LayerForge-X treats context-aware hidden-region completion as one stage inside a semantic layer graph. The output is not only a parallax asset but also an object-editing representation.

---

## 3. Scene decomposition, object layers, and completed occluded content

Several works move closer to the project goal by decomposing scenes into object-level layers. Dhamo et al.’s **Object-Driven Multi-Layer Scene Decomposition From a Single Image** aims to build an LDI from a single RGB image and uses object information to infer occluded intermediate layers. Zheng et al.’s **Layer-by-Layer Completed Scene Decomposition** studies decomposition into individual objects, occlusion relationships, amodal masks, and content completion.

These works are important because they show that correct layering requires more than visible segmentation. Occluded regions matter. If a person stands in front of a car, a practical layer representation should estimate not only the visible car pixels but also the likely hidden car continuation, at least enough to support object removal or parallax.

**Use in our project:** LayerForge-X follows this line but modernizes the toolchain with foundation segmentation models, current monocular geometry, promptable/open-vocabulary controls, and explicit evaluation of recomposition/editability.

---

## 4. Video layer decomposition and omnimattes

Layered decomposition has also been studied heavily in video. **Omnimatte** decomposes a video into object-associated RGBA layers that can include not only the object but also related visual effects such as shadows, smoke, or reflections. This is conceptually important because a clean object cutout is often not enough for editing: moving a person without their shadow looks wrong.

Generative Omnimatte and related later works extend this idea with stronger generative priors. However, video methods can exploit motion and temporal consistency. A single-image project does not have optical flow or multiple frames, so the task is more ambiguous.

**Use in our project:** LayerForge-X can optionally export associated effect layers: object core, shadow/reflection/effect region, and clean background. Even a simple shadow layer heuristic makes the project look more like serious layer decomposition than ordinary segmentation.

---

## 5. Modern generative layer decomposition

Recent generative work makes the project topic especially timely. **LayerDecomp** targets image layer decomposition with visual effects, producing a clean background and transparent foreground while preserving effects such as shadows and reflections. **DiffDecompose** studies layer-wise decomposition of alpha-composited images, especially transparent and semi-transparent layers. **Qwen-Image-Layered** proposes an end-to-end diffusion model that decomposes a single RGB image into multiple semantically disentangled RGBA layers. **Referring Layer Decomposition** frames the task as prompt-conditioned RGBA layer extraction.

These papers are very close to the project statement and should be cited to show awareness of the frontier. The important distinction is that LayerForge-X should not claim to beat these large generative systems without evidence. Instead, it should claim a different contribution: a transparent, modular, geometry-aware layer graph that can be benchmarked component-by-component.

**Use in our project:** These works define the current research frontier. LayerForge-X positions itself as a practical, inspectable alternative that fuses segmentation, geometry, amodal reasoning, alpha refinement, inpainting, and intrinsic decomposition.

---

## 6. Panoptic and open-vocabulary segmentation

Layered decomposition needs object and stuff regions. Panoptic segmentation is a natural fit because it unifies instance-level “things” such as people, animals, and vehicles with amorphous “stuff” such as sky, road, wall, grass, and water. The panoptic segmentation paper also introduced **Panoptic Quality (PQ)** as a unified metric.

**Mask2Former** is a strong closed-set baseline because it is a universal segmentation architecture for semantic, instance, and panoptic segmentation. However, closed-set segmenters are limited by their training label vocabulary.

Open-vocabulary models address this limitation. **GroundingDINO** detects arbitrary text-specified objects or referring expressions, while **SAM/SAM2** produces promptable segmentation masks. This combination allows prompts such as “left chair,” “red car,” “window,” or “foreground person,” which is exactly the kind of control a layer editor needs.

**Use in our project:** The report should compare closed-set panoptic segmentation and open-vocabulary grounded segmentation. The latter is more exciting, but the former is easier to benchmark with standard PQ/mIoU metrics.

---

## 7. Monocular depth and geometry

Depth ordering is central. A stack of layers must be sorted near to far, and average object depth is often insufficient because large regions such as walls, floors, tables, and roads span a wide depth range.

Modern monocular geometry models provide strong priors. **Depth Anything V2** improves robustness and detail over earlier monocular depth models using synthetic labeled data plus large-scale pseudo-labeled real data. **Depth Pro** estimates sharp metric depth from a single image without camera intrinsics. **Marigold** repurposes diffusion priors for affine-invariant depth estimation. **MoGe** predicts richer monocular geometry such as point maps, depth, normals, and camera field of view.

**Use in our project:** Instead of sorting layers only by global median depth, LayerForge-X should infer pairwise ordering from boundary-local depth evidence. This is more robust when objects overlap or when a background region covers both near and far pixels.

---

## 8. Amodal segmentation and occlusion reasoning

Visible masks are modal: they describe only what can be seen. Amodal segmentation estimates the full object extent, including invisible occluded regions. KINS is a key amodal instance segmentation benchmark, and newer foundation-model-based work such as **SAMEO** adapts Segment Anything-style mask decoders for occluded objects.

Amodal reasoning is crucial for editing. If a user removes a foreground object, the background must be completed. If a user moves a partially occluded object, its hidden parts may need to be hallucinated. This is inherently ambiguous, so results should be treated as plausible completions rather than ground truth on real images.

**Use in our project:** LayerForge-X should report modal visible masks separately from amodal masks. This avoids overclaiming and makes the representation honest.

---

## 9. Alpha matting and edge quality

Hard segmentation masks create jagged cutouts. Real layers need soft alpha around hair, fur, glass, motion blur, antialiased vector edges, smoke, and transparent objects. Matting methods solve this by estimating a fractional alpha matte.

**Matting Anything** is relevant because it combines Segment Anything features with a lightweight mask-to-matte module and supports visual or linguistic prompts. This fits the project’s open-vocabulary layer extraction direction.

**Use in our project:** Even if the implementation uses a simple boundary feathering fallback, the report should explicitly evaluate hard alpha vs soft alpha. The visual difference is usually obvious.

---

## 10. Inpainting and hidden-region completion

Layer editing requires plausible content behind removed or moved objects. **LaMa** is a strong inpainting reference because it was designed for large masks and high-resolution generalization using Fourier convolutions and large-mask training.

Inpainting should not be treated as “ground truth recovery” on real images. For real images, it is plausible completion. For synthetic composites, where the hidden background is known, it can be evaluated quantitatively.

**Use in our project:** Use synthetic data to score background completion with PSNR/SSIM/LPIPS inside removed-object regions. Use real images for qualitative object removal and parallax demos.

---

## 11. Intrinsic images: albedo and shading

Intrinsic image decomposition separates an image into reflectance/albedo and illumination/shading. The problem is highly ambiguous from one image. **Intrinsic Images in the Wild (IIW)** introduced a large in-the-wild benchmark using human reflectance judgments and WHDR. Recent diffusion-based intrinsic methods, including Marigold-IID, provide stronger modern baselines.

For this project, intrinsic decomposition should be presented as a stretch module, not as the core claim. A Retinex-style fallback is acceptable, but the report should be honest that physical correctness is limited.

**Use in our project:** Export per-layer albedo and shading as useful editing approximations. Evaluate with IIW/WHDR if possible; otherwise use synthetic scenes with known albedo/shading.

---

## Gap summary

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

Single-image layered scene decomposition sits at the intersection of image-based rendering, scene parsing, monocular geometry, amodal perception, matting, inpainting, and intrinsic image decomposition. Classical Layered Depth Images show why a single visible surface per pixel is insufficient for view synthesis, since disoccluded content requires multiple depth/color samples along camera rays. Modern 3D-photo methods extend this idea with learned color-and-depth inpainting, but they primarily target parallax rather than semantic object editing. Panoptic segmentation provides a natural source of object and stuff proposals, while open-vocabulary detectors and promptable segmenters allow user-specified layer extraction beyond fixed label sets. Recent monocular depth and geometry models improve the reliability of depth ordering, but depth alone does not produce editable layers. Amodal segmentation and inpainting address the invisible portions of occluded objects and backgrounds, while matting improves layer boundaries. Recent generative layer-decomposition systems demonstrate the importance of RGBA layers for editing, but their end-to-end nature makes component-wise analysis difficult. Our work therefore proposes an inspectable Depth-Aware Amodal Layer Graph that combines semantic masks, depth ordering, soft alpha, amodal extent, completion, and optional intrinsic decomposition into a re-composable representation.
