# 2. Related Work

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
