# Literature Review and Theoretical Framework

This document provides a condensed overview of the research areas and foundational technologies that inform the LayerForge-X architecture. A comprehensive, long-form review is available in `docs/final_report_pack/sources/01_LITERATURE_REVIEW_ADVANCED.md`.

## Layered Scene Representations

The concept of the **Layered Depth Image (LDI)**, as proposed by Shade et al., serves as the primary theoretical precursor to the current work. LDIs store multiple depth and color samples along each camera ray, facilitating disocclusion-aware rendering and view synthesis. LayerForge-X extends this principle by formalizing these layers into a semantically meaningful, editable scene graph (DALG) that supports both instance-level and region-level manipulation.

## Semantic and Instance Segmentation

High-fidelity scene decomposition necessitates robust segmentation of both discrete objects ("things") and amorphous background regions ("stuff").

- **Panoptic Segmentation:** Serves as the baseline for closed-set scene understanding, with **Mask2Former** providing a unified architecture for semantic and instance-level proposals.
- **Open-Vocabulary Grounded Segmentation:** For dynamic, user-defined layer extraction, the integration of **GroundingDINO** and **SAM2** enables natural-language conditioning, allowing for the decomposition of entities beyond fixed datasets (e.g., COCO, ADE20K).

## Monocular Geometry and Depth Estimation

Accurate depth ordering is critical for establishing a consistent occlusion graph. LayerForge-X leverages modern monocular depth estimation (MDE) models, including **Depth Anything V2**, **Depth Pro**, **Marigold**, and **MoGe**, to drive three core functions:

1. **Pairwise Ordering:** Determining the relative front-to-back relationship between adjacent layers.
2. **Spatial Stratification:** Partitioning large, depth-spanning background regions into manageable planes.
3. **Parallax Synthesis:** Providing the geometric priors required for interactive 3D previews.

## Alpha Matting and Boundary Refinement

The transition from binary segmentation masks to production-quality layers requires precise alpha matting to handle semi-transparent structures, such as hair, fur, glass, and motion blur. Drawing from the "Matting Anything" line of research, LayerForge-X treats soft alpha as a fundamental layer attribute, employing boundary-aware refinement to ensure seamless recomposition.

## Amodal Reasoning and Content Completion

To support true editability, the system must reason about occluded content.

- **Amodal Segmentation:** Estimates the full spatial extent of objects, even when partially hidden.
- **Image Inpainting:** Fills disoccluded regions using advanced architectures like **LaMa**, ensuring that background layers remain coherent after foreground removal. Together, these components facilitate the transition from a 2D image to a multi-layered, interactive scene.

## Intrinsic Decomposition

The decoupling of **albedo** (surface reflectance) and **shading** (illumination effects) is essential for advanced editing tasks, such as relighting or recoloring. By integrating intrinsic decomposition, LayerForge-X provides a more granular representation of scene appearance, evaluated against standard benchmarks such as **Intrinsic Images in the Wild (IIW)**.
