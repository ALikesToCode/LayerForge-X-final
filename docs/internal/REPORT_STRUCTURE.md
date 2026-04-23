# Report Structure

This document records the intended structure of the final report.

1. **Abstract** — one paragraph summarizing the DALG representation and the headline results.
2. **Introduction** — why flat RGB images are difficult to edit, what an editable layer representation requires, and what LayerForge-X provides.
3. **Related Work** — layered depth images, panoptic segmentation, SAM2 and GroundedSAM, monocular depth, matting, inpainting, intrinsic images, and recent generative layer decomposition.
4. **Method** — the pipeline in order: segment proposal, depth, graph ordering, alpha refinement, amodal reasoning, background completion, intrinsics, and export.
5. **Implementation** — configurations, model modes, and exported artifacts.
6. **Experiments** — the synthetic benchmark, public subtask benchmarks, and the real-image qualitative set.
7. **Ablations** — the measured component sweep and hybrid comparisons.
8. **Results** — tables, figures, and interpretation.
9. **Failure Cases and Limitations** — glass, mirrors, shadows, hair, large occlusions, and other remaining weaknesses.
10. **Conclusion** — the DALG as a compact editable single-image scene representation and the main remaining gaps.
