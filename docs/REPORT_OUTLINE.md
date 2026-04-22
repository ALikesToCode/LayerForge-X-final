# Report Outline

The skeleton for the write-up. Numbers are suggestions, not a hard contract.

1. **Abstract** — one paragraph summarising the DALG representation and the headline results.
2. **Introduction** — why flat RGB images are hard to edit, what an editable layer representation should look like, and what the project delivers.
3. **Related Work** — LDIs, panoptic segmentation, SAM2 and GroundedSAM, monocular depth, matting, inpainting, intrinsic images.
4. **Method** — the pipeline in order: segment proposal, depth, graph ordering, alpha, amodal reasoning, background completion, intrinsics.
5. **Implementation** — configs, model modes, exported artifacts.
6. **Experiments** — the synthetic benchmark, the public subtask benchmarks, and the real-image qualitative set.
7. **Ablations** — the A-to-E (or A-to-H) variant sweep.
8. **Results** — the filled-in tables and figures.
9. **Failure Cases** — glass, mirrors, shadows, hair, large occlusions, the anime-depth-ambiguity problem. Don't hide them.
10. **Conclusion** — the DALG as a compact editable single-image scene representation, plus what's missing.
