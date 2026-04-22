# Next Review Checklist (2026-04-22)

This file records the post-fix state after the final 2026-04-22 measurement pass.

## Fixed now

- ordered-layer export is collision-free and regression-tested;
- recursive peeling is implemented as `layerforge peel`;
- associated-effect extraction is implemented and exported to `layers_effects_rgba/`;
- the synthetic generator supports a richer `layerbench_pp` format with visible masks, amodal masks, alpha mattes, effects, intrinsics, depth, and occlusion metadata;
- submission-safe `report_artifacts/` snapshots are generated from measured runs already present locally;
- the docs now distinguish coarse-group public benchmarks from planned full panoptic evaluation.
- same-image five-image comparison has been run and summarized in `runs/qwen_five_image_review/`, covering native LayerForge, raw Qwen, `Qwen + graph preserve`, and `Qwen + graph reorder`;
- at least one measured recursive-peeling demo run is saved under `runs/effects_demo_peel/`;
- at least one measured associated-effect demo figure is saved as `docs/figures/effects_layer_demo.png`;
- the final report markdown and DOCX have been rebuilt around the updated figures and tables.

## Remaining gap, if someone asks for more

1. If the report wants official PQ, add a real panoptic evaluator rather than the current coarse-group IoU path.
2. If the report wants stronger effect-layer claims, improve the associated-effect extractor beyond the current weak heuristic.

## What the current evidence supports

1. The repo now contains a measured same-image five-image comparison across native LayerForge, raw Qwen, and both fair hybrid modes.
2. The repo contains at least one measured recursive-peeling run and one measured associated-effect demo.
3. The submission-safe report artifact bundle contains JSON snapshots for the new Qwen and effects evidence.

## Review framing

The honest repo claim after the current fixes is:

> LayerForge-X is now a structured, benchmarked layer-representation system with a fair Qwen comparison path, recursive peeling, associated-effect layers, and auditable artifact snapshots. The main remaining weakness is not missing implementation but that the effect-layer extractor is still heuristic and should not be oversold.
