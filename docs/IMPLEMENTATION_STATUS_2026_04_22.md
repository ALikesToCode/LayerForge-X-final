# Implementation Status (2026-04-22)

This document records the repository state after the final 2026-04-22 measurement pass.

## Implemented and measured components

- ordered-layer export is collision-free and regression-tested;
- recursive peeling is implemented as `layerforge peel`;
- associated-effect extraction is implemented and exported to `layers_effects_rgba/`;
- the synthetic generator supports the richer `layerbench_pp` format with visible masks, amodal masks, alpha mattes, effects, intrinsics, depth, and occlusion metadata;
- submission-safe `report_artifacts/` snapshots are generated from measured local runs;
- the documentation distinguishes coarse-group public benchmarks from planned full panoptic evaluation;
- a same-image five-image comparison has been run and summarized in `runs/qwen_five_image_review/`, covering native LayerForge, raw Qwen, `Qwen + graph preserve`, and `Qwen + graph reorder`;
- a five-image frontier candidate-bank review has been run and summarized in `runs/frontier_review/`, covering native LayerForge, recursive peeling, raw Qwen, and both hybrid modes;
- a measured recursive-peeling demo run is saved under `runs/effects_demo_peel/`;
- a measured associated-effect figure is saved as `docs/figures/effects_layer_demo.png`;
- the final report markdown and DOCX have been rebuilt around the updated figures and tables.

## Current limits

1. Official COCO-style panoptic quality would require a full PQ evaluator rather than the current coarse-group IoU path.
2. Stronger effect-layer claims would require an improved associated-effect extractor beyond the current heuristic implementation.

## Evidence supported by the current archive

1. The repository contains a measured same-image five-image comparison across native LayerForge, raw Qwen, and both hybrid modes.
2. The repository contains at least one measured recursive-peeling run and one measured associated-effect demo.
3. The submission-safe artifact bundle contains JSON snapshots for the Qwen, frontier, prompt, transparent, and effects evidence.
4. The frontier candidate-bank evidence is measured on the same five-image set and records per-image best selections under `best_decomposition.json`.

## Repository position

LayerForge-X is best characterized as a structured, benchmarked layer-representation system with a fair Qwen comparison path, recursive peeling, associated-effect layers, and auditable artifact snapshots. The principal remaining limitation is that the associated-effect extractor remains heuristic and should be framed accordingly.
