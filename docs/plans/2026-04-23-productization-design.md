# 2026-04-23 Productization Design Note

## Goal

Turn the existing research engine into a clearer product core without widening
the model stack. The immediate move is to standardize the scene artifact and
document the intended API and launch shape around what already exists.

## Decisions

1. DALG becomes the canonical object.
   - native runs, recursive peeling, and Qwen enrichment all normalize into one
     `dalg_manifest.json`
   - the engine-specific `manifest.json` remains, but it is no longer the only
     shape consumers need to understand
2. API-first direction is documented now, even before a control-plane service
   exists.
   - `docs/api/openapi.yaml` defines the public job and DALG retrieval contract
3. Export surface stays narrow.
   - implement JSON design-manifest export first
   - leave PSD/editor adapters as future work rather than mock them

## Why this slice

The highest-value differentiation for LayerForge lies in structure, evaluation,
editability, and export rather than in attempting to out-scale a generative
baseline. A canonical graph export and a contract-first API are the smallest
concrete changes that advance the repository in that direction while remaining
aligned with the implemented scope.

## Deliverables

- `src/layerforge/dalg.py`
- `schemas/dalg.schema.json`
- `layerforge export-design`
- `docs/api/openapi.yaml`
- product architecture and launch doc
- regression tests proving DALG export exists and is stable
