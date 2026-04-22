# LayerForge Product, Architecture, and Launch Plan

This repository now has a clear product anchor: LayerForge is not "just a decomposition
pipeline" and not "just a hosted model call". The strongest shape is a
**self-evaluating layer-representation platform** built around a canonical
Depth-Aware Amodal Layer Graph (DALG).

## Product thesis

LayerForge should treat the graph manifest as the system of record. PNG stacks,
PSD-ready folders, debug previews, and benchmark tables are projections of that
graph, not the primary object.

The practical promise is:

- convert one still image into an editable scene asset;
- preserve near/far order, occlusion evidence, and optional amodal support;
- route between native, recursive, and Qwen-enriched decompositions;
- score those candidates with fidelity and editability metrics;
- export a stable design manifest for downstream tools.

## Core surfaces

The current repository already contains the engine pieces needed for the product
shape below:

| Surface | Repo anchor | Product role |
| --- | --- | --- |
| Inference engine | `layerforge run`, `peel`, `enrich-qwen` | Generate candidate layer graphs |
| Evaluator | `self_eval.py`, `editability.py`, `scripts/run_frontier_comparison.py` | Select the best usable representation |
| Canonical scene artifact | `dalg_manifest.json`, `schemas/dalg.schema.json` | Stable graph contract across modes |
| API contract | `docs/api/openapi.yaml` | Defines how a future control plane should expose jobs and DALG assets |
| Export surface | `layerforge export-design` | Produce a design-manifest JSON from any completed run |

## Architecture

LayerForge should evolve as four planes, not one giant service:

1. Experience plane
   - web editor
   - shared review links
   - SDKs and docs
2. Control plane
   - tenants, workspaces, projects, recipes, jobs, exports, billing
3. Inference plane
   - native pipeline
   - recursive peeling
   - Qwen raw and Qwen graph enrichment
   - quality routing and self-evaluation
4. Data and telemetry plane
   - immutable assets
   - DALG manifests
   - metrics, traces, audit logs, and recipe history

The current repo now reflects the object model required for that transition,
even though it does not yet ship the control plane or browser editor.

## Why DALG is the canonical object

The canonical DALG export exists to prevent the product from fragmenting into
mode-specific JSON blobs. The same schema now covers:

- native LayerForge runs;
- recursive peeling runs;
- Qwen/external RGBA enrichment runs.

That means the future editor, API, and export adapters can consume one scene
graph shape instead of reverse-engineering multiple manifests.

## Launch strategy

The most credible first customers are:

- e-commerce media operations teams;
- creative operations teams with repeated asset cleanup work;
- AI-native product teams that need a decomposition API rather than a desktop tool.

Recommended sequence:

1. Shared SaaS for fast preview and export.
2. Team features: comments, approvals, recipes, compare views.
3. Enterprise controls: SSO, audit logs, isolated deployments.
4. Public plugin/export SDK only after the DALG contract stays stable.

## Scope discipline

The repo should keep these boundaries:

- still images first;
- API-first and editor-second, not native desktop first;
- productization over training a new foundation model;
- quality routing over adding model soup.

## Implemented now vs later

Implemented now:

- canonical DALG schema and DALG export command;
- contract-first OpenAPI document;
- native/Qwen/peeling normalization into one design-manifest JSON;
- self-evaluation and editability metrics already wired into frontier review.

Planned next:

- control plane and tenant-scoped job model;
- browser graph inspector and review workflow;
- additional exports such as PSD or partner-tool manifests;
- enterprise deployment packaging once repeated demand exists.
