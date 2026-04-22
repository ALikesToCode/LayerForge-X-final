# LayerForge API Notes

`openapi.yaml` is intentionally ahead of the current CLI implementation.

This repository still exposes its engine primarily through Python and the
`layerforge` CLI, but the product direction is API-first. The OpenAPI contract
therefore serves two purposes:

- it freezes the public object model around the canonical DALG artifact;
- it gives future control-plane work a stable contract before any web service is
  introduced.

Current repo surface:

- `layerforge run`
- `layerforge peel`
- `layerforge enrich-qwen`
- `layerforge extract`
- `layerforge export-design`

Future service surface represented in `openapi.yaml`:

- job submission
- job status
- DALG retrieval
- design-manifest export retrieval
- immutable asset metadata lookup
