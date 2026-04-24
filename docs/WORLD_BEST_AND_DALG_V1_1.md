# LayerForge World-Best Runtime Notes

LayerForge represents a single RGB image as a Depth-Aware Amodal Layer Graph
(DALG): ordered RGBA layers, semantic groups, visible/amodal/hidden masks,
completed hidden content, alpha confidence, depth evidence, intrinsic
albedo/shading, diagnostics, and metrics.

## Architecture

- `segment.py` produces visible proposals from classical, Mask2Former,
  Grounded-SAM2, Gemini, or hybrid fusion.
- `depth.py` produces normalized near-to-far depth with configurable
  orientation calibration.
- `graph.py` builds per-mask depth stats, evidence-carrying graph edges,
  same-plane relations, cycle resolution, amodal masks, hidden masks, and
  completed RGBA layers.
- `matting.py` provides heuristic and model-backed alpha refinement with
  fallback.
- `inpaint.py` completes removed background regions and hidden object regions
  through pluggable CPU-safe backends.
- `intrinsics.py` computes albedo/shading and residual metrics.
- `dalg.py` writes DALG v1.1 while still reading v1.0 manifests.
- `validation.py` checks output files, graph evidence, alpha validity,
  recomposition residuals, and DALG validity.

## Backend Registry

Run:

```bash
layerforge doctor --preset world_best
```

The registry reports segmentation, depth, matting, inpainting, intrinsics,
amodal, and generative-layer backends. Optional model backends do not hard-fail
when missing; they report warnings and fall back to deterministic CPU-safe
methods.

## DALG v1.1

DALG v1.1 adds:

- global `dalg_version`, `alpha_mode`, `color_space`, `input_hash`,
  `model_manifest`, `creation_time`, and `config_hash`
- per-layer visible/amodal/hidden/completed paths, depth stats, semantic
  labels, provenance, and quality metrics
- edge `relation`, `evidence`, `confidence`, and `conflict_notes`

Existing v1.0 manifests load through `load_dalg_manifest(...)` and receive a
compatibility `dalg_version` marker.

## World-Best Preset

Run:

```bash
layerforge run --preset world_best input.jpg -o out/
layerforge benchmark --preset world_best --dataset-dir examples/synth --output-dir results/world_best
```

The preset enables hybrid segmentation fusion, ensemble depth, model-backed
matting selection, amodal completion, layer-aware inpainting, intrinsic
decomposition, DALG v1.1 output, and contact-sheet diagnostics. Missing
checkpoints fall back gracefully.

The local Web UI exposes the preset from the config selector and returns a
layer inspector with visibility toggles, per-layer depth statistics, graph edge
evidence, mask/completion/albedo/shading asset links, validation status, and a
recomposition-error heatmap.

## Benchmarks

The synthetic benchmark writes CSV, JSON, and Markdown summaries for:

- semantic segmentation quality
- alpha matting quality
- depth ordering quality
- amodal mask quality
- inpainting quality
- intrinsic decomposition quality
- recomposition quality
- runtime and peak memory

Public COCO/ADE20K/DIODE scripts remain optional and run only when datasets are
present.

## Model Installation

Base install:

```bash
python -m pip install -e .[dev]
```

Optional model stack:

```bash
python -m pip install -r requirements-models.txt
```

`simple-lama-inpainting` is skipped on Python 3.14 and newer because its
published package pins `pillow<10`; OpenCV Telea remains the default CPU-safe
inpainting fallback in that environment.

Optional external backends such as SAM2/Grounding-DINO, BiRefNet, Qwen layered
generation, SameObject-style amodal models, diffusion inpainting, or intrinsic
models should be installed according to their upstream checkpoint licenses and
hardware requirements. LayerForge will continue with CPU-safe fallbacks if they
are absent.

## CPU Fallbacks

- segmentation: classical grid/proposal fallback
- depth: geometric luminance fallback
- matting: heuristic trimap/distance-transform alpha
- amodal: morphology and dilation fallback
- inpainting: OpenCV Telea fallback
- intrinsics: Retinex or identity fallback

## Example Domains

- Photorealistic: use `world_best` with Grounded-SAM2/ensemble depth when
  checkpoints are installed.
- Anime/stylized: use prompts plus fusion to keep character/effect layers
  grouped.
- Vector/flat graphics: classical segmentation plus heuristic alpha and Retinex
  often gives stable CPU-only outputs.
