# Examples

This directory contains lightweight example inputs that work immediately after installation.

## Synthetic smoke scenes

`examples/synth/scene_000` through `scene_002` are compact synthetic scenes used for smoke tests and deterministic regression checks.

Each scene contains:

- `image.png`: the input RGB image
- `ground_truth.json`: compact scene metadata used by the synthetic benchmark utilities

## Immediate commands

Generate a fresh smoke run:

```bash
layerforge run \
  --input examples/synth/scene_000/image.png \
  --output runs/smoke \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance \
  --no-parallax
```

Run the light synthetic benchmark:

```bash
layerforge benchmark \
  --dataset-dir examples/synth \
  --output-dir results/examples_benchmark \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```
