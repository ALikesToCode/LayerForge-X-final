# Public Benchmarks

This file tracks which public benchmarks are actually usable for LayerForge-X today, what each benchmark supervises, and what still needs new loaders or metric code.

## Implemented now

### COCO Panoptic val2017

- role: visible semantic grouping benchmark
- official site: https://cocodataset.org/
- official downloads:
  - `http://images.cocodataset.org/zips/val2017.zip`
  - `http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip`
- repo support: implemented
- command:

```bash
python scripts/download_coco_panoptic_val.py \
  --output-dir data/coco_panoptic_val \
  --archive-dir data/downloads/coco

layerforge benchmark-coco-panoptic \
  --dataset-dir data/coco_panoptic_val \
  --output-dir results/coco_panoptic_mask2former_512 \
  --config configs/fast.yaml \
  --segmenter mask2former \
  --device cuda \
  --max-images 512 \
  --seed 7
```

- measured result in repo:
  - `images_evaluated`: `512`
  - `group mIoU`: `0.5660`
  - `thing mIoU`: `0.5842`
  - `stuff mIoU`: `0.5479`
  - summary: `results/coco_panoptic_mask2former_512/coco_panoptic_group_benchmark_summary.json`

Notes:

- this is a **coarse-group IoU** benchmark, not full COCO PQ;
- COCO is useful for visible grouping, but it cannot score near/far order, amodal completion, or intrinsic decomposition.

### ADE20K SceneParse150 validation

- role: dense scene parsing and stronger background/stuff supervision
- official site: https://sceneparsing.csail.mit.edu/
- official downloads:
  - `https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`
- repo support: implemented
- command:

```bash
python scripts/download_ade20k.py \
  --output-dir data/ade20k \
  --archive-dir data/downloads/ade20k

layerforge benchmark-ade20k \
  --dataset-dir data/ade20k \
  --output-dir results/ade20k_mask2former_512 \
  --config configs/ade20k_mask2former.yaml \
  --segmenter mask2former \
  --device cuda \
  --max-images 512 \
  --seed 7
```

- measured result in repo:
  - `images_evaluated`: `512`
  - `group mIoU`: `0.6015`
  - `thing mIoU`: `0.5579`
  - `stuff mIoU`: `0.6451`
  - `mean image mIoU`: `0.5569`
  - summary: `results/ade20k_mask2former_512/ade20k_group_benchmark_summary.json`

Notes:

- this uses the same **coarse-group IoU** protocol as the COCO benchmark, not the official 150-class ADE metric;
- ADE20K is a stronger public benchmark than COCO for scene/background parsing and complements COCO well.

### DIODE validation

- role: public RGB-D depth benchmark with indoor and outdoor scenes
- official site: https://diode-dataset.org/
- official downloads:
  - `https://diode-dataset.s3.amazonaws.com/val.tar.gz`
- repo support: implemented
- commands:

```bash
python scripts/download_diode_val.py \
  --output-dir data/diode \
  --archive-dir data/downloads/diode

layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_depthpro_full \
  --config configs/diode_depthpro.yaml \
  --depth depth_pro \
  --device cuda \
  --seed 7

layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_depthpro_scale_full \
  --config configs/diode_depthpro.yaml \
  --depth depth_pro \
  --alignment scale \
  --device cuda \
  --seed 7
```

- measured results in repo:
  - raw `depth_pro`: `AbsRel 0.5230`, `delta1 0.4057`, `SILog 26.8766`
  - scale-aligned `depth_pro`: `AbsRel 0.3629`, `RMSE 6.1891`, `delta1 0.6452`
  - geometric scale baseline: `AbsRel 0.6298`, `RMSE 7.0934`, `delta1 0.2714`
  - summaries:
    - `results/diode_depthpro_full/diode_depth_benchmark_summary.json`
    - `results/diode_depthpro_scale_full/diode_depth_benchmark_summary.json`
    - `results/diode_geometric_full/diode_depth_benchmark_summary.json`

Notes:

- this gives the repo a real public **depth** benchmark instead of leaving depth validated only on synthetic data;
- the raw `depth_pro` run is the direct metric-depth result;
- the scale-aligned `depth_pro` run is the fair head-to-head comparison against the geometric fallback.

## Planned public benchmark extensions

### NYU Depth V2

- role: indoor depth and pairwise order benchmark
- public source: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- why it matters: the cleanest public benchmark for indoor RGB-D depth ordering.
- status: next best target for a public **pairwise order accuracy** benchmark.

### KINS

- role: amodal instance segmentation with occlusion reasoning in street scenes
- public source: https://dblp.org/rec/conf/cvpr/QiJ0SJ19.html
- why it matters: directly relevant to modal/amodal mask quality and occluded vehicles/objects.
- status: needs an amodal evaluator and instance matching path.

### COCOA

- role: semantic amodal segmentation
- public source: https://github.com/Wakeupbuddy/amodalAPI
- why it matters: one of the standard public amodal segmentation benchmarks.
- status: needs COCOA-format mask loading and amodal IoU evaluation.

### MAW / IIW-style intrinsic benchmarks

- role: albedo / shading evaluation
- public sources:
  - MAW: https://measuredalbedo.github.io/
  - OpenSurfaces / IIW-related Cornell resources: https://opensurfaces.cs.cornell.edu/
- why it matters: intrinsic decomposition is a stretch goal in this project, and these datasets are the most direct public path for evaluating it.
- status: not integrated yet; good next target after visible grouping and depth/order benchmarks.

## Recommended benchmark narrative

For the report, the clean public-benchmark story is:

1. COCO Panoptic for visible grouping quality.
2. ADE20K for broader scene parsing and background/stuff grouping.
3. DIODE for real public depth supervision on indoor and outdoor scenes.
4. Synthetic LayerBench for full layered evaluation with known order and known RGBA ground truth.
5. NYU Depth V2 for a stronger pairwise-order benchmark.
6. KINS / COCOA for amodal evaluation.
7. MAW / IIW-style data for intrinsic evaluation if that stretch goal is emphasized.
