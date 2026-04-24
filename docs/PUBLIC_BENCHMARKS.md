# Public Benchmarks and Reference Datasets

This document catalogs the public datasets and benchmarking protocols integrated into the LayerForge-X framework. It delineates the role of each benchmark, its current implementation status, and the roadmap for future extensions.

## Implemented Benchmarking Tracks

### Optional Dataset Runner

Use the orchestration helper when public datasets are mounted locally:

```bash
python scripts/run_public_benchmarks_if_present.py \
  --data-root data \
  --output-root results/public_benchmarks \
  --preset world_best \
  --max-images 512
```

It checks for the expected COCO Panoptic, ADE20K, and DIODE directories before
launching any benchmark. Missing datasets are reported as `skipped`; present
datasets are executed through the existing `layerforge benchmark-*` CLIs. The
runner writes `public_benchmark_run_report.json` and
`public_benchmark_run_report.md`.

### 1. COCO Panoptic (val2017)
- **Role:** Visible semantic and instance-level grouping benchmark.
- **Reference:** [COCO Dataset](https://cocodataset.org/)
- **Protocol:** Evaluates **coarse-group Intersection-over-Union (mIoU)** for "thing" (instance) and "stuff" (region) categories.
- **Implementation:** Automated download and evaluation harness provided in `scripts/download_coco_panoptic_val.py` and `layerforge benchmark-coco-panoptic`.
- **Measured Performance (Mask2Former):**
  - **Images Evaluated:** 512
  - **Group mIoU:** `0.5660`
  - **Thing mIoU:** `0.5842`
  - **Stuff mIoU:** `0.5479`
- **Summary Artifact:** `report_artifacts/metrics_snapshots/coco_panoptic_group_benchmark_summary.json`

### 2. ADE20K SceneParse150 (Validation)
- **Role:** High-density scene parsing and background/stuff categorization.
- **Reference:** [ADE20K Dataset](https://sceneparsing.csail.mit.edu/)
- **Protocol:** Employs the same coarse-group mIoU protocol as the COCO benchmark, providing a more challenging environment for dense scene understanding.
- **Implementation:** Automated harness via `scripts/download_ade20k.py` and `layerforge benchmark-ade20k`.
- **Measured Performance (Mask2Former):**
  - **Images Evaluated:** 512
  - **Group mIoU:** `0.6015`
  - **Thing mIoU:** `0.5579`
  - **Stuff mIoU:** `0.6451`
  - **Mean Image mIoU:** `0.5569`
- **Summary Artifact:** `report_artifacts/metrics_snapshots/ade20k_group_benchmark_summary.json`

### 3. DIODE Validation
- **Role:** Diverse indoor and outdoor monocular depth benchmark.
- **Reference:** [DIODE Dataset](https://diode-dataset.org/)
- **Protocol:** Validates monocular geometry models (e.g., Depth Pro) against ground-truth RGB-D data. Supports both raw metric evaluation and scale-aligned comparative studies.
- **Implementation:** Automated harness via `scripts/download_diode_val.py` and `layerforge benchmark-diode`.
- **Measured Performance (Depth Pro):**
  - **Raw Metric:** AbsRel `0.5230`, Delta1 `0.4057`, SILog `26.8766`
  - **Scale-Aligned:** AbsRel `0.3629`, RMSE `6.1891`, Delta1 `0.6452`
- **Summary Artifacts:** `report_artifacts/metrics_snapshots/diode_depth_benchmark_summary.json`, `diode_depth_scale_benchmark_summary.json`

## Benchmarking Roadmap and Planned Extensions

The following datasets are identified as high-priority targets for future integration to further validate specific DALG attributes:

### 1. NYU Depth V2
- **Objective:** Establish a robust indoor pairwise-order and depth-accuracy benchmark.
- **Rationale:** NYU Depth V2 provides the highest-quality ground truth for indoor scene geometry and relative object ordering.

### 2. KINS / COCOA
- **Objective:** Quantitative validation of amodal segmentation and occlusion reasoning.
- **Rationale:** These datasets provide ground-truth amodal masks for street scenes and general object categories, facilitating a rigorous assessment of hidden-region estimation.

### 3. Intrinsic Images in the Wild (IIW) / MAW
- **Objective:** Standardized evaluation of albedo and shading decomposition.
- **Rationale:** While intrinsic decomposition is currently a stretch goal, integration with IIW/WHDR metrics will allow for direct comparison against specialized intrinsic decomposition architectures.

## Strategic Significance of Multi-Axial Benchmarking

The integration of these diverse public benchmarks ensures that LayerForge-X is not evaluated in isolation. By measuring performance across visibility (COCO/ADE20K), geometry (DIODE), and synthetic structure (LayerBench), the framework establishes a verifiable and transparent performance baseline for the task of layered scene decomposition.
