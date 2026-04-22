# Results Summary (2026-04-22 update)

This file is the ground-truth snapshot of what has actually been run in the repo so far. Use it when writing the report, slides, or viva notes.

## Environment

- machine: RTX 5090 laptop GPU
- Python: `3.14.4`
- torch: `2.11.0+cu130`
- validation: `pytest -q` passed with `40` tests

## What was implemented

- deterministic fast pipeline
- optional model-backed segmentation/depth hooks
- Qwen/external RGBA layer enrichment
- synthetic benchmark generator and benchmark CLI
- learned pairwise layer-order ranker trained on synthetic scenes
- compatibility fix for current `transformers` GroundingDINO post-processing API
- real GPU-backed run verified with `grounded_sam2 + depth_pro`
- Gemini-assisted prompt generation for open-vocabulary segmentation
- adaptive layer merging to reduce fragmentation in the native pipeline
- prompt-strategy ablation for the upgraded native recipe
- COCO Panoptic coarse-group benchmark on a real public dataset
- ADE20K coarse-group benchmark on a real public dataset
- DIODE depth benchmark on a real public dataset

## Commands already run

Base validation:

```bash
.venv/bin/pytest -q
.venv/bin/pytest -q tests/test_smoke.py tests/test_merge.py tests/test_segment_api.py
python scripts/export_report_artifacts.py
python scripts/build_report_docx.py
```

Synthetic benchmark dataset:

```bash
.venv/bin/python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 12
```

Fast baseline benchmark:

```bash
.venv/bin/layerforge benchmark \
  --dataset-dir data/synthetic_layerbench \
  --output-dir results/synthetic_fast \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```

Ranker training:

```bash
.venv/bin/python scripts/make_synthetic_dataset.py --output data/synth_train --count 40 --seed 101
.venv/bin/python scripts/make_synthetic_dataset.py --output data/synth_test --count 12 --seed 501

.venv/bin/layerforge train-ranker \
  --dataset-dir data/synth_train \
  --output models/order_ranker_fast.json \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance
```

Held-out comparison:

```bash
.venv/bin/layerforge benchmark \
  --dataset-dir data/synth_test \
  --output-dir results/synth_boundary_test \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance \
  --ordering boundary

.venv/bin/layerforge benchmark \
  --dataset-dir data/synth_test \
  --output-dir results/synth_learned_test \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance \
  --ordering learned \
  --ranker-model models/order_ranker_fast.json
```

COCO Panoptic download:

```bash
python scripts/download_coco_panoptic_val.py \
  --output-dir data/coco_panoptic_val \
  --archive-dir data/downloads/coco
```

COCO visible-group benchmark:

```bash
.venv/bin/layerforge benchmark-coco-panoptic \
  --dataset-dir data/coco_panoptic_val \
  --output-dir results/coco_panoptic_mask2former_512 \
  --config configs/fast.yaml \
  --segmenter mask2former \
  --device cuda \
  --max-images 512 \
  --seed 7
```

## Main measured results

### Fast deterministic baseline

Source: `results/synthetic_fast/synthetic_benchmark_summary.json`

| Metric | Value |
|---|---:|
| mean best IoU | 0.1549 |
| pairwise layer-order accuracy | 0.1667 |
| mean recompose PSNR | 19.1360 |

### Held-out ordering ablation

Sources:

- `results/synth_boundary_test/synthetic_benchmark_summary.json`
- `results/synth_learned_test/synthetic_benchmark_summary.json`

| Variant | Mean best IoU | PLOA | Mean recompose PSNR | Mean recompose SSIM |
|---|---:|---:|---:|---:|
| boundary ordering | 0.1549 | 0.1667 | 19.1589 | 0.8966 |
| learned ranker | 0.1549 | 0.1667 | 19.4138 | 0.8954 |

Delta from learned ordering over boundary ordering:

- PSNR: `+0.2549` dB
- mean best IoU: `0.0000`
- PLOA: `0.0000`

### Public COCO visible-group benchmark

Source: `results/coco_panoptic_mask2former_512/coco_panoptic_group_benchmark_summary.json`

What this benchmark measures:

- visible semantic grouping only;
- public COCO panoptic categories collapsed into LayerForge coarse groups;
- dataset-level IoU rather than full COCO PQ.

Measured result on a random `512`-image validation sample with seed `7`:

| Metric | Value |
|---|---:|
| group mIoU | 0.5660 |
| thing mIoU | 0.5842 |
| stuff mIoU | 0.5479 |
| mean predicted segments | 6.4492 |

Per-group IoU:

| Group | IoU |
|---|---:|
| person | 0.6854 |
| animal | 0.6483 |
| vehicle | 0.6801 |
| furniture | 0.4245 |
| plant | 0.7115 |
| sky | 0.8632 |
| road | 0.3975 |
| ground | 0.4634 |
| building | 0.5785 |
| water | 0.7798 |
| stuff | 0.2048 |
| object | 0.3556 |

Interpretation:

- the public-dataset path is now real, not hypothetical;
- `mask2former` is a strong closed-set visible-grouping baseline on COCO and gives the repo an external benchmark that is distinct from the synthetic layer benchmark;
- this does **not** replace the synthetic benchmark, because COCO does not supervise layer order, amodal completion, or intrinsics.

### Public ADE20K visible-group benchmark

Source: `results/ade20k_mask2former_512/ade20k_group_benchmark_summary.json`

What this benchmark measures:

- visible semantic grouping only;
- public ADE20K SceneParse150 validation labels collapsed into LayerForge coarse groups;
- dataset-level IoU and per-image mean IoU rather than official 150-class ADE mIoU.

Command used:

```bash
python scripts/download_ade20k.py \
  --output-dir data/ade20k \
  --archive-dir data/downloads/ade20k

.venv/bin/layerforge benchmark-ade20k \
  --dataset-dir data/ade20k \
  --output-dir results/ade20k_mask2former_512 \
  --config configs/ade20k_mask2former.yaml \
  --segmenter mask2former \
  --device cuda \
  --max-images 512 \
  --seed 7
```

Measured result on a random `512`-image validation sample with seed `7`:

| Metric | Value |
|---|---:|
| group mIoU | 0.6015 |
| thing mIoU | 0.5579 |
| stuff mIoU | 0.6451 |
| mean image mIoU | 0.5569 |
| median image mIoU | 0.5867 |
| mean predicted segments | 6.2480 |

Per-group IoU:

| Group | IoU |
|---|---:|
| animal | 0.3710 |
| building | 0.7757 |
| furniture | 0.6403 |
| ground | 0.5105 |
| object | 0.4669 |
| person | 0.5412 |
| plant | 0.6387 |
| road | 0.8300 |
| sky | 0.8455 |
| stuff | 0.3504 |
| vehicle | 0.6896 |
| water | 0.5585 |

Interpretation:

- this benchmark is broader and more scene-centric than COCO;
- the ADE-specific Mask2Former checkpoint improves the realism of the public benchmark path because it is tuned for ADE scene parsing rather than COCO panoptic categories;
- the public real-data story is now two-dataset rather than one-dataset: COCO for visible grouping in common scenes, ADE20K for broader scene parsing.

### Public DIODE depth benchmark

Sources:

- `results/diode_geometric_full/diode_depth_benchmark_summary.json`
- `results/diode_depthpro_full/diode_depth_benchmark_summary.json`
- `results/diode_depthpro_scale_full/diode_depth_benchmark_summary.json`

What this benchmark measures:

- public RGB-D depth quality on indoor and outdoor scenes;
- honest metric-depth performance for `depth_pro`;
- scale-aligned relative-shape quality for a fair comparison against the geometric fallback.

Commands used:

```bash
python scripts/download_diode_val.py \
  --output-dir data/diode \
  --archive-dir data/downloads/diode

.venv/bin/layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_geometric_full \
  --config configs/diode_depthpro.yaml \
  --depth geometric_luminance \
  --alignment scale \
  --device cpu \
  --seed 7

.venv/bin/layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_depthpro_full \
  --config configs/diode_depthpro.yaml \
  --depth depth_pro \
  --device cuda \
  --seed 7

.venv/bin/layerforge benchmark-diode \
  --dataset-dir data/diode \
  --output-dir results/diode_depthpro_scale_full \
  --config configs/diode_depthpro.yaml \
  --depth depth_pro \
  --alignment scale \
  --device cuda \
  --seed 7
```

Measured result on the full `771`-image DIODE validation split:

| Variant | Alignment | AbsRel | RMSE | delta1 | SILog |
|---|---|---:|---:|---:|---:|
| geometric luminance | scale | 0.6298 | 7.0934 | 0.2714 | 184.6629 |
| depth_pro | none | 0.5230 | 29.0380 | 0.4057 | 26.8766 |
| depth_pro | scale | 0.3629 | 6.1891 | 0.6452 | 26.8766 |

Indoor/outdoor split on `AbsRel`:

| Variant | Alignment | Indoor AbsRel | Outdoor AbsRel |
|---|---|---:|---:|
| geometric luminance | scale | 0.4822 | 0.7373 |
| depth_pro | none | 0.1995 | 0.7588 |
| depth_pro | scale | 0.0880 | 0.5632 |

Interpretation:

- raw `depth_pro` is the honest metric-depth result and shows a clear indoor strength, but outdoor absolute scale still drifts;
- for a fair relative-shape comparison, scale-aligned `depth_pro` beats the geometric fallback by a large margin on `AbsRel`, `delta1`, and `SILog`;
- DIODE gives the repo a real public depth axis in addition to COCO and ADE20K, so the external evaluation story is no longer semantics-only.

### Real-image qualitative run

Source directory: `runs/demo_grounded_depthpro_final`

Command:

```bash
.venv/bin/layerforge run \
  --input data/demo/truck.jpg \
  --output runs/demo_grounded_depthpro_final \
  --config configs/cutting_edge.yaml \
  --segmenter grounded_sam2 \
  --depth depth_pro \
  --prompts 'truck,road,sky,tree,building,window,wheel,car' \
  --device cuda \
  --no-parallax
```

Observed output:

- end-to-end run completed successfully
- `45` ordered RGBA layers were exported
- segmentation method: `grounded_sam2`
- depth method: `depth_pro`
- inpainting method actually used: `opencv_telea_fallback`
- recompose PSNR: `14.6477`
- recompose SSIM: `0.8348`

Important reading of this run:

- it proves that the full promptable, GPU-backed pipeline works on a real image on this machine;
- the layer count is still high, so the main qualitative discussion should emphasize interpretability and editability rather than claiming minimal layers;
- the fallback inpainting path is now part of the verified execution path on Python `3.14`.

### Upgraded native best-score runs

These are the runs that materially changed the native pipeline quality on `data/demo/truck.jpg`.

Automated best-score run:

```bash
.venv/bin/layerforge run \
  --input data/demo/truck.jpg \
  --output runs/truck_best_score \
  --config configs/best_score.yaml \
  --device cuda \
  --no-parallax
```

Highest measured run:

```bash
.venv/bin/layerforge run \
  --input data/demo/truck.jpg \
  --output runs/truck_best_score_manual \
  --config configs/best_score.yaml \
  --prompts 'truck,road,sky,tree,building,window,wheel,car' \
  --prompt-source manual \
  --device cuda \
  --no-parallax
```

Prompt-augmentation run:

```bash
.venv/bin/layerforge run \
  --input data/demo/truck.jpg \
  --output runs/truck_best_score_augment \
  --config configs/best_score.yaml \
  --prompts 'truck,road,sky,tree,building,window,wheel,car' \
  --prompt-source augment \
  --device cuda \
  --no-parallax
```

Measured comparison:

| Run | Prompt strategy | Layers | Pre-merge layers | Merge reduction | PSNR | SSIM |
|---|---|---:|---:|---:|---:|---:|
| `runs/demo_grounded_depthpro_final` | older native recipe | 45 | - | - | 14.6477 | 0.8348 |
| `runs/truck_best_score` | Gemini-assisted prompts | 26 | 38 | 13 | 30.8214 | 0.9812 |
| `runs/truck_best_score_augment` | curated + Gemini augment | 19 | 31 | 13 | 31.1524 | 0.9804 |
| `runs/truck_best_score_manual` | curated prompts | 19 | 35 | 17 | 31.3040 | 0.9813 |
| `runs/truck_state_of_art_search_v2/best` | autotune-selected winner (`manual_precision`) | 20 | 32 | 13 | 32.1053 | 0.9848 |

Important reading of these runs:

- the native LayerForge path is now dramatically better than the old truck run on both recomposition fidelity and layer count;
- Gemini-only prompting is a good automatic mode, but it is not the best measured score on this truck scene;
- the strongest result still comes from good prompt curation plus the new depth ensemble and merge pass.
- the strongest overall native result now comes from the new `autotune` search mode, which selects the best candidate by measured PSNR, SSIM, and layer count.

### Qualitative pack

Source directory: `runs/qualitative_pack_cutting_edge`

Inputs:

- `data/qualitative_pack/astronaut.png`
- `data/qualitative_pack/coffee.png`
- `data/qualitative_pack/chelsea_cat.png`

Command:

```bash
.venv/bin/layerforge batch \
  --input-dir data/qualitative_pack \
  --output-dir runs/qualitative_pack_cutting_edge \
  --config configs/cutting_edge.yaml \
  --segmenter grounded_sam2 \
  --depth depth_pro \
  --prompts 'person,animal,cat,coffee,mug,food,plant,sky,building,window,road,vehicle,bag,phone' \
  --device cuda \
  --no-parallax
```

Compact metrics:

| Run | Layers | PSNR | SSIM | Inpaint |
|---|---:|---:|---:|---|
| astronaut | 13 | 18.9531 | 0.9007 | opencv_telea_fallback |
| coffee | 9 | 34.5492 | 0.9847 | opencv_telea_fallback |
| chelsea_cat | 10 | 41.9994 | 0.9974 | opencv_telea_fallback |

Best figure entry points for the report:

- `docs/figures/qualitative_gallery.png`
- `docs/figures/truck_recomposition_comparison.png`
- `docs/figures/truck_layer_stack_comparison.png`

These runs are useful because they show the system on multiple real-image categories with much cleaner layer counts than the truck street scene.

## Qwen baseline and hybrid comparison

The official Qwen baseline path is not just wired in; it has been run successfully in this repo:

```bash
.venv/bin/python scripts/run_qwen_image_layered.py \
  --input data/demo/truck.jpg \
  --output-dir runs/qwen_truck_layers_raw_640_20 \
  --layers 4 \
  --resolution 640 \
  --steps 20 \
  --device cuda \
  --dtype bfloat16 \
  --offload sequential

.venv/bin/layerforge enrich-qwen \
  --input data/demo/truck.jpg \
  --layers-dir runs/qwen_truck_layers_raw_640_20 \
  --output runs/qwen_truck_enriched_640_20 \
  --config configs/cutting_edge.yaml \
  --depth depth_pro
```

Measured truck comparison:

| Run | Mode | Layers | PSNR | SSIM |
|---|---|---:|---:|---:|
| `runs/qwen_truck_layers_raw_640_20` | raw Qwen RGBA | 4 | 26.7874 | 0.7723 |
| `runs/qwen_truck_enriched_640_20` | Qwen + LayerForge graph enrichment | 2 | 27.4612 | 0.7953 |
| `runs/demo_grounded_depthpro_final` | old native LayerForge | 45 | 14.6477 | 0.8348 |
| `runs/truck_best_score` | improved native LayerForge, automated | 26 | 30.8214 | 0.9812 |
| `runs/truck_best_score_manual` | improved native LayerForge, best measured | 19 | 31.3040 | 0.9813 |
| `runs/truck_state_of_art_search_v2/best` | improved native LayerForge, autotune winner | 20 | 32.1053 | 0.9848 |

Notes:

- the Qwen run used the official `Qwen/Qwen-Image-Layered` diffusers pipeline
- because the model is too large for a naive full-GPU load on a 24 GB card, the successful run used `--offload sequential`
- the raw-Qwen recomposition metric now scores both manifest and reversed-manifest interpretations and keeps the better reconstruction
- the old native LayerForge run clearly lost to Qwen on this example
- the upgraded native LayerForge recipe now beats both the raw Qwen row and the `Q+G` hybrid on this truck image
- the autotune-selected native winner is the strongest measured run currently in the repo on this truck benchmark
- Qwen is still the right frontier baseline because it remains a strong low-layer generative decomposer without prompt curation

That means the submission can now honestly say:

- Qwen is treated as the correct frontier baseline
- the repo contains a measured raw Qwen baseline, a measured `Q+G` hybrid result, and multiple measured native LayerForge variants
- completed measured results cover fast synthetic ablations, learned ordering, real-image qualitative packs, direct Qwen comparisons, and a prompt-strategy ablation on the native path

### Five-image Qwen raw versus hybrid review

The repo now also contains a measured five-image Qwen review:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/run_curated_comparison.py \
  --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png data/qualitative_pack/coffee.png data/qualitative_pack/chelsea_cat.png examples/synth/scene_000/image.png \
  --output-root runs/qwen_five_image_review \
  --native-config configs/best_score.yaml \
  --native-segmenter grounded_sam2 \
  --native-depth ensemble \
  --qwen-layers 4 \
  --qwen-steps 10 \
  --qwen-resolution 640 \
  --qwen-device cuda \
  --qwen-dtype bfloat16 \
  --qwen-offload sequential \
  --skip-existing
```

Aggregate mean results:

| Method | Images | Graph | Mean PSNR | Mean SSIM | Mean amodal extra ratio |
|---|---:|---|---:|---:|
| `LayerForge native` | 5 | yes | 27.3438 | 0.9464 | 0.3057 |
| `Qwen raw (4)` | 5 | no | 29.0757 | 0.8850 | 0.0000 |
| `Qwen + graph preserve (4)` | 5 | yes | 28.5539 | 0.8638 | 2.9970 |
| `Qwen + graph reorder (4)` | 5 | yes | 28.5397 | 0.8637 | 2.9970 |

Per-image results:

| Image | Native PSNR | Native SSIM | Qwen raw PSNR | Qwen raw SSIM | Q+G preserve PSNR | Q+G preserve SSIM | Q+G reorder PSNR | Q+G reorder SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| truck | 28.8247 | 0.9732 | 28.8062 | 0.8614 | 26.8096 | 0.7829 | 26.8160 | 0.7827 |
| astronaut | 26.1679 | 0.9299 | 29.4532 | 0.9091 | 29.3737 | 0.9153 | 29.3737 | 0.9153 |
| coffee | 29.4754 | 0.9788 | 28.2737 | 0.8762 | 28.0643 | 0.8719 | 28.0333 | 0.8718 |
| chelsea_cat | 27.4906 | 0.9562 | 29.5828 | 0.9050 | 29.5169 | 0.8920 | 29.4704 | 0.8918 |
| synth image | 24.7604 | 0.8939 | 29.2627 | 0.8733 | 29.0052 | 0.8571 | 29.0052 | 0.8571 |

What this means:

- raw Qwen still wins mean PSNR on this five-image review;
- native LayerForge now wins mean SSIM on the same images, but it does so with a much larger average stack (`16.6` layers versus Qwen's `4`);
- `Qwen + graph preserve` is now the fair metadata-first hybrid row because it keeps the interpreted Qwen visual stack and adds graph, amodal, and intrinsic artifacts;
- `Qwen + graph reorder` is the separate graph-order export row, and it remains only slightly below the preserve-order variant on this sweep.

## Frontier five-image self-evaluation review

The repo now also contains the measured frontier candidate-bank run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/run_frontier_comparison.py \
  --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png data/qualitative_pack/coffee.png data/qualitative_pack/chelsea_cat.png examples/synth/scene_000/image.png \
  --output-root runs/frontier_review \
  --native-config configs/frontier.yaml \
  --peeling-config configs/recursive_peeling.yaml \
  --qwen-layers 4 \
  --qwen-steps 10 \
  --qwen-resolution 640 \
  --qwen-device cuda \
  --qwen-dtype bfloat16 \
  --qwen-offload sequential \
  --skip-existing
```

Aggregate mean results:

| Method | Images | Mean PSNR | Mean SSIM | Mean self-eval score | Best-image wins |
|---|---:|---:|---:|---:|---:|
| `LayerForge native` | 5 | 37.6688 | 0.9708 | 0.6597 | 3 |
| `LayerForge peeling` | 5 | 27.0988 | 0.9096 | 0.5050 | 1 |
| `Qwen raw (4)` | 5 | 29.0757 | 0.8850 | 0.2530 | 0 |
| `Qwen + graph preserve (4)` | 5 | 28.5539 | 0.8638 | 0.4951 | 1 |
| `Qwen + graph reorder (4)` | 5 | 28.5397 | 0.8637 | 0.4949 | 0 |

Best-per-image selections from `best_decomposition.json`:

| Image | Selected candidate |
|---|---|
| truck | `LayerForge peeling` |
| astronaut | `LayerForge native` |
| chelsea_cat | `LayerForge native` |
| coffee | `LayerForge native` |
| synth image | `Qwen + graph preserve (4)` |

What this means:

- the current measured frontier score now prefers `LayerForge native` overall and selects it for `3/5` images;
- the recursive path is no longer only conceptual: `LayerForge peeling` wins the truck scene in the measured candidate bank;
- the fair hybrid remains useful because `Qwen + graph preserve` wins the synthetic scene where the compact external stack is already strong;
- `Qwen raw` still matters as the compact generative baseline, but the full candidate-bank comparison shows that raw RGBA quality alone is not the same as the best editable representation.

## Recursive peeling and associated-effect demos

The repo also now contains measured demo artifacts for the new recursive-peeling and effect-aware paths.

Recursive peeling demo:

```bash
.venv/bin/layerforge peel \
  --input runs/effects_demo_source/scene_000/image.png \
  --output runs/effects_demo_peel \
  --config configs/fast.yaml \
  --segmenter classical \
  --depth geometric_luminance \
  --max-layers 4
```

Measured result:

| Run | Mode | Layers | Effect layers | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| `runs/effects_demo_peel` | recursive peeling | 5 | 0 | 33.1022 | 0.9724 |

Associated-effect extractor demo on a `layerbench_pp` synthetic scene:

```bash
.venv/bin/python scripts/run_effects_demo.py \
  --scene-dir runs/effects_demo_source/scene_000 \
  --output runs/effects_groundtruth_demo_cutting_edge \
  --config configs/cutting_edge.yaml
```

Measured result:

| Run | Effect detected | Predicted effect px | Ground-truth effect px | Effect IoU |
|---|---|---:|---:|---:|
| `runs/effects_groundtruth_demo_cutting_edge` | yes | 4853 | 13750 | 0.3529 |

Notes:

- the effect demo proves that the repo now has a real, exportable associated-effect artifact and a matching figure at `docs/figures/effects_layer_demo.png`;
- the clean-reference rerun materially improves the heuristic, but the current effect-layer claim should still be framed as "early heuristic demo" rather than "solved shadow decomposition."

## Report figure pack

The repo now contains a generated figure pack derived directly from the measured runs:

```bash
.venv/bin/python scripts/generate_report_figures.py
```

Main figure files:

- `docs/figures/truck_recomposition_comparison.png`
- `docs/figures/truck_layer_stack_comparison.png`
- `docs/figures/truck_metrics_comparison.png`
- `docs/figures/truck_prompt_ablation.png`
- `docs/figures/synthetic_ordering_ablation.png`
- `docs/figures/qualitative_gallery.png`
- `docs/figures/effects_layer_demo.png`

Use `docs/FIGURES.md` for the figure index and suggested report placement.

## Honest interpretation

- The learned pairwise ranker helps recomposition on held-out synthetic scenes.
- The current pairwise ordering metric does not improve because the classical fast segmenter badly over-segments the image.
- In the current synthetic runs, the system predicts about `65` layers for scenes with `5` ground-truth layers.
- The strongest improvement in this repo came from stronger proposals, better merging, and test-time candidate selection, not from a bigger ordering model.
- On scenes where maximum score matters, prompt quality still matters a lot for open-vocabulary segmentation.

## Important caveat

`requirements-models.txt` currently includes `simple-lama-inpainting`, which fails to build on Python `3.14` because of an older Pillow dependency. The rest of the model-backed stack used here works with:

```bash
.venv/bin/python -m pip install torch torchvision transformers accelerate diffusers safetensors
```

The repo still runs because it falls back to OpenCV inpainting when LaMa is unavailable.

## Best next experiments for marks

1. Run `grounded_sam2 + depth_pro` on a curated real-image set and include qualitative layer stacks.
2. Create a small prompt sweep per image to show controllability of open-vocabulary segmentation.
3. Import Qwen-Image-Layered RGBA outputs with `enrich-qwen` and compare "generated layers" vs "generated layers + graph enrichment".
4. Show one failure case where over-segmentation breaks ordering, then explain how stronger proposals fix it.
5. In the report, frame the novelty as the representation plus benchmarked fusion, not as a new foundation model.
