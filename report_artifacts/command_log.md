# Command Log

These are the exact command families used to produce the auditable summaries copied into `report_artifacts/metrics_snapshots/`.

## Validation

```bash
./.venv/bin/pytest -q
./.venv/bin/pytest -q tests/test_smoke.py tests/test_merge.py tests/test_segment_api.py
```

## Synthetic ordering benchmark

```bash
python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 20
layerforge benchmark --dataset-dir data/synthetic_layerbench --output-dir results/synthetic_fast --config configs/fast.yaml --segmenter classical --depth geometric_luminance
```

## Qwen baseline and hybrid enrichment

```bash
python scripts/run_qwen_image_layered.py --input data/demo/truck.jpg --output-dir runs/qwen_truck_layers_raw_640_20 --layers 4 --resolution 640 --steps 20 --device cuda --dtype bfloat16 --offload sequential
layerforge enrich-qwen --input data/demo/truck.jpg --layers-dir runs/qwen_truck_layers_raw_640_20 --output runs/qwen_truck_enriched_640_20 --config configs/cutting_edge.yaml --depth depth_pro
```

## Five-image Qwen review

```bash
python scripts/score_qwen_raw_layers.py --input data/demo/truck.jpg --layers-dir runs/qwen_five_image_review/truck/qwen_4
python scripts/score_qwen_raw_layers.py --input data/qualitative_pack/astronaut.png --layers-dir runs/qwen_five_image_review/astronaut/qwen_4
python scripts/score_qwen_raw_layers.py --input data/qualitative_pack/coffee.png --layers-dir runs/qwen_five_image_review/coffee/qwen_4
python scripts/score_qwen_raw_layers.py --input data/qualitative_pack/chelsea_cat.png --layers-dir runs/qwen_five_image_review/chelsea_cat/qwen_4
python scripts/score_qwen_raw_layers.py --input examples/synth/scene_000/image.png --layers-dir runs/qwen_five_image_review/image/qwen_4
python scripts/run_curated_comparison.py --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png data/qualitative_pack/coffee.png data/qualitative_pack/chelsea_cat.png examples/synth/scene_000/image.png --output-root runs/qwen_five_image_review --native-config configs/best_score.yaml --native-segmenter grounded_sam2 --native-depth ensemble --qwen-layers 4 --qwen-steps 10 --qwen-resolution 640 --qwen-device cuda --qwen-dtype bfloat16 --qwen-offload sequential --skip-existing
```

## Associated-effect demo

```bash
python scripts/make_synthetic_dataset.py --output runs/effects_demo_source --count 1 --seed 77 --width 640 --height 420 --output-format layerbench_pp --with-effects
python scripts/run_effects_demo.py --scene-dir runs/effects_demo_source/scene_000 --output runs/effects_groundtruth_demo_cutting_edge --config configs/cutting_edge.yaml
```

## Native search run

```bash
layerforge autotune --input data/demo/truck.jpg --output runs/truck_state_of_art_search_v2 --config configs/best_score.yaml --prompts "truck,road,sky,tree,building,window,wheel,car" --device cuda --no-parallax
```

## Public grouping and depth benchmarks

```bash
layerforge benchmark-coco-panoptic --dataset-dir data/coco_panoptic_val --output-dir results/coco_panoptic_mask2former_512 --config configs/fast.yaml --segmenter mask2former --device cuda --max-images 512 --seed 7
layerforge benchmark-ade20k --dataset-dir data/ade20k --output-dir results/ade20k_mask2former_512 --config configs/ade20k_mask2former.yaml --segmenter mask2former --device cuda --max-images 512 --seed 7
layerforge benchmark-diode --dataset-dir data/diode --output-dir results/diode_depthpro_full --config configs/diode_depthpro.yaml --depth depth_pro --device cuda --seed 7
layerforge benchmark-diode --dataset-dir data/diode --output-dir results/diode_depthpro_scale_full --config configs/diode_depthpro.yaml --depth depth_pro --alignment scale --device cuda --seed 7
layerforge benchmark-diode --dataset-dir data/diode --output-dir results/diode_geometric_full --config configs/diode_depthpro.yaml --depth geometric_luminance --alignment scale --device cpu --seed 7
```
