# Figure Pack

This is the figure index for the measured runs currently present in the local repo.

Regenerate everything with:

```bash
.venv/bin/python scripts/generate_report_figures.py
```

Full regeneration requires the local heavyweight `runs/`, `results/`, and `data/` directories. In the submission ZIP, the pre-generated PNG files in `docs/figures/` are the delivered artifacts, and `report_artifacts/figure_sources/figure_manifest.json` records the raw dependencies used to make them.

Generated assets:

- [Truck recomposition comparison](figures/truck_recomposition_comparison.png): reference RGB, raw Qwen, Qwen plus LayerForge, old native LayerForge, automated upgraded LayerForge, and the autotune-selected LayerForge winner on `data/demo/truck.jpg`.
- [Truck layer stack comparison](figures/truck_layer_stack_comparison.png): raw Qwen RGBA layers, enriched Qwen ordered stack, old LayerForge grouped layers, and the tightened autotune-selected LayerForge stack.
- [Truck metrics comparison](figures/truck_metrics_comparison.png): side-by-side charts for layer count, recomposition PSNR, and recomposition SSIM across the main truck baselines and improved LayerForge variants, including the search-selected native winner.
- [Truck prompt ablation](figures/truck_prompt_ablation.png): Gemini-only prompting versus curated prompts versus augment mode on the upgraded native recipe.
- [Synthetic ordering ablation](figures/synthetic_ordering_ablation.png): held-out `boundary` versus `learned ranker` comparison showing that PSNR improves while IoU and PLOA stay flat.
- [Qualitative gallery](figures/qualitative_gallery.png): astronaut, coffee, and cat scenes with input RGB, segmentation overlay, and ordered layer contact sheet.
- [Effects layer demo](figures/effects_layer_demo.png): controlled synthetic `layerbench_pp` scene showing input RGB, clean reference without the object, ground-truth shadow layer, and the extracted associated-effect layer.
- [Public benchmark comparison](figures/public_benchmark_comparison.png): COCO versus ADE20K coarse-group benchmark charts for group mIoU, thing mIoU, stuff mIoU, and mean image mIoU.
- [Public depth comparison](figures/public_depth_comparison.png): DIODE validation charts comparing the geometric fallback against DepthPro on depth error and indoor/outdoor split metrics.
- [Frontier review](figures/frontier_review.png): the hardened five-image self-evaluation candidate bank, comparing native, peeling, raw Qwen, and fair hybrid rows.
- [Prompt extraction benchmark](figures/prompt_extract_benchmark.png): synthetic prompt-conditioned extraction comparison across text, point, box, and hybrid query types.
- [Transparent benchmark](figures/transparent_benchmark.png): AlphaBlend-style synthetic transparent-scene benchmark covering alpha MAE, background PSNR, and recomposition PSNR.
- [Figure manifest](figures/figure_manifest.json): machine-readable index of the figure files.

Recommended use in the report:

- intro / headline comparison: `truck_recomposition_comparison.png`
- methods / representation slide: `truck_layer_stack_comparison.png`
- main quantitative results: `truck_metrics_comparison.png`
- native-method tuning evidence: `truck_prompt_ablation.png`
- novelty / ablation evidence: `synthetic_ordering_ablation.png`
- qualitative results section: `qualitative_gallery.png`
- associated-effect demo section: `effects_layer_demo.png`
- public benchmark section: `public_benchmark_comparison.png`
- public depth section: `public_depth_comparison.png`
- hardened frontier selector section: `frontier_review.png`
- promptable extraction section: `prompt_extract_benchmark.png`
- transparent decomposition section: `transparent_benchmark.png`
