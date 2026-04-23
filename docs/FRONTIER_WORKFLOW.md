# LayerForge-X++ Frontier Workflow

LayerForge-X++ defines the repository's primary hybrid evaluation workflow:

1. generate multiple decomposition candidates;
2. preserve both generative and graph-ordered variants where needed;
3. compare native, peeling, raw Qwen, and hybrid rows on the same images;
4. run an explicit self-evaluation pass to select the best editable representation per image.

The objective is not arbitrary model accumulation. The objective is a stronger structured representation.

## Candidate bank

The frontier comparison bank currently covers:

- `LayerForge native`
- `LayerForge peeling`
- `Qwen raw (N)`
- `Qwen + graph preserve (N)`
- `Qwen + graph reorder (N)`

The preserve row keeps Qwen's interpreted visual stack and adds LayerForge graph metadata. The reorder row exports the same imported layers in graph order.

## Run the frontier bank

```bash
python scripts/run_frontier_comparison.py \
  --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png \
  --output-root runs/frontier_review \
  --native-config configs/frontier.yaml \
  --peeling-config configs/recursive_peeling.yaml \
  --qwen-layers 4 \
  --qwen-steps 20 \
  --qwen-resolution 640 \
  --qwen-device cuda \
  --qwen-dtype bfloat16 \
  --qwen-offload sequential \
  --skip-existing
```

Outputs:

- `frontier_summary.json`
- `frontier_summary.md`
- `editability_suite_summary.json` after `python scripts/run_editability_suite.py`
- per-image `best_decomposition.json`
- per-image `why_selected.md`

## Self-evaluation

The current self-evaluation stage is intentionally explicit. It scores each successful candidate per image using:

- recomposition fidelity;
- edit-preservation and anti-trivial copy penalties;
- semantic separation;
- alpha quality;
- graph confidence;
- runtime.

The score is not a claim of universal quality. It is a routing metric for selecting the most useful editable decomposition among the candidates the repo can already measure, and it now includes explicit penalties for stacks that reconstruct well only because the background layer copies too much of the source image.

Default weights live in `configs/frontier.yaml` under `self_eval.weights`.

## Promptable extraction

Promptable extraction now has a dedicated entrypoint:

```bash
.venv/bin/layerforge extract \
  --input data/demo/truck.jpg \
  --output runs/truck_extract \
  --config configs/frontier.yaml \
  --segmenter grounded_sam2 \
  --depth ensemble \
  --prompt "the truck in the foreground"
```

This writes a normal run and a `target_extract/` folder with the selected target RGBA, alpha, background-completed image, and edit previews.

The older general pipeline form still works:

```bash
.venv/bin/layerforge run \
  --input data/demo/truck.jpg \
  --output runs/truck_promptable \
  --config configs/frontier.yaml \
  --segmenter grounded_sam2 \
  --prompts "the truck, the road, the sky" \
  --prompt-source manual
```

For iterative foreground removal and residual completion, use:

```bash
.venv/bin/layerforge peel \
  --input data/demo/truck.jpg \
  --output runs/truck_peeling \
  --config configs/recursive_peeling.yaml \
  --segmenter grounded_sam2 \
  --prompt-source manual
```

## Measured current run

The current full measured five-image run lives at `runs/frontier_review/frontier_summary.json`.

Aggregate results:

| Method | Images | Mean PSNR | Mean SSIM | Mean self-eval score | Best-image wins |
|---|---:|---:|---:|---:|---:|
| LayerForge native | 5 | 37.6688 | 0.9708 | 0.6283 | 4 |
| LayerForge peeling | 5 | 27.0988 | 0.9096 | 0.4783 | 0 |
| Qwen raw (4) | 5 | 29.0757 | 0.8850 | 0.2541 | 0 |
| Qwen + graph preserve (4) | 5 | 28.5539 | 0.8638 | 0.5259 | 0 |
| Qwen + graph reorder (4) | 5 | 28.5397 | 0.8637 | 0.5251 | 1 |

Best-per-image selections:

- `truck`: `LayerForge native`
- `astronaut`: `LayerForge native`
- `coffee`: `LayerForge native`
- `chelsea_cat`: `Qwen + graph reorder (4)`
- `synth image`: `LayerForge native`

## Artifact interpretation

The frontier path is strongest when reported honestly:

- `Qwen raw` is the generative baseline.
- `Qwen + graph preserve` remains the fair metadata-first hybrid even when it is not the top-scoring row on the current five-image bank.
- `Qwen + graph reorder` shows what changes when the graph owns visual ordering.
- `LayerForge peeling` is the graph-guided recursive decomposition path.
- `best_decomposition.json` is the repo's self-selected editable representation, not a claim of state-of-the-art quality.

## Promptable extraction benchmark

The repository includes a measured prompt-conditioned extraction benchmark on synthetic LayerBench++ scenes:

```bash
python scripts/make_synthetic_dataset.py \
  --output data/layerbenchpp_prompt_benchmark \
  --count 10 \
  --output-format layerbench_pp \
  --with-effects

python scripts/run_extract_benchmark.py \
  --dataset-dir data/layerbenchpp_prompt_benchmark \
  --output-dir runs/extract_benchmark_prompted_grounded \
  --segmenter grounded_sam2 \
  --depth ensemble \
  --device cuda \
  --max-scenes 10
```

Measured summary from `runs/extract_benchmark_prompted_grounded/extract_benchmark_summary.json`:

| Prompt type | Queries | Target hit rate | Mean target IoU | Mean alpha MAE |
|---|---:|---:|---:|---:|
| text | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + point | 10 | 1.0000 | 0.3776 | 0.1503 |
| text + box | 10 | 1.0000 | 0.3776 | 0.1503 |
| point | 10 | 0.0000 | 0.8654 | 0.0222 |
| box | 10 | 0.0000 | 0.8654 | 0.0222 |

Interpretation:

- text-bearing prompts hit the intended semantic target on the measured synthetic set;
- point-only and box-only prompts currently lock onto a neighboring region with high overlap but wrong semantics;
- the benchmark is therefore useful precisely because it separates semantic hit rate from raw overlap and alpha quality.

## Transparent decomposition benchmark

The transparent path is now benchmarked separately on a small AlphaBlend-style synthetic set:

```bash
python scripts/make_transparent_dataset.py \
  --output data/transparent_benchmark \
  --count 12

python scripts/run_transparent_benchmark.py \
  --dataset-dir data/transparent_benchmark \
  --output-dir runs/transparent_benchmark
```

Measured summary from `runs/transparent_benchmark/transparent_benchmark_summary.json`:

| Metric | Mean |
|---|---:|
| Transparent alpha MAE | 0.1131 |
| Background PSNR | 25.9863 |
| Background SSIM | 0.9541 |
| Recompose PSNR | 56.0066 |
| Recompose SSIM | 0.9996 |

Transparent recomposition is a sanity check here; alpha error and clean-background quality are the primary transparent-layer metrics. This path should be framed as an approximate transparent-layer recovery mode, not a claim of state-of-the-art generative transparent decomposition.
