# LayerForge-X++ Frontier Workflow

LayerForge-X++ is the repo's highest-leverage hybrid path:

1. generate multiple decomposition candidates;
2. preserve both generative and graph-ordered variants where needed;
3. compare native, peeling, raw Qwen, and hybrid rows on the same images;
4. run a heuristic self-evaluation pass to select the best editable representation per image.

The goal is not "more models". The goal is a better structured representation.

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
  --qwen-layers 4,6 \
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
- per-image `best_decomposition.json`
- per-image `why_selected.md`

## Self-evaluation

The current self-evaluation stage is intentionally described as heuristic. It scores each successful candidate per image using:

- recomposition fidelity;
- explicit structure availability;
- editability bias toward usable layer counts;
- runtime.

The score is not a claim of universal quality. It is a routing heuristic for selecting the most useful editable decomposition among the candidates the repo can already measure.

Default weights live in `configs/frontier.yaml` under `self_eval.weights`.

## Promptable extraction

Promptable extraction already works through the existing pipeline entrypoints:

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
  --prompt-source gemini
```

## Artifact interpretation

The frontier path is strongest when reported honestly:

- `Qwen raw` is the generative baseline.
- `Qwen + graph preserve` is the fair metadata-first hybrid.
- `Qwen + graph reorder` shows what changes when the graph owns visual ordering.
- `LayerForge peeling` is the graph-guided recursive decomposition path.
- `best_decomposition.json` is the repo's self-selected editable representation, not a claim of state-of-the-art quality.
