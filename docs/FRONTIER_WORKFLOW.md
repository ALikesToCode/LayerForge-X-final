<h1 align="center">Frontier Evaluation and Hybrid Workflow</h1>

The **Frontier Workflow** establishes a unified framework for orchestrating and evaluating diverse scene decomposition strategies. By comparing multiple candidate representations within a single image context, LayerForge-X ensures that the most robust and semantically coherent layer graph is selected for downstream tasks.

## Candidate Bank and Orchestration

The frontier evaluation bank manages a diverse set of decomposition methods:

- **LayerForge Native:** The core geometry-aware pipeline.
- **LayerForge Peeling:** Recursive residual decomposition for high-fidelity extraction.
- **Qwen Raw:** Direct RGBA export from the Qwen-Image-Layered generative baseline.
- **Qwen + Graph (Preserve):** Hybrid representation that augments Qwen layers with DALG metadata while maintaining the original visual stack.
- **Qwen + Graph (Reorder):** Hybrid representation where the imported generative stack is re-ordered based on the LayerForge depth graph.

The primary objective is to select the optimal **structured representation** rather than relying on a single, fixed model output.

## Execution and Evaluation

### Orchestration Command
The primary public entrypoint is the `layerforge frontier` command. Use it to execute the shared candidate bank across one or more images:

```bash
layerforge frontier \
  --inputs data/demo/truck.jpg data/qualitative_pack/astronaut.png \
  --output-root runs/frontier_review \
  --native-config configs/frontier.yaml \
  --peeling-config configs/recursive_peeling.yaml \
  --qwen-layers 4 \
  --qwen-steps 20 \
  --qwen-device cuda
```

The underlying script remains available for report regeneration and automation:

```bash
python scripts/run_frontier_comparison.py --inputs data/demo/truck.jpg --output-root runs/frontier_review
```

### Generated Artifacts
The workflow produces a complete frontier artifact set, including:
- `frontier_summary.json/md`: Aggregate performance metrics and candidate comparisons.
- `editability_suite_summary.json`: Quantitative scores for object removal, movement, and intrinsic edits.
- `best_decomposition.json`: The canonical DALG manifest for the selected winner.
- `why_selected.md`: A diagnostic report detailing the selection rationale.

## Automated Self-Evaluation Logic

The self-evaluation stage employs a multi-axial scoring mechanism to identify the most functional editable representation. The current criteria include:

1. **Recomposition Fidelity:** PSNR and SSIM of the reconstructed scene.
2. **Edit Stability:** Preservation of non-edited regions during manipulation.
3. **Anti-Triviality Check:** Penalties for representations that achieve high fidelity by simply copying source pixels into the background layer.
4. **Graph Confidence:** Topological consistency and depth-ordering certainty.
5. **Alpha Quality:** Boundary stability and matting precision.

Weights for these metrics are configurable via the `self_eval.weights` section in `configs/frontier.yaml`.

## Specialized Functional Modes

### 1. Prompt-Conditioned Target Extraction
For interactive, user-directed layer extraction, use the `extract` command:

```bash
.venv/bin/layerforge extract \
  --input data/demo/truck.jpg \
  --output runs/truck_extract \
  --prompt "the truck in the foreground"
```

### 2. Recursive Residual Peeling
For scenarios requiring iterative foreground removal and background completion:

```bash
.venv/bin/layerforge peel \
  --input data/demo/truck.jpg \
  --output runs/truck_peeling \
  --prompt-source manual
```

## Experimental Baseline and Benchmarking

The five-image frontier review establishes the performance baseline for the current repository state. Detailed aggregate results and per-image takeaways are documented in `report_artifacts/metrics_snapshots/frontier_review_summary.json`, confirming that while generative models provide strong visual priors, the integrated DALG approach offers superior structural coherence and editability.
