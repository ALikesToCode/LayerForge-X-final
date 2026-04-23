.PHONY: install install-models smoke benchmark ablate qwen-enrich clean

install:
	python -m pip install -e .

install-models:
	python -m pip install -r requirements-models.txt

smoke:
	python scripts/make_synthetic_dataset.py --output examples/synth --count 3
	layerforge run --input examples/synth/scene_000/image.png --output runs/smoke --config configs/fast.yaml --segmenter classical --depth geometric_luminance --no-parallax

benchmark:
	python scripts/make_synthetic_dataset.py --output data/synthetic_layerbench --count 10
	layerforge benchmark --dataset-dir data/synthetic_layerbench --output-dir results/synthetic_fast --config configs/fast.yaml --segmenter classical --depth geometric_luminance

ablate:
	python scripts/run_grid.py --input examples/synth/scene_000/image.png --config configs/fast.yaml --output-root runs/ablations

qwen-enrich:
	layerforge enrich-qwen --input examples/synth/scene_000/image.png --layers-dir examples/synth/scene_000/gt_layers --output runs/qwen_like_enriched --config configs/fast.yaml --depth geometric_luminance

clean:
	rm -rf runs results data .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
