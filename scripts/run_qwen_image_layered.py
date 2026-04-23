#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from layerforge.qwen_io import score_raw_rgba_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the official Qwen-Image-Layered pipeline and export RGBA layers.")
    p.add_argument("--input", required=True, help="Input RGB/RGBA image")
    p.add_argument("--output-dir", required=True, help="Directory for exported RGBA layers")
    p.add_argument("--model", default="Qwen/Qwen-Image-Layered", help="Hugging Face model id")
    p.add_argument("--layers", type=int, default=4, help="Requested number of output layers")
    p.add_argument("--resolution", type=int, default=640, help="Bucketed inference resolution; official recommendation is 640")
    p.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    p.add_argument("--seed", type=int, default=777, help="Random seed")
    p.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen layered true CFG scale")
    p.add_argument("--negative-prompt", default=" ", help="Negative prompt")
    p.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Torch dtype for the pipeline")
    p.add_argument("--offload", default="none", choices=["none", "model", "sequential"], help="Enable CPU offload instead of moving the full pipeline onto the target device")
    p.add_argument("--caption-language", default="auto", choices=["auto", "en"], help="Force English auto-captioning or leave automatic selection")
    return p.parse_args()


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def main() -> int:
    args = parse_args()
    import torch

    from diffusers import QwenImageLayeredPipeline

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.input).convert("RGBA")

    pipe = QwenImageLayeredPipeline.from_pretrained(args.model, torch_dtype=torch_dtype(args.dtype))
    if args.offload == "model":
        pipe.enable_model_cpu_offload(device=args.device)
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=args.device)
    else:
        pipe = pipe.to(args.device, torch_dtype=torch_dtype(args.dtype))
    pipe.set_progress_bar_config(disable=None)

    inputs = {
        "image": image,
        "generator": torch.Generator(device=args.device).manual_seed(args.seed),
        "true_cfg_scale": float(args.true_cfg_scale),
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": int(args.steps),
        "num_images_per_prompt": 1,
        "layers": int(args.layers),
        "resolution": int(args.resolution),
        "cfg_normalize": True,
        "use_en_prompt": args.caption_language == "en" or args.caption_language == "auto",
    }

    with torch.inference_mode():
        output = pipe(**inputs)

    layer_images = output.images[0]
    layer_paths: list[str] = []
    for i, layer_image in enumerate(layer_images):
        path = out_dir / f"{i:02d}.png"
        layer_image.save(path)
        layer_paths.append(str(path))

    manifest = {
        "input": str(Path(args.input)),
        "output_dir": str(out_dir),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "offload": args.offload,
        "seed": args.seed,
        "layers_requested": args.layers,
        "resolution": args.resolution,
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "layer_paths": layer_paths,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics_path, recomposed_path = score_raw_rgba_layers(args.input, out_dir)
    print(
        json.dumps(
            {
                "manifest": str(out_dir / "manifest.json"),
                "metrics": str(metrics_path),
                "recomposed_rgb": str(recomposed_path),
                "num_layers": len(layer_paths),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
