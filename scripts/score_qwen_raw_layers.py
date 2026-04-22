#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from layerforge.qwen_io import score_raw_rgba_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score an existing Qwen raw RGBA export by recomposing the manifest-ordered layers.")
    p.add_argument("--input", required=True, help="Original input image used for the Qwen export")
    p.add_argument("--layers-dir", required=True, help="Directory containing Qwen RGBA layers and manifest.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path, recomposed_path = score_raw_rgba_layers(args.input, args.layers_dir)
    payload = {
        "metrics": str(Path(metrics_path)),
        "recomposed_rgb": str(Path(recomposed_path)),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
