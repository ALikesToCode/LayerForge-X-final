#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path

VARIANTS = [
    ("A_classical_luminance", "classical", "geometric_luminance"),
    ("B_mask2former_luminance", "mask2former", "geometric_luminance"),
    ("C_mask2former_depth_anything", "mask2former", "depth_anything_v2"),
    ("D_grounded_sam2_depthpro", "grounded_sam2", "depth_pro"),
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--output-root", default="runs/ablations")
    args = ap.parse_args()
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, seg, dep in VARIANTS:
        out = root / name
        cmd = ["layerforge", "run", "--input", args.input, "--output", str(out), "--config", args.config, "--segmenter", seg, "--depth", dep]
        try:
            subprocess.run(cmd, check=True)
            rows.append({"variant": name, **json.load(open(out / "metrics.json"))})
        except Exception as exc:
            rows.append({"variant": name, "error": str(exc)})
    json.dump(rows, open(root / "ablation_summary.json", "w"), indent=2)
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
