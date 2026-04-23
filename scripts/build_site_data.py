#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layerforge.site_data import write_project_site_payload  # noqa: E402


def main() -> int:
    output = ROOT / "docs" / "site-data" / "project_site.json"
    write_project_site_payload(ROOT, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
