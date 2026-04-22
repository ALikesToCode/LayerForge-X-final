#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


ADE20K_URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"using cached archive: {path}")
        return
    print(f"downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)


def extract_zip(archive_path: Path, output_dir: Path) -> Path:
    dataset_root = output_dir / "ADEChallengeData2016"
    if dataset_root.exists():
        print(f"using existing extraction: {dataset_root}")
        return dataset_root
    print(f"extracting {archive_path} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(output_dir)
    return dataset_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract the official ADE20K SceneParse150 zip")
    parser.add_argument("--output-dir", default="data/ade20k", help="Directory where ADEChallengeData2016 will be extracted")
    parser.add_argument("--archive-dir", default="data/downloads/ade20k", help="Directory where the ADE zip archive will be cached")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    archive_dir = Path(args.archive_dir)
    archive_path = archive_dir / "ADEChallengeData2016.zip"
    download_file(ADE20K_URL, archive_path)
    dataset_root = extract_zip(archive_path, output_dir)
    print(f"dataset root: {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
