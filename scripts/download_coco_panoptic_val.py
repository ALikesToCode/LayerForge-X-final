#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_PANOPTIC_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"


def download_file(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"skip existing archive: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as f:
        shutil.copyfileobj(response, f)
    print(f"downloaded: {target}")


def extract_members(zip_path: Path, output_dir: Path, prefixes: tuple[str, ...], exact_names: tuple[str, ...] = ()) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if any(name.startswith(prefix) for prefix in prefixes) or name in exact_names]
        for member in members:
            target = output_dir / member
            if target.exists():
                continue
            zf.extract(member, path=output_dir)
    print(f"extracted {len(members)} entries from {zip_path.name}")


def extract_nested_zip_member(zip_path: Path, member_name: str, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as outer:
        data = outer.read(member_name)
    with zipfile.ZipFile(io.BytesIO(data)) as inner:
        inner.extractall(path=output_dir / "annotations")
    print(f"extracted nested archive {member_name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the official COCO val2017 images and panoptic val2017 annotations")
    parser.add_argument("--output-dir", default="data/coco_panoptic_val")
    parser.add_argument("--archive-dir", default="data/downloads/coco")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    archive_dir = Path(args.archive_dir)
    val_zip = archive_dir / "val2017.zip"
    panoptic_zip = archive_dir / "panoptic_annotations_trainval2017.zip"

    download_file(COCO_VAL_URL, val_zip)
    download_file(COCO_PANOPTIC_URL, panoptic_zip)

    extract_members(val_zip, output_dir, prefixes=("val2017/",))
    extract_members(
        panoptic_zip,
        output_dir,
        prefixes=(),
        exact_names=("annotations/panoptic_val2017.json",),
    )
    nested_target = output_dir / "annotations" / "panoptic_val2017"
    if not nested_target.exists():
        extract_nested_zip_member(panoptic_zip, "annotations/panoptic_val2017.zip", output_dir)
    print(f"dataset ready under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
