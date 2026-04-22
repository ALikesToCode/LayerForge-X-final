#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


DIODE_VAL_URL = "https://diode-dataset.s3.amazonaws.com/val.tar.gz"
DIODE_VAL_MD5 = "5c895d09201b88973c8fe4552a67dd85"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"using cached archive: {path}")
        return
    print(f"downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    split_root = output_dir / "val"
    if (split_root / "indoors").exists() or (split_root / "outdoor").exists():
        print(f"using existing extraction: {split_root}")
        return split_root
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive_path} -> {output_dir}")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(path=output_dir, filter="data")
    return split_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract the official DIODE validation depth split")
    parser.add_argument("--output-dir", default="data/diode", help="Directory where the DIODE val split will be extracted")
    parser.add_argument("--archive-dir", default="data/downloads/diode", help="Directory where the DIODE tarball will be cached")
    parser.add_argument("--skip-md5", action="store_true", help="Skip MD5 verification after download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    archive_dir = Path(args.archive_dir)
    archive_path = archive_dir / "val.tar.gz"
    download_file(DIODE_VAL_URL, archive_path)
    if not args.skip_md5:
        digest = md5_file(archive_path)
        if digest != DIODE_VAL_MD5:
            raise RuntimeError(f"MD5 mismatch for {archive_path}: expected {DIODE_VAL_MD5}, got {digest}")
        print(f"verified md5: {digest}")
    split_root = extract_archive(archive_path, output_dir)
    print(f"dataset root: {split_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
