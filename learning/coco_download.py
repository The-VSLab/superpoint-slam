"""COCO 2017 downloader (train/val images) with parallel download + extract.

Usage:
  python scripts/coco_download.py --out_dir dataset/coco2017 --split train val
"""
from __future__ import annotations

import argparse
import os
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

COCO_URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
}


def _download(url: str, dst_path: str, chunk_size: int = 8 * 1024 * 1024) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".part"
    lock = threading.Lock()
    
    with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as f:
        total = resp.length or 0
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100.0 / total
                with lock:
                    print(f"[download] {os.path.basename(dst_path)} {pct:6.2f}%", end="\r")
    os.replace(tmp_path, dst_path)
    print(f"[download] 완료: {dst_path}")


def _extract(zip_path: str, out_dir: str) -> None:
    print(f"[extract] 시작: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"[extract] 완료: {out_dir}")


def download_and_extract(split: str, out_dir: str) -> None:
    if split not in COCO_URLS:
        raise ValueError(f"Unknown split: {split}")
    url = COCO_URLS[split]
    zip_name = os.path.basename(url)
    zip_path = os.path.join(out_dir, zip_name)

    if not os.path.exists(zip_path):
        _download(url, zip_path)
    else:
        print(f"[download] 이미 존재: {zip_path}")

    extract_dir = out_dir
    folder = os.path.join(out_dir, f"{split}2017")
    if not os.path.exists(folder):
        _extract(zip_path, extract_dir)
    else:
        print(f"[extract] 이미 존재: {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO 2017 downloader")
    parser.add_argument("--out_dir", type=str, default="dataset/coco2017")
    parser.add_argument("--split", nargs="+", default=["train", "val"], choices=["train", "val"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(len(args.split), 4)) as ex:
        futures = [ex.submit(download_and_extract, s, args.out_dir) for s in args.split]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
