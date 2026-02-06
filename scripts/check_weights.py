import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.superpoint_frontend import SuperPointFrontend


def parse_args():
    parser = argparse.ArgumentParser(description="SuperPoint 가중치 간단 검증")
    parser.add_argument(
        "--weights",
        required=True,
        help="가중치 파일 경로 (.pth)",
    )
    parser.add_argument(
        "--image",
        default="debug_input.jpg",
        help="테스트 이미지 경로 (기본: debug_input.jpg)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="장치 선택 (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.015,
        help="검출 임계값 (기본: 0.015)",
    )
    return parser.parse_args()


def resolve_device(device_opt):
    if device_opt != "auto":
        return device_opt
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = resolve_device(args.device)
    use_cuda = device == "cuda"

    print(f"[Check] Weights: {args.weights}")
    print(f"[Check] Image: {args.image}")
    print(f"[Check] Device: {device}")

    fe = SuperPointFrontend(weights_path=args.weights, conf_thresh=args.conf_thresh, cuda=use_cuda)

    img = cv2.imread(args.image, 0)
    if img is None:
        raise RuntimeError("Failed to read input image")

    h, w = img.shape[:2]
    H = h - (h % 8)
    W = w - (w % 8)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0)

    pts, desc, heatmap = fe.run(img)

    print(f"[Result] pts shape: {pts.shape}")
    print(f"[Result] desc shape: {None if desc is None else desc.shape}")
    print(f"[Result] heatmap shape: {None if heatmap is None else heatmap.shape}")
    print(f"[Result] num keypoints: {pts.shape[1]}")
    if desc is not None and desc.size > 0:
        desc_min = float(np.min(desc))
        desc_max = float(np.max(desc))
        desc_mean = float(np.mean(desc))
        print(f"[Result] desc stats: min={desc_min:.4f}, max={desc_max:.4f}, mean={desc_mean:.4f}")


if __name__ == "__main__":
    main()
