import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.py_superpoint import SuperPointNetV2


def select_top_k_heatmap(heatmap, k=600):
    b, h, w = heatmap.shape
    top_k_list = []
    for i in range(b):
        flat = heatmap[i].view(-1)
        k_val = min(k, flat.size(0))
        vals, indices = torch.topk(flat, k_val)
        mask = torch.zeros_like(flat)
        mask[indices] = vals
        top_k_list.append(mask.view(h, w))
    return torch.stack(top_k_list)


def load_images(root_dir):
    exts = ["jpg", "jpeg", "png", "JPG", "PNG"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(root_dir, "**", f"*.{e}"), recursive=True))
    return sorted(list(set(paths)))


def main():
    parser = argparse.ArgumentParser(description="SuperPoint COCO validation (simple stats + visualization)")
    parser.add_argument("--data_dir", type=str, required=True, help="COCO val2017 directory")
    parser.add_argument("--weights", type=str, required=True, help="Student weights (.pth)")
    parser.add_argument("--output_dir", type=str, default="results_val_coco", help="Output directory")
    parser.add_argument("--max_images", type=int, default=200, help="Max images to sample")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--top_k", type=int, default=600)
    parser.add_argument("--detection_threshold", type=float, default=0.2)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.join(ROOT_DIR, args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    paths = load_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    random.seed(args.seed)
    if args.max_images > 0:
        paths = random.sample(paths, min(args.max_images, len(paths)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperPointNetV2().to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    total_kpts = 0.0
    processed = 0

    for idx, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.repeat(1, 3, 1, 1).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=args.fp16):
                semi, _ = model(img_tensor)
            heatmap = F.softmax(semi, dim=1)[:, :-1]
            heatmap = heatmap.view(-1, 8, 8, args.height // 8, args.width // 8)
            heatmap = heatmap.permute(0, 3, 1, 4, 2).reshape(-1, args.height, args.width)
            top_k_map = select_top_k_heatmap(
                heatmap * (heatmap > args.detection_threshold),
                k=args.top_k,
            )
            pts = (top_k_map > 0)[0].cpu().numpy()

        ys, xs = np.where(pts)
        total_kpts += float(len(xs))
        processed += 1

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y in zip(xs, ys):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

        out_name = f"val_{idx:05d}.png"
        cv2.imwrite(os.path.join(vis_dir, out_name), vis)

    avg_kpts = total_kpts / max(1, processed)
    summary = {
        "images": processed,
        "avg_kpts": avg_kpts,
        "height": args.height,
        "width": args.width,
        "top_k": args.top_k,
        "detection_threshold": args.detection_threshold,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Processed: {processed} images")
    print(f"avg_kpts: {avg_kpts:.1f}")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
