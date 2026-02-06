import argparse
import glob
import math
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from models.superpoint_mobilenet import SuperPointNetV2
except ImportError:
    from models.superpoint_mobilenet import SuperPointNetV2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(root: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(paths)


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: Tuple[int, int]) -> None:
        self.paths = list_images(root)
        if not self.paths:
            raise ValueError(f"No images found under: {root}")
        self.height, self.width = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)


def sample_homography(
    height: int,
    width: int,
    max_translate: float,
    max_rotate: float,
    max_scale: float,
    max_perspective: float,
) -> np.ndarray:
    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    cx, cy = width / 2.0, height / 2.0
    angle = np.deg2rad(random.uniform(-max_rotate, max_rotate))
    scale = random.uniform(1.0 - max_scale, 1.0 + max_scale)
    tx = random.uniform(-max_translate, max_translate) * width
    ty = random.uniform(-max_translate, max_translate) * height

    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    dst = src - np.array([[cx, cy]], dtype=np.float32)
    dst = dst @ rot.T * scale
    dst = dst + np.array([[cx + tx, cy + ty]], dtype=np.float32)

    persp = np.random.uniform(-max_perspective, max_perspective, size=(4, 2))
    dst = dst + persp.astype(np.float32) * np.array([[width, height]], dtype=np.float32)

    h_mat = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return h_mat


def homography_grid(h_mat: torch.Tensor, height: int, width: int) -> torch.Tensor:
    device = h_mat.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    coords = torch.stack([x, y, ones], dim=-1).float()  # H x W x 3
    coords = coords.view(-1, 3).t()  # 3 x (H*W)

    h_inv = torch.linalg.inv(h_mat)
    warped = h_inv @ coords  # 3 x (H*W)
    warped = warped / (warped[2:3, :] + 1e-8)

    x_w = warped[0, :].view(height, width)
    y_w = warped[1, :].view(height, width)

    x_norm = (x_w / (width - 1)) * 2 - 1
    y_norm = (y_w / (height - 1)) * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return grid


def warp_tensor(
    tensor: torch.Tensor,
    h_mat: torch.Tensor,
    mode: str = "bilinear",
    use_cpu: bool = None,
) -> torch.Tensor:
    b, c, h, w = tensor.shape
    
    # [Cross-Platform Fix]
    if use_cpu is None:
        use_cpu = (tensor.device.type == "mps")

    warped = []
    for i in range(b):
        grid = homography_grid(h_mat[i], h, w)
        grid = grid.unsqueeze(0)
        
        if use_cpu:
            tensor_i = tensor[i:i + 1].cpu()
            grid = grid.cpu()
            warped_i = F.grid_sample(tensor_i, grid, mode=mode, align_corners=True)
            warped.append(warped_i.to(tensor.device))
        else:
            warped.append(F.grid_sample(tensor[i:i + 1], grid, mode=mode, align_corners=True))
            
    return torch.cat(warped, dim=0)


def semi_to_heatmap(semi: torch.Tensor, cell: int = 8) -> torch.Tensor:
    prob = F.softmax(semi, dim=1)
    nodust = prob[:, :-1, :, :]
    b, c, h, w = nodust.shape  # c=64
    nodust = nodust.view(b, cell, cell, h, w)
    nodust = nodust.permute(0, 3, 1, 4, 2)
    heatmap = nodust.reshape(b, h * cell, w * cell)
    return heatmap


def normalize_desc(desc: torch.Tensor) -> torch.Tensor:
    return F.normalize(desc, p=2, dim=1)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = SuperPointNetV2().to(device)

    # [가중치 로드 부분]
    if args.weights_in:
        print(f"Loading weights from {args.weights_in}...")
        if os.path.isfile(args.weights_in):
            state = torch.load(args.weights_in, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            
            msg = model.load_state_dict(state, strict=False)
            print(f"Loaded weights with msg: {msg}")
        else:
            print(f"⚠️ Warning: Weights file not found at {args.weights_in}")

    dataset = ImageFolderDataset(args.data_dir, (args.height, args.width))
    
    use_pin_memory = (device.type == "cuda")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # AMP는 CUDA에서만 활성화
    use_amp = args.fp16 and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if args.fp16 and not use_amp:
        print("Notice: --fp16 requested but running on non-CUDA device. FP16 disabled.")

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch in pbar:
            img = batch.to(device)

            # [긴급 점검 코드] 첫 번째 배치의 이미지를 저장해서 눈으로 확인
            if epoch == 0 and pbar.n == 0: # 첫 에폭, 첫 배치일 때만
                debug_img = (img[0, 0].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite("debug_input.jpg", debug_img)
                print("📸 Debug image saved to debug_input.jpg Check it NOW!")

            b, _, h, w = img.shape

            h_mats = []
            for _ in range(b):
                h_mat = sample_homography(
                    h,
                    w,
                    args.max_translate,
                    args.max_rotate,
                    args.max_scale,
                    args.max_perspective,
                )
                h_mats.append(torch.from_numpy(h_mat).float())
            h_mats = torch.stack(h_mats).to(device)

            # [Cross-Platform] Mac/Windows 자동 처리
            img_warp = warp_tensor(img, h_mats, mode="bilinear")

            optimizer.zero_grad(set_to_none=True)
            
            # Autocast context (Mac에서는 자동 비활성)
            with torch.cuda.amp.autocast(enabled=use_amp):
                semi, desc = model(img)
                semi_w, desc_w = model(img_warp)

                heatmap = semi_to_heatmap(semi)
                heatmap_w = semi_to_heatmap(semi_w)

                heatmap_w_from_orig = warp_tensor(
                    heatmap.unsqueeze(1),
                    h_mats,
                    mode="bilinear",
                ).squeeze(1)

                det_loss = F.mse_loss(heatmap_w_from_orig, heatmap_w)

                desc = normalize_desc(desc)
                desc_w = normalize_desc(desc_w)

                desc_w_from_orig = warp_tensor(
                    desc, h_mats, mode="bilinear"
                )
                
                heatmap_w_down = F.avg_pool2d(heatmap_w.unsqueeze(1), 8).squeeze(1)
                weight = heatmap_w_down.clamp(min=0.0, max=1.0).unsqueeze(1)

                # [Mode Collapse 방지]
                safe_weight = weight + 0.1 
                
                cos_sim = (desc_w_from_orig * desc_w).sum(dim=1, keepdim=True)
                desc_loss_term = (1.0 - cos_sim) * safe_weight
                desc_loss = desc_loss_term.sum() / (safe_weight.sum() + 1e-6)

                loss = args.det_weight * det_loss + args.desc_weight * desc_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_every == 0:
            out_path = os.path.join(args.output_dir, f"superpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), out_path)

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, args.weights_out)
    torch.save(model.state_dict(), final_path)
    print(f"Saved weights: {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-supervised SuperPoint finetuning (Cross-Platform)"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights_in", type=str, default=None)
    parser.add_argument("--weights_out", type=str, default="superpoint_v2_mobilenet_ft.pth")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="auto picks cuda > mps > cpu",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_translate", type=float, default=0.1)
    parser.add_argument("--max_rotate", type=float, default=10.0)
    parser.add_argument("--max_scale", type=float, default=0.1)
    parser.add_argument("--max_perspective", type=float, default=0.02)

    parser.add_argument("--det_weight", type=float, default=1.0)
    parser.add_argument("--desc_weight", type=float, default=0.5)
    parser.add_argument("--save_every", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)