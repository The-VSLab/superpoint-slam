import argparse
import sys
from pathlib import Path
import glob
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# SuperPoint 모델 임포트 (경로에 맞게 수정)
try:
    from models.superpoint_mobilenet import SuperPointNetV2
except ImportError:
    from models.superpoint_mobilenet import SuperPointNetV2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. 정답지(.npz)를 읽는 데이터셋 클래스 ---
class SyntheticDataset(Dataset):
    def __init__(self, root, height=480, width=640):
        self.root = root
        self.height = height
        self.width = width
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. 이미지 로드
        img_path = self.files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        # 2. 정답(.npz) 로드
        label_path = img_path.replace(".jpg", ".npz")
        try:
            points = np.load(label_path)['points'] # [Row, Col] format
        except:
            points = np.zeros((0, 2))

        # 3. 좌표를 히트맵(Label)으로 변환
        # SuperPoint는 8x8 그리드마다 하나의 채널을 가집니다 (총 65채널: 64개 셀 + 1개 dustbin)
        heatmap = self.labels2Dto3D(points, self.height, self.width)

        return img_tensor, heatmap

    def labels2Dto3D(self, labels, H, W, cell_size=8):
        # Hc x Wc 크기의 그리드 생성
        Hc, Wc = H // cell_size, W // cell_size
        # 65번째 채널(dustbin, 배경)로 초기화
        target = np.zeros((H, W), dtype=np.uint8)
        
        # 라벨이 있는 곳만 1로 표시 (Sparse)
        for pt in labels:
            r, c = int(pt[0]), int(pt[1])
            if 0 <= r < H and 0 <= c < W:
                target[r, c] = 1

        target = torch.from_numpy(target).long()
        
        # 8x8 블록으로 픽셀을 모음 -> (Hc, Wc, 64)가 됨
        # 공간을 채널로 바꾸는 과정 (Space to Depth)
        target = target.view(Hc, cell_size, Wc, cell_size)
        target = target.permute(0, 2, 1, 3).contiguous()
        target = target.view(Hc, Wc, cell_size * cell_size)
        target = target.permute(2, 0, 1) # (64, Hc, Wc)

        # 정답이 있는 곳의 채널 인덱스 찾기
        target_map = torch.argmax(target, dim=0) # (Hc, Wc) 값이 0~63
        
        # 정답이 없는 곳(모두 0)은 65번째(인덱스 64) 'Dustbin' 클래스로 설정
        # sum이 0이면 포인트가 없는 것
        valid_mask = torch.sum(target, dim=0) > 0
        labels_final = torch.full((Hc, Wc), 64, dtype=torch.long) # Default to Dustbin (64)
        labels_final[valid_mask] = target_map[valid_mask]
        
        return labels_final


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 모델 준비
    model = SuperPointNetV2().to(device)
    if args.weights_in:
        if os.path.isfile(args.weights_in):
            state = torch.load(args.weights_in, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"Loaded weights: {args.weights_in}")
        else:
            raise FileNotFoundError(f"weights_in not found: {args.weights_in}")
    
    # 데이터 로더 준비
    dataset = SyntheticDataset(args.data_dir, args.height, args.width)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss Function: CrossEntropy (정답지와 내 예측 비교)
    criterion = torch.nn.CrossEntropyLoss()

    print("==> Starting Supervised Training (Brain Waking)...")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device) # (B, Hc, Wc) 값은 0~64 사이 정수

            optimizer.zero_grad()
            
            # Forward
            semi, _ = model(imgs) # semi: (B, 65, Hc, Wc)
            
            # Loss 계산 (Supervised)
            loss = criterion(semi, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.5f}"})

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(loader):.5f}")

    # 저장
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.weights_out)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights_in", type=str, default="")
    parser.add_argument("--weights_out", type=str, default="superpoint_base_synthetic.pth")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16) # 도형은 가벼워서 배치 키워도 됨
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train(args)
