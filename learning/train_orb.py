import sys
import os
import argparse
import glob
import random
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import cv2

# [1] 경로 설정
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CUR_DIR)

# SuperPoint 모델 임포트
from scripts.py_superpoint import SuperPointNetV2

# --- [2] 속도 최적화된 데이터셋 (ORB를 여기서 미리 계산) ---

class ORBDataset(Dataset):
    def __init__(self, root, size, max_rotate=15.0, max_scale=0.1, n_features=1000):
        self.paths = []
        for e in ["jpg", "jpeg", "png"]:
            self.paths.extend(glob.glob(os.path.join(root, "**", f"*.{e}"), recursive=True))
        self.paths = sorted(list(set(self.paths)))
        self.h, self.w = size
        self.max_rotate = max_rotate
        self.max_scale = max_scale
        self.n_features = n_features
        # self.orb = cv2.ORB_create(...) <- 여기서 지워야 합니다!
        print(f"✅ 데이터셋: {len(self.paths)}개 | 멀티프로세싱 준비 완료")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        # [수정] 각 프로세스 워커가 처음 실행될 때 ORB 객체를 생성하도록 함
        if not hasattr(self, 'orb'):
            self.orb = cv2.ORB_create(nfeatures=self.n_features)

        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None: return self.__getitem__((idx + 1) % len(self.paths))

        # 1. Augmentation (생략 가능하면 속도는 더 빨라집니다)
        if self.max_rotate > 0:
            angle = random.uniform(-self.max_rotate, self.max_rotate)
            scale = 1.0 + random.uniform(-self.max_scale, self.max_scale)
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, scale)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT101)
        
        img = cv2.resize(img, (self.w, self.h))
        
        # 2. ORB 라벨 생성
        kp = self.orb.detect(img, None)
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        for k in kp:
            x, y = map(int, k.pt)
            if 0 <= x < self.w and 0 <= y < self.h:
                mask[y, x] = 1.0
        
        # 3. SuperPoint 65채널 라벨 변환
        patches = mask.reshape(self.h//8, 8, self.w//8, 8).transpose(0, 2, 1, 3).reshape(self.h//8, self.w//8, 64)
        labels = np.argmax(patches, axis=2)
        labels[np.max(patches, axis=2) == 0] = 64 
        
        img_tensor = torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0).repeat(3, 1, 1)
        return img_tensor, torch.from_numpy(labels).long(), torch.from_numpy(mask)
# --- [3] 유틸리티 함수 ---

def save_debug_trace(img, teacher_mask, prediction, path):
    orig = (img[0, 0].cpu().numpy() * 255).astype(np.uint8)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    
    # 학생 (초록색)
    y_p, x_p = np.where(prediction[0].detach().cpu().numpy() > 0)
    for r, c in zip(y_p, x_p):
        cv2.circle(orig_rgb, (c, r), 2, (0, 255, 0), -1)
    # 선생 (빨간색 십자)
    y_t, x_t = np.where(teacher_mask[0].cpu().numpy() > 0)
    for r, c in zip(y_t, x_t):
        cv2.drawMarker(orig_rgb, (c, r), (0, 0, 255), cv2.MARKER_CROSS, 5, 1)
    
    cv2.imwrite(path, orig_rgb)

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

# --- [4] 메인 학습 루프 ---

def train(args):
    # GPU 성능 극대화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    student = SuperPointNetV2().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(args.lr))
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    ce_weight = torch.ones(65, device=device)
    ce_weight[64] = args.dustbin_weight 

    # DataLoader 최적화: num_workers를 높게 잡으세요
    dataset = ORBDataset(os.path.join(ROOT_DIR, args.data_dir), (args.height, args.width),
                         args.max_rotate, args.max_scale, n_features=args.top_k)
    
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2
    )

    save_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 학습 가속 모드 시작 (Step당 {args.batch_size}장 처리)")
    
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Ep {epoch+1}", mininterval=1.0)
        student.train()
        
        for img, labels, teacher_mask in pbar:
            # 비동기 데이터 전송
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=args.fp16):
                semi, _ = student(img)
                loss = F.cross_entropy(semi, labels, weight=ce_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            
            # 100 스텝마다 디버그 이미지 저장
            if global_step % 100 == 0:
                with torch.no_grad():
                    # 히트맵 복원
                    heatmap = F.softmax(semi, dim=1)[:, :-1].view(-1, 8, 8, args.height//8, args.width//8).permute(0, 3, 1, 4, 2).reshape(-1, args.height, args.width)
                    top_k_map = select_top_k_heatmap(heatmap * (heatmap > args.detection_threshold), k=args.top_k)
                    
                    # [수정] 파일명을 'debug.jpg'로 고정하여 덮어쓰기
                    current_dir = os.getcwd() 
                    debug_filename = "debug.jpg"
                    debug_path = os.path.join(current_dir, debug_filename)
                    
                    save_debug_trace(img, teacher_mask, top_k_map, debug_path)

        # 에포크 저장
        torch.save(student.state_dict(), os.path.join(save_dir, f"orb_student_latest.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    
    # ROOT_DIR와 결합하여 설정 파일 로드
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT_DIR, config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        for k, v in config.items(): 
            setattr(args, k, v)
    
    # [핵심] 누락될 수 있는 필수 파라미터 기본값 강제 설정
    defaults = [
        ('top_k', 1000), 
        ('detection_threshold', 0.1), # 에러 발생 원인 해결
        ('debug_every', 100), 
        ('max_rotate', 15.0), 
        ('max_scale', 0.1), 
        ('max_perspective', 0.0),
        ('output_dir', 'checkpoints'),
        ('lr', 1.0e-4),
        ('epochs', 20),
        ('batch_size', 16),
        ('num_workers', 8),
        ('fp16', True),
        ('dustbin_weight', 0.05)
    ]
    
    for k, v in defaults:
        if not hasattr(args, k): 
            setattr(args, k, v)

    train(args)