import sys
import os
import argparse
import glob
import math
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import cv2

# [1] 경로 자동 보정: 실행 위치에 상관없이 루트 및 모듈 폴더 인식
CUR_DIR = os.path.dirname(os.path.abspath(__file__)) # .../training
ROOT_DIR = os.path.dirname(CUR_DIR)                  # .../superpoint-slam
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")

for path in [ROOT_DIR, SCRIPTS_DIR, CUR_DIR]:
    if path not in sys.path:
        sys.path.append(path)

# [2] 모델 임포트
try:
    from models.superpoint_mobilenet import SuperPointNetV2  # Student (MobileNet)
    from learning.original_superpoint import SuperPoint # Teacher (Original VGG)
except ImportError:
    from models.superpoint_mobilenet import SuperPointNetV2
    from training.original_superpoint import SuperPoint

# --- [3] 핵심 유틸리티 함수 ---

def select_top_k_heatmap(heatmap, k=600):
    """히트맵에서 점수가 높은 상위 K개의 특징점만 선별"""
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

def get_teacher_labels(teacher, img_tensor):
    """선생님의 출력을 65채널 라벨로 변환하여 지식 증류 준비"""
    teacher.eval()
    with torch.no_grad():
        outputs = teacher({"image": img_tensor[:, 0:1, :, :]})
        b, _, h, w = img_tensor.shape
        mask = torch.zeros((b, h, w), device=img_tensor.device)
        for i in range(b):
            kps = outputs["keypoints"][i].long()
            if len(kps) > 0:
                mask[i, torch.clamp(kps[:, 1], 0, h-1), torch.clamp(kps[:, 0], 0, w-1)] = 1.0
        
        # 8x8 격자로 압축
        patches = mask.view(b, h//8, 8, w//8, 8).permute(0, 1, 3, 2, 4).reshape(b, h//8, w//8, 64)
        labels = torch.argmax(patches, dim=3)
        labels[torch.max(patches, dim=3)[0] == 0] = 64 # 배경
        teacher_desc = outputs.get("desc") if isinstance(outputs, dict) else None
    return labels, mask, teacher_desc

def save_debug_trace(img, teacher_mask, prediction, path):
    """원본 사이트 스타일의 녹색 점 시각화"""
    orig = (img[0, 0].cpu().numpy() * 255).astype(np.uint8)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    
    # 학생의 예측점 (녹색)
    pred_pts = prediction[0].detach().cpu().numpy()
    y, x = np.where(pred_pts > 0)
    for r, c in zip(y, x):
        cv2.circle(orig_rgb, (c, r), 2, (0, 255, 0), -1)
        
    # 선생님의 정답지 (흰색)
    gt_mask = (teacher_mask[0].cpu().numpy() * 255).astype(np.uint8)
    gt_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(path, np.hstack([orig_rgb, gt_rgb]))


def evaluate(student, teacher, loader, args, device):
    student.eval()
    total_loss = 0.0
    total_det = 0.0
    total_desc = 0.0
    total_kpts = 0.0
    batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="val", mininterval=5.0)
        use_amp = args.fp16 and device.type == "cuda"
        for img in pbar:
            img = img.to(device, non_blocking=True)
            labels, teacher_mask, teacher_desc = get_teacher_labels(teacher, img)

            with torch.cuda.amp.autocast(enabled=use_amp):
                semi, desc = student(img)
                det_loss = F.cross_entropy(semi, labels, weight=args._ce_weight)
                if teacher_desc is not None:
                    # Cosine Embedding Loss로 변경: 벡터 방향성(매칭 성능) 강화를 위해
                    # b*c*h*w 형태를 (N, c)로 변환
                    b, c, h, w = desc.shape
                    student_desc_flat = desc.permute(0, 2, 3, 1).reshape(-1, c)
                    teacher_desc_flat = teacher_desc.permute(0, 2, 3, 1).reshape(-1, c)
                    # 동일한 위치의 디스크립터는 같은 방향(1)을 가리키도록 학습
                    target = torch.ones(student_desc_flat.size(0), device=desc.device)
                    desc_loss = F.cosine_embedding_loss(student_desc_flat, teacher_desc_flat, target)
                else:
                    desc_loss = torch.zeros((), device=det_loss.device)
                loss = det_loss + (float(args.desc_weight) * desc_loss)

            heatmap = F.softmax(semi, dim=1)[:, :-1]
            heatmap = heatmap.view(-1, 8, 8, args.height // 8, args.width // 8)
            heatmap = heatmap.permute(0, 3, 1, 4, 2).reshape(-1, args.height, args.width)
            top_k_map = select_top_k_heatmap(
                heatmap * (heatmap > args.detection_threshold),
                k=args.top_k,
            )
            pred_pts = (top_k_map > 0).float()

            total_kpts += float(pred_pts.sum().item()) / max(1, img.size(0))
            total_loss += float(loss.item())
            total_det += float(det_loss.item())
            total_desc += float(desc_loss.item())
            batches += 1

    if batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / batches,
        total_det / batches,
        total_desc / batches,
        total_kpts / batches,
    )


def save_training_state(path, student, optimizer, scaler, epoch, global_step, next_debug_step):
    payload = {
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "next_debug_step": None if next_debug_step is None else int(next_debug_step),
    }
    torch.save(payload, path)


def load_training_state(path, student, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "student" in ckpt:
        student.load_state_dict(ckpt["student"], strict=True)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as e:
                print(f"[Training] Optimizer state mismatch: newly unfrozen backbone detected. Starting fresh optimizer.")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        next_debug_step = ckpt.get("next_debug_step", None)
        if next_debug_step is not None:
            next_debug_step = int(next_debug_step)
        return epoch, global_step, next_debug_step
    else:
        student.load_state_dict(ckpt, strict=True)
        print(f"[Training] Loaded raw weights from {path}. Starting from epoch 0.")
        return 0, 0, None

# --- [4] 데이터셋 정의 ---

class ImageFolderDataset(Dataset):
    def __init__(self, root, size, max_rotate=0.0, max_scale=0.0, max_perspective=0.0):
        exts = ["jpg", "jpeg", "png", "JPG", "PNG"]
        self.paths = []
        for e in exts:
            self.paths.extend(glob.glob(os.path.join(root, "**", f"*.{e}"), recursive=True))
        self.paths = sorted(list(set(self.paths)))
        self.h, self.w = size
        self.max_rotate = float(max_rotate)
        self.max_scale = float(max_scale)
        self.max_perspective = float(max_perspective)
        print(f"✅ 데이터셋 경로: {root} | 📊 이미지 개수: {len(self.paths)}개")

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None: return self.__getitem__((idx + 1) % len(self.paths))
        img = self._apply_homography(img)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype(np.float32)/255.0).unsqueeze(0).repeat(3, 1, 1)

    def _apply_homography(self, img):
        if self.max_rotate <= 0 and self.max_scale <= 0 and self.max_perspective <= 0:
            return img

        h, w = img.shape[:2]
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        scale = 1.0 + random.uniform(-self.max_scale, self.max_scale)

        rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = rot

        if self.max_perspective > 0:
            dx = w * self.max_perspective
            dy = h * self.max_perspective
            src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            dst = src + np.array([
                [random.uniform(-dx, dx), random.uniform(-dy, dy)],
                [random.uniform(-dx, dx), random.uniform(-dy, dy)],
                [random.uniform(-dx, dx), random.uniform(-dy, dy)],
                [random.uniform(-dx, dx), random.uniform(-dy, dy)],
            ], dtype=np.float32)
            Hp = cv2.getPerspectiveTransform(src, dst)
            H = Hp @ H

        warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        return warped

# --- [5] 메인 학습 함수 ---

def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    student = SuperPointNetV2().to(device)
    use_teacher_desc = bool(getattr(args, "use_teacher_desc", True))
    teacher = SuperPoint(return_desc=use_teacher_desc).to(device)
    
    # 가중치 로드
    w_path = os.path.join(ROOT_DIR, "superpoint_v1.pth")
    teacher.load_state_dict(torch.load(w_path, map_location=device, weights_only=False))
    teacher.eval()

    # [핵심 수렴 안정화 조치]
    # 학생의 뼈대(MobileNetV2 Backbone)는 이미 ImageNet으로 완성된 지능이므로, 
    # 머리(Head) 계층 초반 훈련 시에는 완전 동결(Freeze)하고, 2차 파인튜닝 시에는 BatchNorm 통계만 얼려둡니다.
    freeze_backbone = bool(getattr(args, "freeze_backbone", True))
    
    def set_student_train():
        student.train()
        if freeze_backbone:
            for param in student.backbone.parameters():
                param.requires_grad = False
        else:
            for module in student.backbone.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
                    if module.weight is not None: module.weight.requires_grad = False
                    if module.bias is not None: module.bias.requires_grad = False

    set_student_train()
    
    if freeze_backbone:
        print("[Training] 🛡️ Pre-trained MobileNetV2 Backbone is FROZEN.")
    else:
        print("[Training] 🔓 Backbone UNFROZEN. Heads and Backbone will be fine-tuned together (BN frozen).")

    # 학습은 그래디언트가 열려있는(requires_grad=True) 파라미터들만 진행
    trainable_params = filter(lambda p: p.requires_grad, student.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr))
    
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # FP16 가속

    ce_weight = torch.ones(65, device=device)
    ce_weight[64] = args.dustbin_weight # 배경 억제
    args._ce_weight = ce_weight

    # 속도 최적화: num_workers 및 pin_memory 적용
    loader = DataLoader(
        ImageFolderDataset(
            os.path.join(ROOT_DIR, args.data_dir),
            (args.height, args.width),
            max_rotate=args.max_rotate,
            max_scale=args.max_scale,
            max_perspective=args.max_perspective,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    val_dir = getattr(args, "val_dir", None)
    if val_dir:
        val_path = os.path.join(ROOT_DIR, val_dir)
        if os.path.isdir(val_path):
            val_batch = int(getattr(args, "val_batch_size", args.batch_size))
            val_loader = DataLoader(
                ImageFolderDataset(
                    val_path,
                    (args.height, args.width),
                    max_rotate=0.0,
                    max_scale=0.0,
                    max_perspective=0.0,
                ),
                batch_size=val_batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )

    save_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "v14_latest.pth")
    resume_path = getattr(args, "resume_from", None)
    if not resume_path:
        resume_path = os.path.join(save_dir, "v14_resume.pt")
    elif not os.path.isabs(resume_path):
        resume_path = os.path.join(ROOT_DIR, resume_path)

    save_every_steps = max(0, int(getattr(args, "save_every_steps", 2000)))
    debug_every = int(getattr(args, "debug_every", 100))
    val_interval = max(1, int(getattr(args, "val_interval", 1)))

    set_student_train()
    start_epoch = 0
    global_step = 0
    next_debug_step = debug_every if debug_every > 0 else None

    if bool(getattr(args, "resume", False)) and os.path.exists(resume_path):
        start_epoch, global_step, loaded_next_debug = load_training_state(
            resume_path,
            student,
            optimizer,
            scaler,
            device,
        )
        if loaded_next_debug is not None:
            next_debug_step = loaded_next_debug
        print(f"[resume] loaded: epoch={start_epoch}, global_step={global_step}")

    next_save_step = None
    if save_every_steps > 0:
        next_save_step = ((global_step // save_every_steps) + 1) * save_every_steps

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            # [떨림 방지] mininterval 설정을 통해 진행바의 잦은 갱신을 막습니다.
            pbar = tqdm(loader, desc=f"v14-ULTRA Ep {epoch+1}", mininterval=5.0)
            
            for i, img in enumerate(pbar):
                img = img.to(device, non_blocking=True)
                labels, teacher_mask, teacher_desc = get_teacher_labels(teacher, img)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    semi, desc = student(img)
                    det_loss = F.cross_entropy(semi, labels, weight=ce_weight)
                    if use_teacher_desc and float(args.desc_weight) > 0 and teacher_desc is not None:
                        # Cosine Embedding Loss로 변경
                        b, c, h, w = desc.shape
                        student_desc_flat = desc.permute(0, 2, 3, 1).reshape(-1, c)
                        teacher_desc_flat = teacher_desc.permute(0, 2, 3, 1).reshape(-1, c)
                        target = torch.ones(student_desc_flat.size(0), device=desc.device)
                        desc_loss = F.cosine_embedding_loss(student_desc_flat, teacher_desc_flat, target)
                    else:
                        desc_loss = torch.zeros((), device=det_loss.device)
                    loss = det_loss + (float(args.desc_weight) * desc_loss)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += img.size(0)
                if next_debug_step is not None and global_step >= next_debug_step:
                    heatmap = F.softmax(semi, dim=1)[:, :-1].view(-1, 8, 8, args.height//8, args.width//8).permute(0, 3, 1, 4, 2).reshape(-1, args.height, args.width)
                    # 엄격한 선별 적용 (Top-K)
                    top_k_map = select_top_k_heatmap(heatmap * (heatmap > args.detection_threshold), k=args.top_k)
                    save_debug_trace(img, teacher_mask, top_k_map, os.path.join(save_dir, "debug_v14.jpg"))
                    while next_debug_step <= global_step:
                        next_debug_step += debug_every
                    
                    with torch.no_grad():
                        pred_pts = (top_k_map > 0).float()
                        precision = (pred_pts * teacher_mask).sum() / (pred_pts.sum() + 1e-6)
                        max_prob = heatmap.max().item()
                        pbar.set_postfix({
                            "Loss": f"{loss.item():.3f}",
                            "Det": f"{det_loss.item():.3f}",
                            "Prec": f"{precision:.2f}", 
                            "Pts": f"{int(pred_pts.sum()/args.batch_size)}",
                            "MaxP": f"{max_prob:.3f}"
                        })

                if next_save_step is not None and global_step >= next_save_step:
                    save_training_state(
                        resume_path,
                        student,
                        optimizer,
                        scaler,
                        epoch,
                        global_step,
                        next_debug_step,
                    )
                    while next_save_step <= global_step:
                        next_save_step += save_every_steps

            torch.save(student.state_dict(), checkpoint_path)
            save_training_state(
                resume_path,
                student,
                optimizer,
                scaler,
                epoch + 1,
                global_step,
                next_debug_step,
            )

            if val_loader is not None and ((epoch + 1) % val_interval == 0):
                val_loss, val_det, val_desc, val_kpts = evaluate(student, teacher, val_loader, args, device)
                print(
                    f"[val] loss={val_loss:.4f} det={val_det:.4f} desc={val_desc:.4f} "
                    f"avg_kpts={val_kpts:.1f}"
                )
                set_student_train()
    except KeyboardInterrupt:
        torch.save(student.state_dict(), checkpoint_path)
        save_training_state(
            resume_path,
            student,
            optimizer,
            scaler,
            current_epoch,
            global_step,
            next_debug_step,
        )
        print(f"\n[interrupt] checkpoint saved: {resume_path}")
        print(f"[interrupt] latest weights saved: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    
    with open(os.path.join(ROOT_DIR, args.config), 'r') as f:
        for k, v in yaml.safe_load(f).items(): setattr(args, k, v)
    
    # 기본값 보장
    for k, v in [
        ('top_k', 3000),
        ('detection_threshold', 0.2),
        ('max_rotate', 15.0),
        ('max_scale', 0.1),
        ('max_perspective', 0.05),
        ('debug_every', 100),
        ('val_interval', 1),
        ('save_every_steps', 2000),
        ('resume', False),
        ('resume_from', None),
        ('use_teacher_desc', True),
        ('val_batch_size', None),
        ('output_dir', 'checkpoints'),
    ]:
        if not hasattr(args, k): setattr(args, k, v)

    os.makedirs(os.path.join(ROOT_DIR, args.output_dir), exist_ok=True)
    train(args)