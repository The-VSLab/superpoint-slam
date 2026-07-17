"""
Descriptor-head-only fine-tune (v14 디스크립터 붕괴 복구용).

배경: weights/v14_latest.pth는 검출/트래킹은 정상이나 descriptor head가
붕괴되어(프레임 내 키포인트 간 유사도 p95=0.999, 인접 프레임 매칭 inlier 7%)
루프 클로저가 불가능하다. 백본+검출 헤드는 그대로 두고(FPS/트래킹 성능 보존),
descriptor head(convDa, convDb)만 원본 SuperPoint(VGG) teacher로부터
재증류한다.

데이터: dataset/training/<seq>/image_0/*.png (KITTI odometry, 라벨 불필요).
평가 시퀀스(07, 08)는 학습에서 제외한다.

실행:
  ./.venv/bin/python learning/finetune_descriptor.py \
      --epochs 2 --batch_size 8 --stride 3 --lr 1e-3
"""
import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
for p in [ROOT_DIR, CUR_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from models.superpoint_mobilenet import SuperPointNetV2
from learning.original_superpoint import SuperPoint
from learning.train_superpoint import ImageFolderDataset


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset/training")
    ap.add_argument("--exclude", nargs="*", default=["07", "08"],
                    help="학습에서 제외할 시퀀스 (평가용)")
    ap.add_argument("--init_weights", default="weights/v14_latest.pth")
    ap.add_argument("--teacher_weights", default="weights/superpoint_v1.pth")
    ap.add_argument("--out", default="checkpoints/v14_desc_ft.pth")
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3, help="프레임 서브샘플 간격")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kp_weight", type=float, default=4.0,
                    help="teacher 키포인트 셀의 desc loss 가중 (배경 대비)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--limit_steps", type=int, default=0, help="타이밍 측정용 (0=전체)")
    return ap.parse_args()


def main():
    args = build_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[ft] device={device}")

    # --- student: v14 로드, descriptor head 재초기화 + 그것만 학습 ---
    student = SuperPointNetV2().to(device)
    sd = torch.load(os.path.join(ROOT_DIR, args.init_weights), map_location=device, weights_only=False)
    if isinstance(sd, dict) and "student" in sd:
        sd = sd["student"]
    student.load_state_dict(sd, strict=True)

    # 붕괴된 국소해에서 시작하지 않도록 desc head는 새로 초기화
    for m in list(student.convDa.modules()) + [student.convDb]:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.reset_parameters()

    for p in student.parameters():
        p.requires_grad = False
    trainable = []
    for module in [student.convDa, student.convDb]:
        for p in module.parameters():
            p.requires_grad = True
            trainable.append(p)
    n_train = sum(p.numel() for p in trainable)
    print(f"[ft] trainable params (desc head only): {n_train/1e3:.1f}K")

    # backbone/detector는 eval 모드 고정 (BN 통계 보존), desc head만 train
    student.eval()
    student.convDa.train()

    # --- teacher ---
    teacher = SuperPoint(return_desc=True).to(device)
    teacher.load_state_dict(torch.load(os.path.join(ROOT_DIR, args.teacher_weights),
                                       map_location=device, weights_only=False))
    teacher.eval()

    # --- data: KITTI image_0, 평가 시퀀스 제외, stride 서브샘플 ---
    ds = ImageFolderDataset(os.path.join(ROOT_DIR, args.data_dir), (args.height, args.width),
                            max_rotate=5.0, max_scale=0.05, max_perspective=0.02)
    def keep(path):
        if f"{os.sep}image_0{os.sep}" not in path:
            return False
        return not any(f"{os.sep}{seq}{os.sep}" in path for seq in args.exclude)
    ds.paths = sorted([p for p in ds.paths if keep(p)])[:: max(1, args.stride)]
    print(f"[ft] train images after filter/stride: {len(ds.paths)}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    out_path = os.path.join(ROOT_DIR, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        running = 0.0
        for img in loader:
            img = img.to(device, non_blocking=True)
            with torch.no_grad():
                t_out = teacher({"image": img[:, 0:1]})
                t_desc = t_out["desc"]                       # (B,256,h,w) L2-normalized
                if t_desc is None:
                    raise RuntimeError("teacher desc is None — 증류 불가")
                # teacher 키포인트가 있는 8x8 셀 → 가중치 맵
                b, _, hh, ww = t_desc.shape
                kp_cell = torch.zeros((b, hh, ww), device=device)
                for i, kps in enumerate(t_out["keypoints"]):
                    if len(kps) > 0:
                        cx = torch.clamp((kps[:, 0] / 8).long(), 0, ww - 1)
                        cy = torch.clamp((kps[:, 1] / 8).long(), 0, hh - 1)
                        kp_cell[i, cy, cx] = 1.0
                w_map = 1.0 + args.kp_weight * kp_cell        # (B,h,w)

            _, s_desc = student(img)                          # (B,256,h,w)
            cos = F.cosine_similarity(s_desc, t_desc, dim=1)  # (B,h,w)
            loss = (((1.0 - cos) * w_map).sum() / w_map.sum())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            step += 1
            if step % args.log_every == 0:
                print(f"[ft] ep{epoch+1} step{step} desc_loss={running/args.log_every:.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                running = 0.0
            if args.limit_steps and step >= args.limit_steps:
                print("[ft] limit_steps reached — stopping (timing mode)")
                torch.save(student.state_dict(), out_path)
                return

        torch.save(student.state_dict(), out_path)
        print(f"[ft] epoch {epoch+1} done -> saved {out_path}", flush=True)

    print(f"[ft] finished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
