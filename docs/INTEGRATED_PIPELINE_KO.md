# 🎬 통합 VSLab SLAM 파이프라인

> 동영상 1개를 입력하면 **프론트엔드 특징점 추출 → Optical Flow 융합 트래킹 → 3D 매핑(g2o)** 이 원스톱으로 자동 진행됩니다!

---

## 🚀 통합 실행 스크립트 

```bash
# 동영상 또는 이미지 시퀀스 폴더를 입력하면 SLAM 이 동작합니다.
uv run python scripts/superpoint_app.py --input your_video.mp4 --weights weights/v14_desc_ft.pth --config config/default.yaml
```

**이제 파편화된 스크립트 없이 오직 `superpoint_app.py` 하나만으로 모든 게 됩니다!** ✨

---

## 📊 처리 흐름

```
동영상 또는 이미지 폴더 입력 (superpoint_app.py)
         ↓
   ┌─────────────────────┐
   │ 1. 프레임 로드 및 필터│
   ├─────────────────────┤
   │ 1. CLAHE 전처리      │
   │ 2. Semantic 마스킹   │
   │ 3. 그림자/하늘 마스킹│
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │ 2. 프론트엔드 (AI)    │
   ├─────────────────────┤
   │ 1. SuperPoint 추출   │
   │    (간격: --sp-interval)│
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │ 3. 트래킹 및 매칭    │
   ├─────────────────────┤
   │ 1. LK Optical Flow   │
   │ 2. 디스크립터 매칭   │
   │ 3. PnP 포즈 추정     │
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │ 4. 매핑 및 백엔드    │
   ├─────────────────────┤
   │ 1. MapPoint 3D 삼각측량 │
   │ 2. Local/Global BA (g2o)│
   │ 3. Loop Closure 심사 │
   └─────────────────────┘
         ↓
   결과 폴더에 저장 ✅ (ply, png, txt)
```

---

## 💻 다양한 사용 예제

### 예제 1: 밸런스 옵션 (권장)

```bash
uv run python scripts/superpoint_app.py \
    --input video.mp4 \
    --weights weights/v14_desc_ft.pth \
    --config config/default.yaml \
    --use_semantic \
    --sp-interval 2
```

### 예제 2: 스케일 드리프트 억제 (강력한 그림자 필터)

야외 주행이나 텍스처가 복잡한 그림자 환경에서 가짜 특징점을 차단해 스케일을 유지합니다.

```bash
uv run python scripts/superpoint_app.py \
    --input video.mp4 \
    --weights weights/v14_desc_ft.pth \
    --config config/default.yaml \
    --aggressive_shadow_filter
```

### 예제 3: 실시간 시각화 끄기 (성능 15ms 확보)

```bash
uv run python scripts/superpoint_app.py \
    --input video.mp4 \
    --weights weights/v14_desc_ft.pth \
    --no-viz
```

---

## 📁 결과 폴더 구조

```
result/superpoint_3d_XX/
│
├── final_slam_map.ply                ← 최종 3D 포인트클라우드 파일 (Open3D 등 확인 가능)
├── topdown_map.png                   ← 2D 궤적 (위에서 내려다본 형태)
└── trajectory_xy.txt                 ← 추정된 카메라 궤적 좌표 로그
```

---

## 📊 결과 분석

실행 완료 후 평가 스크립트를 사용하여 성능을 검증할 수 있습니다.

```bash
uv run python scripts/evaluate.py --result result/superpoint_3d_01 --csv diff.csv
```

결과 예시:
```
======================================================================
  📊 SLAM 성능 평가 보고서
======================================================================
  ⏱️  Latency
    Avg Total:       61.16 ms
    Avg SuperPoint:  11.51 ms
    FPS:             16.35
...
```

---
