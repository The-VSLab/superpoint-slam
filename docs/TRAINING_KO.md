# SuperPoint v2 (MobileNet) 학습 가이드

이 레포는 **라벨(.npz) 기반 지도학습**과 **자가학습(호모그래피+옵션 Harris)** 스크립트를 제공합니다.
아래 가이드는 `scripts/train_synthetic.py`(라벨 기반)와 `scripts/train_superpoint.py`(자가학습)에 맞춰 작성되었습니다.

## 1) 개요
- **방법:** 이미지와 대응되는 키포인트 라벨(.npz)을 사용해 Detector를 **CrossEntropy**로 학습합니다.
- **장점:** 구현이 단순하며, 라벨이 준비된 경우 빠르게 학습 가능
- **출력:** 기본값 `checkpoints/superpoint_base_synthetic.pth` (옵션으로 변경 가능)

## 2) 학습 데이터 준비
- 입력은 **이미지(.jpg)**와 **동일 이름의 라벨(.npz)** 쌍입니다.
- `.npz` 내부에는 `points` 키가 있어야 하며, `[row, col]` 좌표 배열을 담습니다.
- 다양한 장면(조명/모션/텍스처)을 포함할수록 매칭 품질이 좋아집니다.

예시 구조:
```
train_data/
├── seq01/
│   ├── 0001.jpg
│   ├── 0001.npz
│   ├── 0002.jpg
│   └── 0002.npz
└── seq02/
    ├── 0001.jpg
    ├── 0001.npz
    ├── 0002.jpg
    └── 0002.npz
```

## 3) 실행 방법
```bash
python scripts/train_synthetic.py \
  --data_dir train_data/seq01 \
  --epochs 10 \
  --batch_size 16 \
  --height 480 \
  --width 640
```

결과 가중치:
```
checkpoints/superpoint_base_synthetic.pth
```

## 4) 주요 하이퍼파라미터
- `--epochs`, `--batch_size`, `--lr`
  - 학습 반복, 배치 크기, 학습률
- `--height`, `--width`
  - 입력 해상도 (8의 배수 권장)
- `--weights_out`, `--output_dir`
  - 저장 파일명/경로 지정

## 5) 결과 적용
학습한 가중치를 기존 파이프라인에 넣으면 됩니다.
```bash
uv run python scripts/superpoint_app.py --input your_video.mp4 --weights checkpoints/superpoint_base_synthetic.pth --config config/default.yaml
```
결과는 `path_final/final_slam_map.ply`로 저장됩니다.

## 6) 참고
- 이 스크립트는 **라벨(.npz) 기반 지도학습**을 위한 기본 베이스입니다.
- 더 높은 성능이 필요하면, SuperPoint 논문의 **Homography Adaptation** 또는 **descriptor 학습**까지 확장하는 로직을 추가로 구현해야 합니다.

---

# 2) 자가학습 (Homography + Harris 옵션)

## 개요
- **방법:** 입력 이미지에 임의 호모그래피를 적용해 detector/descriptor를 자가학습
- **옵션:** Harris 코너를 정답지로 사용해 detector에 감독 신호 추가 가능
- **출력:** 기본값 `checkpoints/superpoint_v3_mobilenet_ft.pth`

## 실행 방법 (CLI)
```bash
python scripts/train_superpoint.py \
  --data_dir dataset/training \
  --epochs 10 \
  --batch_size 4 \
  --lr 5e-5
```

## Harris 옵션 (grid noise 억제용)
- `--harris_weight`: Harris supervised loss 가중치
- `--harris_block_size`, `--harris_ksize`, `--harris_k`, `--harris_thresh`

예시:
```bash
python scripts/train_superpoint.py \
  --data_dir dataset/training \
  --harris_weight 0.3 \
  --harris_thresh 0.01
```

---

# 3) 모델 성능 극대화 (2-Phase Teacher-Student 증류)

SuperPoint VGG 원본의 성능(Teacher)을 경량화된 MobileNetV2(Student)로 안정적으로 이식하기 위한 **2단계(Phase 1 & 2) 증류(Distillation)** 학습 가이드입니다.

## 🌟 개요 (왜 2단계로 나누어 학습하나요?)
MobileNetV2 백본은 ImageNet으로 사전학습된 고도의 지능(BatchNorm 통계 등)을 가지고 있습니다. 
만약 처음부터 백본 전체를 열고 학습하면, 무작위 초기화된 머리(Head) 계층이 내뿜는 거대한 초기 오차(Loss)와 파괴적인 그래디언트로 인해 백본의 지능이 완전히 오염되는 **'Catastrophic Forgetting(망각)'** 현상이 발생합니다.
이를 방지하기 위해 다음 2단계 조치를 취합니다.

---

### 🔹 Phase 1: 뼈대 동결 학습 (Frozen Backbone)
- **목적:** 백본(MobileNet)의 파라미터를 100% 잠가 보호하고, 머리(Detector & Descriptor Head) 영역만 일차적으로 학습시킵니다.
- **설정 (`learning/config_v13.yml`):**
  - `freeze_backbone: true`
  - `epochs: 20`, `lr: 1.0e-4`
- **결과물:** 약 52~55% 신뢰도를 방어하는 안정된 기초 가중치가 생성됩니다. (예: `checkpoints/v15_depthwise_resume.pt`)

### 🔹 Phase 2: 정밀 튜닝 및 BatchNorm 방어 (Safe Fine-Tuning)
- **목적:** 머리의 학습이 안정되었으므로 백본의 잠금을 풀고(Unfreeze) 픽셀 단위로 함께 교정합니다. **단, BN 오염 방지를 위해 BatchNorm 층은 코드 단에서 강제로 평가 모드(`eval()`)로 동결 유지합니다.**
- **설정 (`learning/config_v13.yml`):**
  - `freeze_backbone: false` (코드 내부적으로 BN만 자동 동결 처리됨)
  - `resume: true`, `resume_from: checkpoints/v15_depthwise_resume.pt` (Phase 1 가중치 로드)
  - `epochs: 30`, `lr: 1.0e-5` (낮은 학습률 사용)
- **결과물:** 신뢰도가 70% 이상으로 치솟는 초정밀 최적화 모델이 완성됩니다. (`checkpoints/v14_latest.pth`)

---

## 💻 실행 방법
두 Phase 모두 실행 명령어는 동일하며, `config_v13.yml`의 설정값만 조절하여 진행합니다.

```bash
uv run python learning/train_superpoint_v13.py --config learning/config_v13.yml
```

## 참고 사항
- Teacher 가중치는 코드 내부에서 `weights/superpoint_v1.pth` (MagicLeap 오리지널 완전체, BatchNorm 미포함 버전)를 로드합니다.
- 평가(Validation) 지표에서 `Prec`(정밀도)와 `MaxP`(최대 확신도)가 극적으로 상승(>0.8)하는 것을 모니터링하세요.
