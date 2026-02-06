# SuperPoint v2 (MobileNet) 학습 가이드

이 레포는 **라벨(.npz) 기반 지도학습** 스크립트를 제공합니다. 아래 가이드는 `scripts/train_synthetic.py`에 맞춰 작성되었습니다.
KITTI 데이터를 이용한 **지도학습 스크립트**는 `scripts/train_superpoint.py`를 참고하세요.

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
python scripts/superpoint_app.py --mode slam --input your_video.mp4 --weights checkpoints/superpoint_base_synthetic.pth
```

## 6) 참고
- 이 스크립트는 **라벨(.npz) 기반 지도학습**을 위한 기본 베이스입니다.
- 더 높은 성능이 필요하면, SuperPoint 논문의 **Homography Adaptation** 또는 **descriptor 학습**까지 확장하는 로직을 추가로 구현해야 합니다.
