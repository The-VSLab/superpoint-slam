# 🚀 빠른 시작 가이드 (5분)

## 폴더 구조 한눈에

```
superpoint-slam/
├── 📁 scripts/
│   ├── py_superpoint.py         ← 특징점 추출
│   ├── matcher_main.py          ← 👈 여기서 매칭 실행
│   ├── superpoint_app.py        ← Demo/SLAM 통합 CLI
│   ├── integrated_matching.py   ← 통합 파이프라인
│   └── test_matching.py         ← 시스템 테스트
├── 📁 slam/                     ← SLAM 모듈 (visual_slam_3d.py)
├── 📁 matcher_module/           ← GPU 매칭 모듈
│   ├── btmatcher.py             ← BT-Matcher 구현
│   └── utils.py                 ← 유틸리티 함수
├── 📁 npy_outputs/              ← 입력: SuperPoint 결과
│   ├── frame_00001_pts.npy
│   ├── frame_00001_desc.npy
│   └── ...
└── 📁 matching_results/         ← 출력: 매칭 결과
    ├── matches_viz/             ← 이미지 (매칭 시각화)
    └── matches_data/            ← 데이터 (매칭 정보)
```

---

## 3단계: 실행하기

### 단계 1️⃣: 특징점 추출 (SuperPoint)

```bash
python scripts/py_superpoint.py --input assets/icl_snippet/ --save_npy
```

또는 통합 CLI 사용:
```bash
python scripts/superpoint_app.py --mode demo --input assets/icl_snippet/ --weights checkpoints/v14_latest.pth --save_npy
```

장치 선택 예시:
```bash
python scripts/superpoint_app.py --mode demo --input assets/icl_snippet/ --weights checkpoints/v14_latest.pth --device auto
python scripts/superpoint_app.py --mode demo --input assets/icl_snippet/ --weights checkpoints/v14_latest.pth --device cuda
```

✅ 결과: `npy_outputs/` 폴더에 파일 생성
- `frame_00001_pts.npy` (특징점 좌표)
- `frame_00001_desc.npy` (특징점 설명자)
- `frame_00001_heatmap.npy` (신뢰도)

---

## 🔧 모듈 경로 설정 (중요)

본 프로젝트는 루트 기준 패키지(`models/`, `frontend/`, `tracking/`, `io_utils/`)를 사용합니다.  
다음 중 하나를 만족해야 합니다.

1. **프로젝트 루트에서 실행**
   - 예: `python scripts/py_superpoint.py ...`
2. **PYTHONPATH 설정**
```bash
export PYTHONPATH=.
python scripts/py_superpoint.py --input assets/icl_snippet/ --save_npy
```

### 단계 2️⃣: 시스템 테스트 (선택사항)

```bash
python scripts/test_matching.py
```

✅ GPU 설정과 필요한 패키지를 확인합니다.

### 단계 3️⃣: 매칭 실행

```bash
python scripts/matcher_main.py --npy_dir npy_outputs --output_dir matching_results
```

---

## 🔁 통합 CLI로 SLAM 실행

```bash
python scripts/superpoint_app.py --mode slam --input <VIDEO_PATH> --weights checkpoints/v14_latest.pth --resize 640 480
```

✅ 결과: `path_final/final_slam_map.ply` 생성 (3D 포인트 클라우드)

---

## SLAM 실행 (옵션 예시)

```bash
python scripts/superpoint_app.py \
  --mode slam \
  --input assets/test2.mp4 \
  --weights checkpoints/v14_latest.pth \
  --resize 640 480 \
  --slam_conf_thresh 0.003 \
  --slam_nms_dist 4
```

---

## 💡 자주 쓰는 명령어

### 특정 프레임 쌍만 매칭하기
```bash
python scripts/matcher_main.py --frame_pair frame_00001:frame_00003
```

### 매칭 민감도 조절
```bash
# 더 정확하게 (엄격함)
python scripts/matcher_main.py --nn_thresh 0.6

# 더 관대하게 (많은 매칭)
python scripts/matcher_main.py --nn_thresh 0.9
```

### 빠른 처리 (기하학 검증 스킵)
```bash
python scripts/matcher_main.py --no_geometric_test
```

---

## 📊 결과 확인하기

### 이미지로 확인
```bash
# Windows
start matching_results\matches_viz\frame_00001_frame_00002_matches.png

# Mac/Linux
open matching_results/matches_viz/frame_00001_frame_00002_matches.png
```

### Python에서 데이터 로드
```python
import numpy as np

# 매칭 데이터 로드
data = np.load('matching_results/matches_data/frame_00001_frame_00002_matches.npy',
              allow_pickle=True).item()

matches = data['matches']          # (L, 3) - 매칭 결과
inlier_mask = data['inlier_mask']  # (L,) - 신뢰도 높은 매칭 표시

print(f"매칭 개수: {len(matches)}")
print(f"좋은 매칭: {np.sum(inlier_mask)}")
```

---

## 🔧 문제 해결

| 문제 | 해결책 |
|------|--------|
| `ModuleNotFoundError: matcher_module` | 현재 디렉토리가 프로젝트 루트인지 확인 |
| `CUDA out of memory` | `--no_geometric_test` 옵션 추가 또는 CPU 사용 |
| 매칭 개수가 너무 적음 | `--nn_thresh 0.9` 로 더 관대하게 설정 |
| 매칭 개수가 너무 많음 | `--nn_thresh 0.5` 로 더 엄격하게 설정 |
| `npy_outputs` 폴더가 없음 | 먼저 `scripts/py_superpoint.py` 실행해서 특징점 추출 |

---

## 📈 성능 팁

| 작업 | 방법 |
|------|------|
| 빠른 처리 | `--no_geometric_test` 추가 (2-3배 빠름) |
| 정확도 향상 | `--nn_thresh 0.6` 로 설정 |
| GPU 메모리 절약 | CPU 사용: `--device cpu` 또는 `--no_cuda` |

---

## 🎓 코드 예제

### 예제 1: 간단한 매칭
```python
from matcher_module import BTMatcher
import numpy as np

desc1 = np.load('npy_outputs/frame_00001_desc.npy').T
desc2 = np.load('npy_outputs/frame_00002_desc.npy').T

matcher = BTMatcher()
matches = matcher.match(desc1, desc2)

print(f"{len(matches)}개 매칭 발견")
```

### 예제 2: 매칭 시각화
```python
from matcher_module import draw_matches
import cv2
import numpy as np

pts1 = np.load('npy_outputs/frame_00001_pts.npy')
pts2 = np.load('npy_outputs/frame_00002_pts.npy')
img1 = np.load('npy_outputs/frame_00001_heatmap.npy')
img2 = np.load('npy_outputs/frame_00002_heatmap.npy')

# 매칭 (위의 예제 코드로 수행)
output = draw_matches(img1, pts1, img2, pts2, matches)
cv2.imwrite('result.png', output)
```

### 예제 3: RANSAC 검증
```python
from matcher_module import compute_fundamental_matrix

F, inlier_mask = compute_fundamental_matrix(pts1, pts2, matches)
print(f"신뢰도 높은 매칭: {np.sum(inlier_mask)}/{len(matches)}")
```

---

## 📚 다음 단계

매칭 후 할 수 있는 것들:
- ✅ 카메라 캘리브레이션
- ✅ 3D 삼각측량 (Triangulation)
- ✅ Structure from Motion (SfM)
- ✅ Visual SLAM 구성
- ✅ 이미지 정합 (Image Stitching)

---

## 💬 Q&A

**Q: py_superpoint.py를 수정해야 하나?**
A: 아니요! 전혀 수정할 필요 없습니다. `scripts/matcher_main.py`가 별도로 작동합니다.

**Q: 자신의 이미지로 시도하려면?**
A: `scripts/py_superpoint.py --input <이미지_폴더> --save_npy` 실행 후 `scripts/matcher_main.py` 실행

**Q: GPU가 없으면?**
A: CPU로도 작동합니다. (다만 느림) 자동으로 GPU가 없으면 CPU 사용

**Q: 매칭 품질이 안 좋으면?**
A: 이미지의 조명, 각도 차이 등이 영향을 미칩니다. `--nn_thresh` 값 조정 시도

---
