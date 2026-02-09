# 🎬 통합 파이프라인 - 한 번에 모든 것!

> 동영상 1개를 입력하면 **특징점 추출 → 실시간 매칭 → 시각화**가 자동으로 진행됩니다!

---

## 🚀 빠른 시작 (1단계!)

```bash
# 동영상 또는 이미지 폴더를 입력하면 끝!
python scripts/integrated_matching.py --input your_video.mp4
```

**그것만으로 모든 게 된다!** ✨

---

## 📊 처리 흐름

```
동영상 또는 이미지 폴더 입력
         ↓
   ┌─────────────────────┐
   │  Frame 1 처리        │
   ├─────────────────────┤
   │ 1. SuperPoint 실행   │
   │    ↓ 특징점 추출    │
   │ 2. 저장하고 다음으로│
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │  Frame 2 처리        │
   ├─────────────────────┤
   │ 1. SuperPoint 실행   │
   │    ↓ 특징점 추출    │
   │ 2. Frame 1과 매칭    │
   │    ↓ GPU 가속       │
   │ 3. 시각화 & 저장    │
   └─────────────────────┘
         ↓
   ┌─────────────────────┐
   │  Frame 3, 4, 5...   │
   │  계속 반복!          │
   └─────────────────────┘
         ↓
   매칭 결과 폴더에 저장 ✅
```

---

## 💻 사용 예제

### 예제 1: 기본 사용 (동영상)

```bash
python scripts/integrated_matching.py --input video.mp4
```

### 예제 2: 이미지 폴더 처리

```bash
python scripts/integrated_matching.py --input ./my_images/
```

### 예제 3: 커스텀 설정

```bash
# 매칭 민감도 조절
python scripts/integrated_matching.py \
    --input video.mp4 \
    --nn_thresh 0.8 \
    --output my_results

# CPU만 사용 (GPU 없는 경우)
python scripts/integrated_matching.py \
    --input video.mp4 \
    --no_cuda

# 실시간 표시 비활성화 (조용한 처리)
python scripts/integrated_matching.py \
    --input video.mp4 \
    --no_display
```

---

## 📁 결과 폴더 구조

```
matching_results_integrated/          (기본값, --output으로 변경 가능)
│
├── frames/                           매칭 시각화 이미지
│   ├── frame_00001_heatmap.png
│   ├── frame_00002_heatmap.png
│   └── ...
│
├── features/                         특징점 데이터 (NPY)
│   ├── frame_00001_pts.npy
│   ├── frame_00001_desc.npy
│   ├── frame_00001_heatmap.npy
│   └── ...
│
├── matches_viz/                      매칭 시각화 이미지
│   ├── frame_00001_frame_00002_matches.png
│   ├── frame_00002_frame_00003_matches.png
│   └── ...
│
└── matches_data/                     매칭 데이터 (NPZ)
   ├── frame_00001_frame_00002_matches.npz
   ├── frame_00002_frame_00003_matches.npz
    └── ...
```

---

## 📊 결과 분석

### 매칭 이미지 보기
```bash
# Windows
start matching_results_integrated\matches_viz\frame_00001_frame_00002_matches.png

# Mac/Linux
open matching_results_integrated/matches_viz/frame_00001_frame_00002_matches.png
```

### 데이터 로드 (Python)
```python
import numpy as np

# 매칭 데이터 로드
data = np.load('matching_results_integrated/matches_data/frame_00001_frame_00002_matches.npz')

matches = data['matches']          # (L, 2) [idx1, idx2]
inlier_mask = data['inlier_mask']  # (L,) 신뢰도 높은 매칭

print(f"매칭: {len(matches)}개")
print(f"Inliers: {np.sum(inlier_mask)}개")

# 특징점 데이터 로드
pts = np.load('matching_results_integrated/features/frame_00001_pts.npy')
desc = np.load('matching_results_integrated/features/frame_00001_desc.npy')
heatmap = np.load('matching_results_integrated/features/frame_00001_heatmap.npy')

print(f"특징점: {pts.shape[1]}개")
print(f"디스크립터: {desc.shape}")
```

---

## ⚙️ 파라미터 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input` | 필수 | 입력: 동영상 또는 이미지 폴더 |
| `--output` | `matching_results_integrated` | 결과 저장 디렉토리 |
| `--weights` | `superpoint_v2_mobilenet.pth` | SuperPoint 가중치 파일 |
| `--nn_thresh` | `0.7` | 매칭 거리 임계값 (0.5-1.0) |
| `--no_cuda` | - | GPU 사용 안 함 (CPU만 사용) |
| `--no_display` | - | 콘솔 출력 비활성화 |

---

## 🎯 일반적인 사용 케이스

### 케이스 1: 빠른 테스트 (이미지 폴더)
```bash
python scripts/integrated_matching.py --input assets/icl_snippet/
```
→ 결과: `matching_results_integrated/` 자동 생성

### 케이스 2: 장시간 비디오 처리
```bash
python scripts/integrated_matching.py --input long_video.mp4 --no_display
```
→ 콘솔 출력 없이 백그라운드에서 조용히 처리

### 케이스 3: GPU 메모리 부족
```bash
python scripts/integrated_matching.py --input video.mp4 --no_cuda
```
→ CPU로 처리 (느리지만 메모리 사용 적음)

### 케이스 4: 높은 정확도 필요
```bash
python scripts/integrated_matching.py --input video.mp4 --nn_thresh 0.6
```
→ 더 엄격한 매칭 (매칭 개수 ↓, 정확도 ↑)

### 케이스 5: 더 많은 매칭 필요
```bash
python scripts/integrated_matching.py --input video.mp4 --nn_thresh 0.9
```
→ 더 관대한 매칭 (매칭 개수 ↑, 정확도 ↓)

---

## 💡 처리 흐름 상세

### Frame 1 처리
```
Frame 1 로드
   ↓
SuperPoint로 특징점 추출 → pts1, desc1
   ↓
저장하고 상태 업데이트
   ↓
다음 프레임으로!
```

### Frame 2 처리
```
Frame 2 로드
   ↓
SuperPoint로 특징점 추출 → pts2, desc2
   ↓
GPU BT-Matcher로 Frame 1과 매칭
   ↓
RANSAC으로 기하학적 검증
   ↓
매칭 이미지 시각화 & 저장
   ↓
특징점 데이터 저장 (Frame 2 → Frame 3으로 사용)
   ↓
다음 프레임으로!
```

### Frame 3, 4, 5... (계속 반복)

---

## 📈 성능

| 항목 | 성능 |
|------|------|
| 특징점 추출 | ~100-200ms per frame (GPU) |
| 매칭 | ~10-50ms per pair (GPU) |
| 총 처리 시간 | ~110-250ms per frame |
| 예상 FPS | ~4-9 FPS (GPU) |

---

## 🎬 출력 예시

실행 중:
```
============================================================
처리 시작: 100 프레임
============================================================

[  1/100] frame_00001: 특징점= 512개, 매칭=   0개 (Inliers:   0), 시간=0.15s, FPS=6.7
[  5/100] frame_00005: 특징점= 487개, 매칭= 345개 (Inliers: 298), 시간=0.18s, FPS=5.6
[ 10/100] frame_00010: 특징점= 523개, 매칭= 398개 (Inliers: 347), 시간=0.17s, FPS=5.9
...
[100/100] frame_00100: 특징점= 505개, 매칭= 378개 (Inliers: 321), 시간=0.16s, FPS=6.2

============================================================
처리 완료!
총 시간: 17.34s
평균 FPS: 5.8
결과 저장: matching_results_integrated
============================================================
```

---

## ❓ FAQ

**Q: 동영상과 이미지 폴더 모두 지원하나?**  
A: ✅ 네, 둘 다 지원합니다!

**Q: 실시간 프리뷰가 나타나나?**  
A: 기본적으로 콘솔에만 출력됩니다. 매칭 이미지는 저장됩니다.

**Q: GPU가 없으면?**  
A: `--no_cuda` 옵션으로 CPU만 사용할 수 있습니다 (느릴 수 있음).

**Q: 매칭 개수가 적으면?**  
A: `--nn_thresh 0.9`로 더 관대하게 설정하세요.

**Q: 매칭 개수가 많으면?**  
A: `--nn_thresh 0.5`로 더 엄격하게 설정하세요.

**Q: 긴 비디오 처리하면?**  
A: `--no_display` 옵션으로 콘솔 출력을 비활성화하면 더 빠릅니다.

---

## 🔧 문제 해결

### 문제: "superpoint_v2_mobilenet.pth를 찾을 수 없음"
```bash
# 가중치 파일 경로 지정
python scripts/integrated_matching.py --input video.mp4 --weights ./superpoint_v2_mobilenet.pth
```

### 문제: "CUDA out of memory"
```bash
# CPU 사용으로 변경
python scripts/integrated_matching.py --input video.mp4 --no_cuda
```

### 문제: "비디오 포맷을 지원하지 않음"
```bash
# 이미지로 변환
# 또는 지원하는 포맷(MP4, AVI, MOV)으로 변환
```

---

## 📚 다음 단계

매칭 결과가 생성되면:
1. ✅ `matches_viz/` 폴더에서 시각화 이미지 확인
2. ✅ `matches_data/` 폴더에서 매칭 데이터 분석
3. ✅ `features/` 폴더에서 특징점 데이터 사용

추가 분석:
- Python에서 데이터 로드하여 3D 삼각측량
- SLAM 백엔드와 통합
- 구조 복원 (Structure from Motion)

---

