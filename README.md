# MobileNet 기반 SuperPoint SLAM 프론트엔드

## 초록 (Abstract)

본 프로젝트는 SuperPoint 기반의 시각 SLAM 프론트엔드를 경량화하기 위해,
기존의 연산량이 큰 VGG 계열 백본을 MobileNet 구조로 대체한
경량 SuperPoint 프론트엔드(0.16M 파라미터, 0.59MB) 기반의 Python 3D 시각 SLAM 시스템을 제안한다.
시스템은 Optical Flow 기반 추적, g2o Bundle Adjustment,
그리고 **Sim3(7-DOF) Pose Graph Optimization 기반 루프 클로저**를 갖춘
Python 친화적인 독자 SLAM 백엔드를 포함한다.
KITTI odometry 벤치마크에서 루프 검출(시퀀스 07: PnP 92 inliers)과
오탐 방지(시퀀스 08: 후보 460개 전원 기각)를 실측으로 검증하였다.

---

## 1. 서론 (Introduction)

ORB-SLAM과 같은 기존 시각 SLAM 시스템은 ORB와 같은 수작업 특징점에
의존하고 있으며, 이는 연산 효율은 높지만 저텍스처 환경, 조명 변화,
모션 블러 등 도전적인 시각 환경에서는 성능 저하가 발생한다.

SuperPoint는 딥러닝 기반 특징점 검출 및 기술자 추출 기법으로
이러한 환경 변화에 강인하지만, 원본 VGG 백본은 임베디드·실시간
SLAM에 적용하기에 연산량이 과도하다. 본 프로젝트는 MobileNet 기반
경량 SuperPoint 프론트엔드와 Python 백엔드 파이프라인(Optical Flow
트래킹 + g2o BA/Sim3 PGO)을 결합하여 강인성과 실시간성을 동시에 확보한다.

---

## 2. 환경 설정 및 실행 (Setup & Execution)

### 2.1 설치

```bash
# 방법 A: uv (권장)
uv sync

# 방법 B: pip
pip install -r requirements.txt
```

프로젝트 루트에서 실행하거나 `export PYTHONPATH=.`를 설정한다.

### 2.2 가중치

| 파일 | 용도 |
|------|------|
| `weights/v14_desc_ft.pth` | **권장** — v14 검출 헤드 + 재증류된 디스크립터 헤드 (루프 클로저 동작) |
| `weights/v14_latest.pth` | 검출/트래킹 전용 — **디스크립터 헤드 붕괴** 확인됨(인접 프레임 매칭 inlier 7%), 루프 클로저 불가 |
| `weights/superpoint_v1.pth` | 원본 SuperPoint(VGG) — 디스크립터 재증류의 teacher |

### 2.3 KITTI 실행

```bash
./.venv/bin/python scripts/superpoint_app.py \
  --input "dataset/training/07/image_0/%06d.png" \
  --weights weights/v14_desc_ft.pth \
  --calib dataset/training/07/calib.txt \
  --config config/kitti_urban.yaml \
  --resize 1226 370 \
  --no-show_display --no-viz
```

핵심 옵션:

- `--input`은 디렉토리가 아닌 **`%06d.png` 패턴** (cv2.VideoCapture 이미지 시퀀스)
- `--calib`: 시퀀스별 `calib.txt`의 P0 intrinsics 사용 (시퀀스 그룹마다 fx가 다르므로 필수 권장)
- `--resize`: 시퀀스 그룹별 원본 해상도 사용 (자동 640 축소는 정확도·루프 검출 모두 저하, 속도 이득 없음 — 실측)

| 시퀀스 | 원본 해상도 | fx | config |
|--------|-------------|-----|--------|
| 00–02 | `--resize 1241 376` | 718.86 | `kitti_urban.yaml` |
| 03 | `--resize 1242 375` | 721.54 | `kitti_urban.yaml` |
| 04–10 | `--resize 1226 370` | 707.09 | `kitti_urban.yaml` (01만 `kitti_highway.yaml`) |

출력 (`result/superpoint_3d_XX/`): `final_slam_map.ply`, `topdown_map.png`,
`trajectory_xyz.txt`(프레임별), **`trajectory_kf.txt`(post-PGO 키프레임, ATE 평가용: frame_idx x y z)**

---

## 3. 제안 방법 (Methodology)

### 3.1 네트워크 구조

- VGG 백본 → **MobileNetV2 백본** (ImageNet 사전학습, 1/8 해상도 지점까지 사용)
- Depthwise Separable Conv 기반 Detector Head (65채널) / Descriptor Head (256차원)
- 총 0.16M 파라미터, 0.59MB

### 3.2 디스크립터 헤드 재증류 (Descriptor Distillation)

학습 과정에서 디스크립터 헤드가 붕괴하는 사고(모든 키포인트의 기술자가
사실상 동일 벡터로 수렴, 프레임 내 유사도 p95=0.999)가 발생할 수 있다.
이를 복구하기 위해 `learning/finetune_descriptor.py`는 백본·검출 헤드를
동결한 채 디스크립터 헤드(74.6K 파라미터)만 원본 SuperPoint teacher로부터
재증류한다 (KITTI 이미지 6천 장, 2 epochs, 약 2분).

| 지표 | 붕괴 상태 (v14_latest) | 재증류 후 (v14_desc_ft) | Teacher |
|------|------------------------|--------------------------|---------|
| 인접 프레임 매칭 inlier | 7% | **89%** | 96% |
| 진짜 루프쌍 inlier | 7% (구분 불가) | **74%** | 83% |
| 프레임 내 자기유사도(중앙값) | 0.618 | 0.04 | 0.02 |

---

## 4. 시스템 통합 (System Integration)

### 4.1 파이프라인

```
입력 영상
→ CLAHE + ROI/시맨틱 마스킹
→ MobileNet-SuperPoint (특징점 + 256D 기술자)
→ Optical Flow 트래킹 (+ 로컬 맵 재투영 매칭)
→ PnP RANSAC 포즈 추정 → 포즈 안정화 → 키프레임 선정
→ 삼각측량 + MapPoint 관리
→ 루프 클로저 검출/검증 → Sim3 Pose Graph Optimization
→ 3D 포인트클라우드 + 궤적 출력
```

### 4.2 루프 클로저 (Loop Closure)

1. **검출**: 프레임별 기술자의 mean-pooling 벡터를 저장하고, 질의 시
   저장분 평균(μ)을 빼는 **centering** 후 코사인 유사도로 후보 검색.
   (max-pooling은 포화로 모든 쌍이 유사도 1.0이 되어 사용 불가 — 실측)
2. **검증**: top-k 후보에 대해 **PnP(3D-2D, metric) 우선** — 실패 시
   Essential Matrix(회전만 신뢰) 폴백. 임계값(`verify_min_inliers: 25`,
   `ratio: 0.30`)은 KITTI 실측(진짜 루프 92 vs 가짜 9–14 inliers)으로 설정.
3. **교정**: g2o **Sim3(7-DOF) PGO** — 단안 스케일 드리프트까지 교정.
   PnP 루프 엣지는 회전+병진+스케일, Essential 엣지는 회전만 반영.
   루프용 3D 쌍은 추적 검증을 통과한 클린 소스에서 기하 매칭으로 구성.

---

## 5. 실험 결과 (Experimental Results) — KITTI odometry 실측

평가: Umeyama(Sim3) 정렬 ATE, `trajectory_kf.txt`(post-PGO 키프레임) 기준.
환경: Apple Silicon(MPS), Python 3.11.

| 시퀀스 | 특성 | ATE RMSE | 루프 클로저 | FPS |
|--------|------|----------|-------------|-----|
| **07** (695m, 루프 O) | 도심, 같은 방향 재방문 | **13.2–14.4 m** | **PnP 검출 성공** (92 inliers), 폐합 확인 | 15–17 |
| **08** (3.2km, 루프 X) | 도심, 역주행 재방문만 존재 | **45.3–48.3 m** | 0건 = 정답 (후보 460개 전원 기각, **오탐 0**) | 15.5 |

- 루프 클로저 효과 (07): 시작-끝 폐합 오차 38.5m → 25.4m, ATE 최대 17.2 → 13.2m
- 08의 오차는 루프 보정이 불가능한 순수 odometry 드리프트 (경로 대비 ~1.4%)
- 시퀀스 01(고속도로)은 재방문이 없어 루프 클로저 미발화가 정상

### 성능 최적화 노트 (실측)

- 기술자 매칭(BTMatcher)은 **CPU가 MPS보다 6배 빠름** (소규모 행렬의 커널
  디스패치 오버헤드) — CUDA에서만 GPU 매칭 사용
- 시각화 비활성(`--no-show_display --no-viz`) 시 프레임당 ~12ms 절약
- 자동 리사이즈(640px)는 속도 이득 없이 ATE +41%, 루프 검출 실패 → 원본 해상도 권장

---

## 6. 학습 (Training)

### 6.1 디스크립터 헤드 재증류 (권장 진입점)

```bash
./.venv/bin/python learning/finetune_descriptor.py --epochs 2 --batch_size 8 --stride 3
# 출력: weights/v14_desc_ft.pth (검출 헤드는 v14_latest와 동일 유지)
```

- 데이터: `dataset/training/<seq>/image_0` (KITTI, 라벨 불필요 — teacher 증류)
- 평가 시퀀스(07, 08)는 학습에서 자동 제외

### 6.2 전체 학습 (teacher-student)

```bash
./.venv/bin/python learning/train_superpoint.py --config learning/train_config.yml
```

주의: `use_teacher_desc: true`인데 teacher가 desc를 반환하지 않으면
디스크립터 헤드가 붕괴한다 — 현재 코드는 이 경우 **에러를 발생**시킨다.
학습 로그의 `desc=` 손실이 0으로 유지되면 즉시 중단하고 원인을 확인할 것.
자세한 내용은 `docs/TRAINING_KO.md` 참고.

---

## 7. 한계점 및 향후 연구 (Limitations & Future Work)

- **MapPoint 연관 딕셔너리의 인덱스 네임스페이스 혼입** — flow/SP 인덱스가
  섞여 관측 기록·기술자 갱신·컬링의 정확도를 낮춤 (추적 outlier ~20%의
  원인으로 추정). 루프 클로저는 클린 소스로 우회 완료, 근본 수정은 진행 예정.
- Global BA 비활성 상태 — 재활성화 시 odometry 정확도 개선 여지
- 루프 엣지가 시퀀스당 1개 수준 — 다중 루프(KITTI 00/05) 검증 필요
- RANSAC 비결정성으로 런 간 ATE 편차 존재 (07: ±1m, 08: ±3m 수준)
- 검출 결과는 런마다 다를 수 있어 다중 런 평균으로 보고할 것을 권장

---

## 8. 결론 (Conclusion)

본 프로젝트는 0.59MB의 경량 딥러닝 프론트엔드로 15+ FPS의 Python 단안
SLAM을 구동하면서, Sim3 PGO 기반 루프 클로저의 검출(07)과 오탐 방지(08)를
KITTI 실측으로 입증하였다. 경량성 대비 정확도(07 ATE 13.2m)의 균형은
임베디드 환경에서의 SLAM 적용 가능성을 보여준다.

---

## 9. 라이선스 (License)

본 프로젝트는 MIT License 하에 배포된다.

---

## 10. 감사의 글 (Acknowledgements)

- DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPR Workshop, 2018.
- Mur-Artal et al., "ORB-SLAM: A Versatile and Accurate Monocular SLAM System".

본 저장소는 ORB-SLAM 코드를 포함하지 않는다.
