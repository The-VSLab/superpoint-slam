# VSLab MobileNet-SuperPoint SLAM 통합 기술 문서

> ## ⚠️ 2026-07 업데이트 (필독)
> 이 문서 작성 이후 다음이 확인/변경되었습니다. 상충하는 내용은 아래가 우선합니다.
>
> 1. **`v14_latest.pth`의 디스크립터 헤드 붕괴 확인** — 검출/트래킹은 정상이나
>    기술자 매칭이 불가능한 상태 (인접 프레임 inlier 7%). 루프 클로저를 쓰려면
>    **`weights/v14_desc_ft.pth`**(디스크립터 헤드 재증류본)를 사용할 것.
>    복구 방법: `learning/finetune_descriptor.py` ([TRAINING_KO.md](TRAINING_KO.md) 참고)
> 2. **루프 클로저 전면 개편** — centered mean-pooling 검출 + PnP 우선 검증 +
>    **Sim3(7-DOF) PGO**. KITTI 07에서 루프 검출(PnP 92 inliers, ATE 13.2m),
>    08에서 오탐 0건 실측 검증.
> 3. **config 이름 변경** — `kitti08.yaml`→`config/kitti_urban.yaml`,
>    `kitti01.yaml`→`config/kitti_highway.yaml`
> 4. **BTMatcher는 CPU 사용** (MPS가 6배 느림 — 실측), 자동 리사이즈 대신
>    시퀀스 원본 해상도 `--resize` + `--calib` 권장. 실행법: [run.md](../run.md)

목표는 다음 3가지입니다.

1. 프로젝트의 기술 구조를 한눈에 이해할 수 있게 정리
2. `v14_latest.pth` 학습 배경과 성능 개선 포인트를 명확히 문서화
3. 실제 실행, 실험, 확장 시 참고할 수 있는 실무형 가이드 제공

---

## 1. 프로젝트 개요

VSLab은 기존 ORB-SLAM 계열 파이프라인의 프론트엔드를 경량 딥러닝 기반 특징점 추출기로 대체한 Visual SLAM 프로젝트입니다.

- 기존 ORB 기반 프론트엔드의 약점:
  - 저텍스처 환경에서 특징점 불안정
  - 조명 변화, 모션 블러 환경에서 강인성 저하
- 해결 전략:
  - SuperPoint 계열의 학습 기반 특징점/기술자 추출기 도입
  - 백본을 VGG에서 MobileNetV2로 변경해 경량화

핵심 컨셉은 다음과 같습니다.

- 프론트엔드(특징점 추출, 기술자 생성, 매칭)는 개선
- 백엔드(Tracking, Pose Estimation, Mapping, Optimization)는 최대한 유지

즉, 기존 SLAM 시스템의 안정적인 후단 구조를 살리면서, 환경 변화에 강한 입력 특징을 공급하는 방식입니다.

---

## 2. 시스템 구조 요약

### 2.1 주요 구성 요소

- 특징점 추출기: MobileNet 기반 SuperPoint (`frontend/superpoint_frontend.py`)
- 매칭 모듈: 실수형 기술자 매칭 (`matcher_module/btmatcher.py`)
- 동적 객체 필터: YOLO 기반 Semantic Filtering (`tracking/semantic_filter.py`)
- 포인트 품질 필터: 하늘/그림자/불안정 포인트 제거 (`tracking/point_filter.py`)
- SLAM 본체:
  - 3D Visual SLAM (`slam/visual_slam_3d.py`)
  - 루프 클로저 (`slam/loop_closure.py`)

### 2.2 전체 데이터 흐름

입력 영상
-> 전처리(CLAHE, 차체 마스킹)
-> SuperPoint(MobileNetV2) 특징점/기술자 추출
-> Semantic + Point Filtering
-> 프레임 간 추적(LK Optical Flow) 및/또는 디스크립터 매칭
-> 포즈 추정(PnP + 에피폴라 보조)
-> 삼각측량 기반 3D 포인트 생성
-> 키프레임 관리/루프 클로저/그래프 최적화
-> 궤적 및 포인트클라우드 출력(Open3D, PLY)

---

## 3. 파이프라인 상세

### 3.1 전처리 (Preprocessing)

목적은 입력 품질 향상과 추적 방해 요소 제거입니다.

- CLAHE:
  - `cv2.createCLAHE` 사용
  - 어두운 구간, 역광 환경에서 국소 대비를 높여 특징점 검출률을 개선
- Dashboard/보닛 마스킹:
  - 차량 하단 고정 구조물 영역을 제거
  - 실제 환경 변화와 무관한 잡음 특징점 유입 억제

### 3.2 프론트엔드 (Feature Extraction and Tracking)

### 특징점 추출

- 모델: SuperPoint + MobileNetV2 백본
- 출력:
  - Detector Head: 코너/키포인트 확률 맵
  - Descriptor Head: 256차원 실수형 기술자

### 필터링

- Semantic Filtering (YOLO):
  - 보행자, 차량 등 동적 객체 위 포인트 제거
- Point Filtering:
  - 하늘/구름/그림자 등 기하학적으로 불안정한 포인트 제거

### 추적

- `cv2.calcOpticalFlowPyrLK` 기반 LK Optical Flow로 인접 프레임 포인트 이동 추적
- 필요 시 디스크립터 기반 매칭(BTMatcher) 병행

### 3.3 포즈 추정 (Pose Estimation)

- 기본: `cv2.solvePnPRansac`
  - 3D 맵 포인트 <-> 2D 관측점 대응으로 카메라 포즈 추정
- 보조: Essential/Homography 분해
  - 초기화 구간 또는 PnP 실패 시 대체 경로로 사용
- 안정화:
  - 차량 주행 특성 반영
  - Y축 이동, Pitch/Roll 회전에 댐핑 적용해 드리프트 완화

### 3.4 매핑/백엔드

- Triangulation: `cv2.triangulatePoints`
- Keyframe Management:
  - 모든 프레임 저장 대신 시각 변화 큰 프레임 중심 관리
- Loop Closure:
  - 재방문 구간 인식 후 누적 오차 보정 (`LoopClosureManager`)
- Graph Optimization:
  - g2o 기반 Sim3 Pose Graph Optimization
  - 스케일 불일치 및 장기 궤적 누적 오차 완화

### 3.5 시각화/출력

- Open3D 기반 포인트클라우드/카메라 궤적 시각화
- 결과 맵 `PLY` 저장
- 궤적 텍스트/통계 JSON 형태 출력 지원

---

## 4. `v14_latest.pth` 학습 전략 상세

`v14_latest.pth`는 MobileNetV2 기반 Student가 Original SuperPoint(VGG 기반 Teacher)의 표현력을 최대한 계승하도록 설계된 2단계 지식 증류 학습 결과물입니다.

### 4.1 학습 목표

- 경량화: 임베디드/실시간 가능성 확보
- 성능 유지: 원본 SuperPoint 수준의 강인한 특징점 품질 유지
- 안정화: 백본 전이학습 과정에서 Catastrophic Forgetting 방지

### 4.2 학습 구성

- Student: MobileNetV2 (ImageNet pretrained)
- Teacher: Original SuperPoint (VGG-based)
- 스크립트: `learning/train_superpoint_v13.py`
- 설정: `learning/config_v13.yml`
- 데이터: KITTI Odometry 00~10 (`dataset/training`)

### 4.3 2-Phase Training

#### Phase 1: Head 안정화 (Backbone Freeze)

- 설정:
  - `freeze_backbone: true`
  - LR: `1.0e-4`
  - Epoch: `20`
- 목적:
  - 랜덤 초기화된 Detector/Descriptor Head가 백본 사전학습 표현을 깨뜨리는 현상 방지
- 결과:
  - 안정적인 중간 체크포인트 확보 (`v15_depthwise_resume.pt`)

#### Phase 2: 전체 미세조정 (Fine-tuning)

- 설정:
  - `freeze_backbone: false`
  - LR: `1.0e-5`
  - Epoch: `30`
- 핵심 기법: BN Defense
  - 백본 학습은 허용하되 BatchNorm은 `eval()` 상태로 고정
  - 주행 데이터 통계 편향으로 BN 통계가 무너지는 문제 방지
- 결과:
  - 최종 `v14_latest.pth` 생성
  - 신뢰도 지표가 70% 이상 수준으로 크게 개선

### 4.4 주요 하이퍼파라미터

| 항목          | 값     | 의미                           |
| ------------- | ------ | ------------------------------ |
| `batch_size`  | 16     | 미니배치 크기                  |
| `lr`          | 1.0e-5 | Phase 2 학습률                 |
| `det_weight`  | 1.0    | Detector loss 가중치           |
| `desc_weight` | 2.0    | Descriptor loss 가중치         |
| `sup_weight`  | 25.0   | Teacher supervised loss 가중치 |
| `max_rotate`  | 15.0   | 데이터 증강 최대 회전 각도     |

---

## 5. 성능 관점 정리

README 기준 비교 요약:

- ORB-SLAM baseline:
  - FPS는 높지만 환경 변화 대응력 한계
- Original SuperPoint(VGG):
  - 정확도 및 강인성 우수
  - 연산량이 커서 임베디드 실시간 부적합
- 제안 모델(MobileNet-SuperPoint, Phase 2):
  - 정확도-속도 균형점 확보
  - 대략 5~9 FPS(파이썬 기준) 범위에서 실용성 확보
  - C++/TensorRT 최적화 시 실시간성 확장 여지 큼

핵심 포인트는 절대 최고 정확도보다, 실제 운용 가능한 연산 비용에서 높은 강인성을 달성했다는 점입니다.

---

## 6. 실행 가이드

### 6.1 환경 설치

프로젝트 루트 기준:

```bash
# 권장
uv sync

# 또는
pip install -r requirements.txt
```

### 6.2 CLI 실행

```bash
# 3D SLAM (기본)
uv run python scripts/superpoint_app.py --input <VIDEO_PATH> --weights <WEIGHTS_PATH> --config config/default.yaml

# 3D SLAM (권장)
uv run python scripts/superpoint_app.py --input <VIDEO_PATH> --weights <WEIGHTS_PATH> --config config/default.yaml --use_semantic
```

출력:

- 실행 결과 디렉토리: `result/superpoint_3d_XX/`
- 포인트 클라우드: `final_slam_map.ply`
- 탑다운 맵: `topdown_map.png`
- 2D 궤적: `trajectory_xy.txt`

---

## 7. 코드 구조 빠른 참조

주요 디렉터리 역할:

- `models/`: SuperPoint MobileNet 모델 정의
- `frontend/`: 특징점 추출 프론트엔드
- `tracking/`: 추적/필터링 모듈
- `matcher_module/`: 디스크립터 매칭 로직
- `slam/`: 3D SLAM, 루프클로저, 맵 관리
- `learning/`: 학습 스크립트 및 학습 설정
- `scripts/`: 실행 진입점 및 평가 스크립트
- `weights/`: 학습된 모델 가중치

---

## 8. 설계 강점

- 백엔드 재사용:
  - 기존 SLAM 구조를 크게 바꾸지 않아 통합 리스크를 낮춤
- 프론트엔드 강인성:
  - 저조도/저텍스처/블러 환경에서 전통 특징점 대비 유리
- 실용적 경량화:
  - 완전 고성능 모델보다 실시간 가능성에 초점
- 다층 방어적 추적:
  - Semantic + Point Filtering + 모션 안정화 조합으로 오인식 억제

---

## 9. 한계 및 향후 개선

현재 문서 기준 한계:

- 루프 클로저/재지역화 성능은 매칭 전략에 민감
- 초저전력 타깃을 위해 양자화/추가 경량화 필요
- ATE 등 절대 정확도 평가는 추가 실험으로 보완 필요

권장 후속 과제:

1. TensorRT/ONNX 최적화 경로 정립
2. 다양한 기상/야간/고속 주행 데이터셋 교차 검증
3. 루프클로저 강화를 위한 글로벌 디스크립터 병합
4. Online calibration 및 실패 복구 전략 고도화

---

## 10. 결론

VSLab의 MobileNet 기반 SuperPoint SLAM 접근은 다음을 동시에 달성하려는 현실적인 설계입니다.

- 학습 기반 특징점의 강인성
- 기존 SLAM 파이프라인과의 통합 용이성
- 임베디드/실시간 적용 가능성

특히 2단계 증류 학습 전략은 단순 경량화로 인한 성능 하락을 억제하면서도
실제 운용 가능한 속도 영역으로 모델을 끌어온 핵심 요소로 평가할 수 있습니다.
(단, `v14_latest.pth`는 이후 디스크립터 헤드 붕괴가 확인되어 — 문서 상단 업데이트 참고 —
현재 권장 가중치는 디스크립터를 재증류한 `weights/v14_desc_ft.pth`입니다.)

---

## 부록 A. 관련 문서

- 프로젝트 개요/실행: `README.md`
- 실행 예시: `run.md`
- 파이프라인 설명: `docs/INTEGRATED_PIPELINE_KO.md`
- 학습 상세: `docs/TRAINING_KO.md`

## 부록 B. 핵심 파일 참조

- `frontend/superpoint_frontend.py`
- `matcher_module/btmatcher.py`
- `slam/visual_slam_3d.py`
- `slam/loop_closure.py`
- `learning/train_superpoint_v13.py`
- `learning/config_v13.yml`
