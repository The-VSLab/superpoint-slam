# 🚀 빠른 시작 가이드 (5분)

> **가중치 안내**: 기본 권장 가중치는 `weights/v14_desc_ft.pth`입니다.
> `weights/v14_latest.pth`는 디스크립터 헤드 붕괴로 루프 클로저가 동작하지 않습니다 (트래킹은 가능).
> KITTI 실행법(시퀀스별 `--calib`/`--resize`)은 [run.md](../run.md) 참고.

## 폴더 구조 한눈에

```
superpoint-slam/
├── 📁 scripts/
│   ├── superpoint_app.py        ← 메인 3D SLAM 진입점
│   └── evaluate.py              ← 평가 및 Diff.csv 생성 스크립트
├── 📁 slam/                     ← SLAM 모듈 (visual_slam_3d.py)
├── 📁 matcher_module/           ← L2 디스크립터 매칭 모듈
└── 📁 result/                   ← 출력: 매칭 및 3D SLAM 결과 
```

---

## 3단계: 실행하기

### 단계 1️⃣: 환경 설정

본 프로젝트 루트 맵핑을 위해 다음을 실행합니다. 권장 방식인 `uv sync`를 통해 패키지를 설치합니다.

```bash
uv sync
```

### 단계 2️⃣: 시스템 테스트 (3D SLAM 실행)

동영상 파일을 이용해 SuperPoint SLAM을 즉시 실행합니다. GPU 가속(CUDA/MPS)이 자동으로 활성화됩니다.

```bash
uv run python scripts/superpoint_app.py --input assets/test2.mp4 --weights weights/v14_desc_ft.pth --config config/default.yaml
```

✅ 결과: `result/superpoint_3d_01/` 폴더에 파일 생성
- `final_slam_map.ply` (3D 포인트 클라우드)
- `topdown_map.png` (2D 궤적 결과)
- `trajectory_xy.txt` (포즈 정보)

---

## 💡 자주 쓰는 명령어

### 권장 실행 (YAML + 시맨틱 필터)
```bash
uv run python scripts/superpoint_app.py --input assets/test2.mp4 --weights weights/v14_desc_ft.pth --config config/default.yaml --use_semantic --sp-interval 2
```

### 그림자 필터를 최대로 적용하여 스케일 관리를 최적화
```bash
uv run python scripts/superpoint_app.py --input assets/test2.mp4 --weights weights/v14_desc_ft.pth --config config/default.yaml --aggressive_shadow_filter
```

---

## 📊 결과 평가하기

실행을 마치면 저장된 `result/superpoint_3d_XX/` 경로를 이용해 성능과 메모리 평가를 수행할 수 있습니다.

```bash
uv run python scripts/evaluate.py --result result/superpoint_3d_01 --csv diff.csv
```

✅ 결과로 SLAM 성능 평가 보고서(Latency, Memory, 프레임 통계 등)가 출력되며 `diff.csv`에 기록됩니다.

---

## 🔧 문제 해결

| 문제 | 해결책 |
|------|--------|
| `ModuleNotFoundError` | 현재 디렉토리가 프로젝트 루트인지 확인하거나 `uv run` 환경인지 확인 |
| 특징점이 부족함 | `--slam_conf_thresh 0.005` 같이 임계값을 낮춰 더 많은 포인트를 추출 |
| 특징점이 그림자에 맺힘 | `--aggressive_shadow_filter` 옵션을 추가 |

---

## 📚 다음 단계

결과 도출 후 할 수 있는 것들:
- ✅ Open3D로 `final_slam_map.ply` 포인트클라우드 불러오기
- ✅ `evaluate.py` 결과를 기반으로 파라미터(`config/default.yaml`) 최적화
- ✅ `docs/TRAINING_KO.md`를 참고하여 새로운 백본 가중치 학습

---
