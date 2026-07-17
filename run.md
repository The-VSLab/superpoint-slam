# 실행 예시

## 1. KITTI (권장 설정)

```bash
# 도심 시퀀스 (00, 02, 04~10) — 예: 07
./.venv/bin/python scripts/superpoint_app.py \
  --input "dataset/training/07/image_0/%06d.png" \
  --weights weights/v14_desc_ft.pth \
  --calib dataset/training/07/calib.txt \
  --config config/kitti_urban.yaml \
  --resize 1226 370 \
  --no-show_display --no-viz

# 고속도로 (01)
./.venv/bin/python scripts/superpoint_app.py \
  --input "dataset/training/01/image_0/%06d.png" \
  --weights weights/v14_desc_ft.pth \
  --calib dataset/training/01/calib.txt \
  --config config/kitti_highway.yaml \
  --resize 1241 376 \
  --no-show_display --no-viz
```

### 시퀀스별 `--resize` (원본 해상도 사용 — 자동 640 축소는 정확도/루프 저하)

| 시퀀스 | resize | config |
|--------|--------|--------|
| 00–02 | `1241 376` | urban (01만 highway) |
| 03 | `1242 375` | urban |
| 04–10 | `1226 370` | urban |

## 2. 일반 동영상

```bash
./.venv/bin/python scripts/superpoint_app.py \
  --input your_video.mp4 \
  --weights weights/v14_desc_ft.pth \
  --config config/default.yaml --use_semantic
```

## 3. 실시간 화면 보기

`--no-show_display --no-viz`를 빼면 특징점 오버레이 창이 뜬다 (프레임당 ~12ms 비용).
디버그 로그(루프 클로저 유사도 등)는 `-v`.

## 옵션 메모

- `--input`은 이미지 시퀀스면 반드시 `"...%06d.png"` 패턴 (쌍따옴표 필수)
- `--calib` 지정 시 yaml의 `focal_multiplier`는 무시됨 (권장)
- 가중치: `weights/v14_desc_ft.pth` 권장 — `weights/v14_latest.pth`는 디스크립터 붕괴로 루프 클로저 불가 (트래킹만 가능)

## 출력 (`result/superpoint_3d_XX/`)

| 파일 | 내용 |
|------|------|
| `final_slam_map.ply` | 3D 포인트클라우드 |
| `topdown_map.png` | 탑다운 맵 + 궤적 |
| `trajectory_xyz.txt` | 프레임별 궤적 (실시간 기록, PGO 미반영) |
| `trajectory_kf.txt` | **post-PGO 키프레임 궤적 (frame_idx x y z) — ATE 평가는 이걸 사용** |
| `slam_stats.json` | 지연시간/맵/inlier 통계 |

## ATE 평가

```bash
python scripts/evaluate.py   # 또는:
python - <<'EOF'
import numpy as np
gt = np.loadtxt('dataset/07.txt').reshape(-1,3,4)[:,:,3]
kf = np.loadtxt('result/superpoint_3d_XX/trajectory_kf.txt')
# Umeyama(Sim3) 정렬 후 RMSE — scripts/evaluate.py의 umeyama_alignment 참고
EOF
```

## 확인할 로그

- `[LOOP FOUND (PnP)] ...` — metric 루프 검출 (07 등 같은 방향 재방문 시퀀스)
- `[Pose Graph] no loop edges - skipping PGO` — 루프 없는 시퀀스(01, 08)의 정상 동작
- 종료 시 `Loop Closure stats:` — max_sim / 기각 사유별 카운트
