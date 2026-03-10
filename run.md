# 3D 실행 예시

## 1 기본 실행

```bash
python scripts/superpoint_app.py --input your_video.mp4 --weights weights/v14_latest.pth
```

## 2 권장 실행 (YAML + 시맨틱 필터)

```bash
python scripts/superpoint_app.py --input your_video.mp4 --weights weights/v14_latest.pth --config config/default.yaml --use_semantic --sp-interval 2
```

## 3 최적 실행 (밸런스 및 그림자 필터 강화)

```bash
uv run python scripts/superpoint_app.py --input your_video.mp4 --weights weights/v14_latest.pth --config config/default.yaml --use_semantic --sp-interval 2 --aggressive_shadow_filter --slam_conf_thresh 0.01
```

## 출력 위치

- `result/superpoint_3d_XX/final_slam_map.ply`
- `result/superpoint_3d_XX/topdown_map.png`
- `result/superpoint_3d_XX/trajectory_xy.txt`
