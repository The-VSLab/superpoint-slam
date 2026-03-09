# 3D 실행 예시

## 1 기본 실행

```bash
python scripts/superpoint_app.py --input assets/test2.mp4 --weights weights/v14_latest.pth
```

## 2 권장 실행 (YAML + 시맨틱 필터)

```bash
python scripts/superpoint_app.py --input assets/test2.mp4 --weights weights/v14_latest.pth --config config/default.yaml --use_semantic --sp-interval 2
```

## 출력 위치

- `result/superpoint_3d_XX/final_slam_map.ply`
- `result/superpoint_3d_XX/topdown_map.png`
- `result/superpoint_3d_XX/trajectory_xy.txt`
