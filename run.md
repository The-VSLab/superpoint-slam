# 2D 모드 (벽 감지 포함)

python .\scripts\superpoint_app.py --mode slam --input .\assets\test2.mp4 --weights .\weights\superpoint_cocoms.pth --filter_floor

# 3D 모드 (포인트 클라우드 시각화)

python .\scripts\superpoint_app.py --mode 3d --input .\assets\test2.mp4 --weights .\weights\superpoint_cocoms.pth --filter_floor

# 2D 비교 모드 (SuperPoint vs ORB)

python .\scripts\superpoint_app.py --mode compare --input .\assets\test2.mp4 --weights .\weights\superpoint_cocoms.pth --filter_floor

# 3D 모드

python scripts/superpoint_app.py --mode 3d --input assets/test2.mp4 --weights weights/v14_latest.pth --config config/default.yaml --use_semantic --yolo-model yolov8n.pt --yolo-conf 0.3 --yolo-classes 0 1 2 3 5 7 --sp-interval 2
