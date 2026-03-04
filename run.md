# 3D 모드 (포인트 클라우드 시각화)

python .\scripts\superpoint_app.py --mode 3d --input .\assets\test2.mp4 --weights .\weights\v14_latest.pth --resize 640 480

# 3D 모드 연산량 감소 및 매칭 속도 향상

python .\scripts\superpoint_app.py --mode 3d --input .\assets\test2.mp4 --weights .\weights\v14_latest.pth --resize 640 480 --max_kpts 500 --slam_conf_thresh 0.015 --sp_scale 0.5 --sp_interval 2
