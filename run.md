# 2D 모드 (벽 감지 포함)
python .\scripts\superpoint_app.py --mode slam --input .\assets\test2.mp4 --weights .\checkpoints\v14_final_epoch_20.pth

# 3D 모드 (포인트 클라우드 시각화)
python .\scripts\superpoint_app.py --mode 3d --input .\assets\test2.mp4 --weights .\checkpoints\v14_final_epoch_20.pth

# 2D 비교 모드 (SuperPoint vs ORB)
python .\scripts\superpoint_app.py --mode compare --input .\assets\test2.mp4 --weights .\checkpoints\v14_final_epoch_20.pth