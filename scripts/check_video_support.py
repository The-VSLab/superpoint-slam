import cv2
import sys

print("=== OpenCV 비디오 포맷 지원 테스트 ===\n")

# OpenCV 버전
print(f"OpenCV 버전: {cv2.__version__}")
print(f"Python 버전: {sys.version}\n")

# 지원하는 백엔드 확인
print("⚙️  설치된 백엔드:")
backends = [
    ("FFMPEG", cv2.CAP_FFMPEG),
    ("V4L2 (Linux)", cv2.CAP_V4L2),
    ("DSHOW (Windows)", cv2.CAP_DSHOW),
    ("MSMF (Windows)", cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW),
]

for name, backend_id in backends:
    try:
        cap = cv2.VideoCapture(0, backend_id)
        is_available = cap.isOpened()
        cap.release()
        status = "✓ 사용 가능" if is_available else "✗ 미지원"
    except:
        status = "✗ 오류"
    print(f"  {name}: {status}")

print("\n📹 지원 형식 (일반적):")
formats = [
    "MP4 (.mp4) - 권장",
    "AVI (.avi)",
    "MOV (.mov)",
    "MKV (.mkv)",
    "FLV (.flv)",
    "YUV (.yuv)",
]
for fmt in formats:
    print(f"  • {fmt}")

print("\n이미지 시퀀스도 지원:")
print("  • %06d.png 형식")
print("  • %05d.jpg 형식")
print("  • 프레임 번호는 0부터 시작")

print("\n예시:")
print("# 동영상 입력")
print("python scripts/compare_superpoint_orb_2d.py \\")
print("  --input video.mp4 \\")
print("  --weights checkpoints/v14_final_epoch_20.pth")
print()
print("# 이미지 시퀀스 입력")
print("python scripts/compare_superpoint_orb_2d.py \\")
print("  --input 'frames/%06d.png' \\")
print("  --weights checkpoints/v14_final_epoch_20.pth")
