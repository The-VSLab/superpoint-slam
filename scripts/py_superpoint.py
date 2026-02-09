import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo_runner import add_demo_args, run_demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch SuperPoint V2 데모 (MobileNet 백본).")
    parser.add_argument(
        "input", type=str, default="", help='이미지 디렉토리 또는 비디오 파일 또는 "camera" (웹캠용).'
    )
    parser.add_argument(
        "--weights",
        dest="weights",
        type=str,
        default=None,
        help="사전 학습된 가중치 파일 경로 (선택사항, None이면 사전 학습된 MobileNet 사용).",
    )
    parser.add_argument(
        "--weights_path",
        dest="weights",
        type=str,
        help="(호환) 사전 학습된 가중치 파일 경로",
    )

    add_demo_args(parser)

    opt = parser.parse_args()
    print(opt)

    run_demo(opt)
