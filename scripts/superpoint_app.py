import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo_runner import add_demo_args, run_demo


def run_slam(opt):
    from slam.visual_slam_3d import VisualSLAM3D

    slam = VisualSLAM3D(
        weights_path=opt.weights,
        input_path=opt.input,
        target_size=(opt.resize[0], opt.resize[1]),
        nn_thresh=opt.nn_thresh,
        mask_car=opt.mask_car,
        conf_thresh=opt.slam_conf_thresh,
        nms_dist=opt.slam_nms_dist,
    )
    slam.process()


def build_parser():
    parser = argparse.ArgumentParser(description="SuperPoint Demo/SLAM 통합 CLI")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["demo", "slam"],
        help="실행 모드 선택 (demo 또는 slam)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help='이미지 디렉토리/비디오/"camera" (demo) 또는 비디오 파일 경로 (slam).',
    )
    parser.add_argument(
        "--weights",
        dest="weights",
        type=str,
        required=False,
        help="SuperPoint 모델 가중치 경로",
    )
    parser.add_argument(
        "--weights_path",
        dest="weights",
        type=str,
        help="(호환) SuperPoint 모델 가중치 경로",
    )

    add_demo_args(parser)

    # SLAM 전용 옵션
    parser.add_argument("--mask_car", action="store_true", help="Enable car masking")
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[640, 480],
        help="Resize input frame to [width height] (slam mode)",
    )
    parser.add_argument(
        "--slam_conf_thresh",
        type=float,
        default=0.003,
        help="SLAM 모드용 conf_thresh (기본: 0.003)",
    )
    parser.add_argument(
        "--slam_nms_dist",
        type=int,
        default=4,
        help="SLAM 모드용 NMS 거리 (기본: 4)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    opt = parser.parse_args()
    if opt.weights is None:
        parser.error("--weights (또는 --weights_path) 가 필요합니다.")

    if opt.mode == "demo":
        run_demo(opt)
    elif opt.mode == "slam":
        run_slam(opt)
