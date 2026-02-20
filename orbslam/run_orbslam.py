import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from orbslam.orb_slam_2d import ORBSLAM2D


def build_parser():
    parser = argparse.ArgumentParser(description="ORB 기반 2D SLAM 실행기")
    parser.add_argument("--input", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--resize", nargs=2, type=int, default=[640, 480], help="입력 리사이즈 [width height]")
    parser.add_argument("--nfeatures", type=int, default=1200, help="ORB 특징점 수")
    parser.add_argument("--max_matches", type=int, default=500, help="프레임당 최대 매칭 수")
    parser.add_argument("--motion_scale", type=float, default=1.0, help="프레임당 이동 스케일")
    parser.add_argument("--output_dir", type=str, default="results_orbslam_2d", help="결과 저장 디렉토리")
    parser.add_argument("--no_display", action="store_true", help="실시간 화면 비활성화")
    return parser


def main():
    opt = build_parser().parse_args()

    slam = ORBSLAM2D(
        input_path=opt.input,
        resize=tuple(opt.resize),
        nfeatures=opt.nfeatures,
        max_matches=opt.max_matches,
        motion_scale=opt.motion_scale,
        output_dir=opt.output_dir,
        show_display=not opt.no_display,
    )

    stats = slam.process()
    print("\n==> ORB 2D SLAM Summary")
    print(f"frames={stats.frames}")
    print(f"avg_extract_ms={stats.avg_extract_ms:.3f}")
    print(f"avg_match_ms={stats.avg_match_ms:.3f}")
    print(f"avg_total_ms={stats.avg_total_ms:.3f}")
    print(f"avg_inlier_ratio={stats.avg_inlier_ratio:.3f}")
    print(f"trajectory_length={stats.trajectory_length:.3f}")
    print(f"output_dir={stats.output_dir}")


if __name__ == "__main__":
    main()
