import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo_runner import add_demo_args, run_demo


def run_slam2d(opt):
    from slam.visual_slam_2d import VisualSLAM2D

    slam = VisualSLAM2D(
        weights_path=opt.weights,
        input_path=opt.input,
        nn_thresh=opt.nn_thresh,
        resize=tuple(opt.resize),
        conf_thresh=opt.slam_conf_thresh,
        nms_dist=opt.slam_nms_dist,
        mask_car=opt.mask_car,
        motion_scale=opt.motion_scale,
        output_dir=opt.output_dir,
        show_display=not opt.no_display,
        sp_scale=opt.sp_scale,
        sp_interval=opt.sp_interval,
        sp_fp16=opt.sp_fp16,
        max_kpts=opt.max_kpts,
        uniform_grid=tuple(opt.uniform_grid),
        use_subpixel_refine=opt.use_subpixel_refine,
        use_uniform_distribution=opt.use_uniform_distribution,
        use_hybrid_matching=opt.use_hybrid_matching,
        ratio_thresh=opt.ratio_thresh,
        ransac_thresh=opt.ransac_thresh,
        com_radius=opt.com_radius,
    )
    stats = slam.process()
    print("\n==> SuperPoint 2D SLAM Summary")
    print(f"frames={stats.frames}")
    print(f"avg_extract_ms={stats.avg_extract_ms:.3f}")
    print(f"avg_match_ms={stats.avg_match_ms:.3f}")
    print(f"avg_total_ms={stats.avg_total_ms:.3f}")
    print(f"avg_inlier_ratio={stats.avg_inlier_ratio:.3f}")
    print(f"trajectory_length={stats.trajectory_length:.3f}")
    print(f"output_dir={stats.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="SuperPoint Demo/2D SLAM 통합 CLI")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["demo", "slam2d"],
        help="실행 모드 선택 (demo 또는 slam2d)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help='이미지 디렉토리/비디오/"camera" (demo) 또는 비디오 파일 경로 (slam2d).',
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

    parser.add_argument("--mask_car", action="store_true", help="Enable car masking")
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[640, 480],
        help="Resize input frame to [width height] (slam2d mode)",
    )
    parser.add_argument(
        "--slam_conf_thresh",
        type=float,
        default=0.003,
        help="SLAM2D 모드용 conf_thresh (기본: 0.003)",
    )
    parser.add_argument(
        "--slam_nms_dist",
        type=int,
        default=4,
        help="SLAM2D 모드용 NMS 거리 (기본: 4)",
    )
    parser.add_argument(
        "--motion_scale",
        type=float,
        default=1.0,
        help="프레임당 이동 스케일 (상대 궤적 크기 조정)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_superpoint_2d",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--sp_scale",
        type=float,
        default=0.75,
        help="SuperPoint 추론 해상도 스케일 (0.25~1.0)",
    )
    parser.add_argument(
        "--sp_interval",
        type=int,
        default=1,
        help="SuperPoint 추론 프레임 간격 (1이면 매 프레임)",
    )
    parser.add_argument(
        "--sp_fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CUDA 환경에서 FP16 추론 사용 여부 (--no-sp_fp16 로 비활성화)",
    )
    parser.add_argument(
        "--max_kpts",
        type=int,
        default=500,
        help="균일 샘플링 후 유지할 최대 특징점 수",
    )
    parser.add_argument(
        "--uniform_grid",
        nargs=2,
        type=int,
        default=[8, 6],
        help="특징점 균일 분포용 그리드 [x y]",
    )
    parser.add_argument(
        "--use_subpixel_refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Heatmap 기반 서브픽셀 Center-of-Mass 보정 사용 여부",
    )
    parser.add_argument(
        "--use_uniform_distribution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="그리드 기반 균일 특징점 분포 샘플링 사용 여부",
    )
    parser.add_argument(
        "--use_hybrid_matching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mutual + ratio + 기하검증 혼합 매칭 사용 여부",
    )
    parser.add_argument(
        "--ratio_thresh",
        type=float,
        default=0.85,
        help="Lowe ratio test 임계값",
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=0.8,
        help="EssentialMat RANSAC reprojection 임계값",
    )
    parser.add_argument(
        "--com_radius",
        type=int,
        default=2,
        help="서브픽셀 Center-of-Mass 반경(px)",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    opt = parser.parse_args()

    if opt.mode == "demo":
        if opt.weights is None:
            parser.error("demo 모드에는 --weights (또는 --weights_path) 가 필요합니다.")
        run_demo(opt)
    elif opt.mode == "slam2d":
        if opt.weights is None:
            parser.error("slam2d 모드에는 --weights (또는 --weights_path) 가 필요합니다.")
        run_slam2d(opt)
