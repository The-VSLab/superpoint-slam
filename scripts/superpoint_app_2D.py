import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo_runner import add_demo_args, run_demo


def run_slam2d(opt):
    from slam.visual_slam_2d import VisualSLAM2D

    # 강한 그림자 억제 프리셋 적용
    if opt.aggressive_shadow_filter:
        opt.shadow_value_thresh = 0.38
        opt.shadow_saturation_thresh = 0.24
        opt.shadow_rel_dark_thresh = 0.72
        opt.min_shadow_grad = 32
        opt.min_shadow_local_std = 16
        opt.top_region_ratio = 0.50
        opt.top_region_min_grad = 45
        opt.top_region_min_std = 18

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
        min_kpts=opt.min_kpts,
        uniform_grid=tuple(opt.uniform_grid),
        use_subpixel_refine=opt.use_subpixel_refine,
        use_uniform_distribution=opt.use_uniform_distribution,
        use_hybrid_matching=opt.use_hybrid_matching,
        ratio_thresh=opt.ratio_thresh,
        ransac_thresh=opt.ransac_thresh,
        com_radius=opt.com_radius,
        kpt_display_radius=opt.kpt_display_radius,
        use_shadow_filter=opt.use_shadow_filter,
        use_top_region_filter=opt.use_top_region_filter,
        shadow_value_thresh=opt.shadow_value_thresh,
        shadow_saturation_thresh=opt.shadow_saturation_thresh,
        min_shadow_grad=opt.min_shadow_grad,
        min_shadow_local_std=opt.min_shadow_local_std,
        shadow_rel_dark_thresh=opt.shadow_rel_dark_thresh,
        top_region_ratio=opt.top_region_ratio,
        top_region_min_grad=opt.top_region_min_grad,
        top_region_min_std=opt.top_region_min_std,
        bottom_region_ratio=opt.bottom_region_ratio,
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
    parser.add_argument(        \"--min_kpts\",
        type=int,
        default=600,
        help=\"프레임당 최소 특징점 개수\",
    )
    parser.add_argument(        "--uniform_grid",
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
    parser.add_argument(
        "--kpt_display_radius",
        type=int,
        default=1,
        help="화면 출력 특징점 반지름 (픽셀, 낮을수록 작음)",
    )
    parser.add_argument(
        "--aggressive_shadow_filter",
        action="store_true",
        help="강한 그림자 억제 프리셋(자동으로 파라미터 조정)",
    )
    parser.add_argument(
        "--use_shadow_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="그림자 기반 특징점 필터 활성화",
    )
    parser.add_argument(
        "--use_top_region_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="영상 상단 영역(하늘) 특징점 억제 필터 활성화",
    )
    parser.add_argument(
        "--shadow_value_thresh",
        type=float,
        default=0.46,
        help="그림자 명도 임계값(낮을수록 더 어두운 영역만 제거)",
    )
    parser.add_argument(
        "--shadow_saturation_thresh",
        type=float,
        default=0.30,
        help="그림자 채도 임계값(낮을수록 무채색 그림자만 제거)",
    )
    parser.add_argument(
        "--min_shadow_grad",
        type=float,
        default=22.0,
        help="그림자 영역 최소 그래디언트 임계값",
    )
    parser.add_argument(
        "--min_shadow_local_std",
        type=float,
        default=10.0,
        help="그림자 영역 최소 로컬 표준편차 임계값",
    )
    parser.add_argument(
        "--shadow_rel_dark_thresh",
        type=float,
        default=0.82,
        help="주변 대비 상대 명도 임계값(낮을수록 그림자 판정 강화)",
    )
    parser.add_argument(
        "--top_region_ratio",
        type=float,
        default=0.38,
        help="상단 억제 영역 비율(0~1)",
    )
    parser.add_argument(
        "--top_region_min_grad",
        type=float,
        default=34.0,
        help="상단 영역 유지용 최소 그래디언트",
    )
    parser.add_argument(
        "--top_region_min_std",
        type=float,
        default=14.0,
        help="상단 영역 유지용 최소 로컬 표준편차",
    )
    parser.add_argument(
        "--bottom_region_ratio",
        type=float,
        default=0.35,
        help="바닥 영역 비율(0~1, 하혼 영역 높이 비율)",
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
