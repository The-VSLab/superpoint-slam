import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from slam.visual_slam_2d import VisualSLAM2D
from orbslam.orb_slam_2d import ORBSLAM2D


def build_parser():
    parser = argparse.ArgumentParser(description="SuperPoint 2D SLAM vs ORB 2D SLAM 비교")
    parser.add_argument("--input", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--weights", type=str, required=True, help="SuperPoint 가중치 경로")
    parser.add_argument("--resize", nargs=2, type=int, default=[640, 480], help="입력 리사이즈 [width height]")
    parser.add_argument("--nn_thresh", type=float, default=0.7, help="SuperPoint 매칭 임계값")
    parser.add_argument("--slam_conf_thresh", type=float, default=0.003, help="SuperPoint conf_thresh")
    parser.add_argument("--slam_nms_dist", type=int, default=4, help="SuperPoint nms_dist")
    parser.add_argument("--sp_scale", type=float, default=0.75, help="SuperPoint 추론 해상도 스케일")
    parser.add_argument("--sp_interval", type=int, default=1, help="SuperPoint 추론 프레임 간격")
    parser.add_argument(
        "--sp_backend",
        type=str,
        default="torch",
        choices=["torch", "trt"],
        help="SuperPoint 백엔드 선택 (torch 또는 trt)",
    )
    parser.add_argument(
        "--sp_fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CUDA 환경에서 FP16 추론 사용 여부",
    )
    parser.add_argument("--trt_engine", type=str, default=None, help="TensorRT 엔진(.engine) 경로")
    parser.add_argument("--trt_onnx", type=str, default=None, help="TensorRT 빌드용 ONNX 경로")
    parser.add_argument("--trt_build", action="store_true", help="trt_engine이 없으면 trt_onnx로 엔진 빌드")
    parser.add_argument("--max_kpts", type=int, default=500, help="균일 샘플링 후 최대 특징점 수")
    parser.add_argument("--uniform_grid", nargs=2, type=int, default=[8, 6], help="특징점 균일 분포 그리드 [x y]")
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
    parser.add_argument("--ratio_thresh", type=float, default=0.85, help="Lowe ratio test 임계값")
    parser.add_argument("--ransac_thresh", type=float, default=0.8, help="EssentialMat RANSAC 임계값")
    parser.add_argument("--com_radius", type=int, default=2, help="서브픽셀 Center-of-Mass 반경")
    parser.add_argument("--orb_nfeatures", type=int, default=1200, help="ORB 특징점 수")
    parser.add_argument("--orb_max_matches", type=int, default=500, help="ORB 최대 매칭 수")
    parser.add_argument("--motion_scale", type=float, default=1.0, help="두 방식 공통 이동 스케일")
    parser.add_argument("--mask_car", action="store_true", help="SuperPoint에서 하단 대시보드 마스킹")
    parser.add_argument("--output_dir", type=str, default="results_compare_2d", help="비교 결과 저장 디렉토리")
    parser.add_argument("--show_display", action="store_true", help="실시간 화면 표시")
    return parser


def print_stats_row(label, stats):
    print(
        f"{label:<14} | frames={stats.frames:<5} total_ms={stats.avg_total_ms:>8.3f} "
        f"extract_ms={stats.avg_extract_ms:>8.3f} match_ms={stats.avg_match_ms:>8.3f} "
        f"inlier_ratio={stats.avg_inlier_ratio:>6.3f} traj_len={stats.trajectory_length:>8.3f}"
    )


def main():
    opt = build_parser().parse_args()

    out_root = Path(opt.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    superpoint = VisualSLAM2D(
        weights_path=opt.weights,
        input_path=opt.input,
        nn_thresh=opt.nn_thresh,
        resize=tuple(opt.resize),
        conf_thresh=opt.slam_conf_thresh,
        nms_dist=opt.slam_nms_dist,
        mask_car=opt.mask_car,
        motion_scale=opt.motion_scale,
        output_dir=str(out_root / "superpoint"),
        show_display=opt.show_display,
        sp_scale=opt.sp_scale,
        sp_interval=opt.sp_interval,
        sp_fp16=opt.sp_fp16,
        sp_backend=opt.sp_backend,
        trt_engine=opt.trt_engine,
        trt_onnx=opt.trt_onnx,
        trt_build=opt.trt_build,
        max_kpts=opt.max_kpts,
        uniform_grid=tuple(opt.uniform_grid),
        use_subpixel_refine=opt.use_subpixel_refine,
        use_uniform_distribution=opt.use_uniform_distribution,
        use_hybrid_matching=opt.use_hybrid_matching,
        ratio_thresh=opt.ratio_thresh,
        ransac_thresh=opt.ransac_thresh,
        com_radius=opt.com_radius,
    )
    orb = ORBSLAM2D(
        input_path=opt.input,
        resize=tuple(opt.resize),
        nfeatures=opt.orb_nfeatures,
        max_matches=opt.orb_max_matches,
        motion_scale=opt.motion_scale,
        output_dir=str(out_root / "orb"),
        show_display=opt.show_display,
    )

    print("\n==> Running SuperPoint 2D SLAM...")
    sp_stats = superpoint.process()

    print("\n==> Running ORB 2D SLAM...")
    orb_stats = orb.process()

    print("\n==> Comparison Summary")
    print_stats_row("superpoint_2d", sp_stats)
    print_stats_row("orbslam_2d", orb_stats)

    comparison = {
        "superpoint_2d": sp_stats.to_dict(),
        "orbslam_2d": orb_stats.to_dict(),
    }
    with open(out_root / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_root / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
