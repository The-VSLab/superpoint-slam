"""
SuperPoint 2D SLAM 단독 앱
- 경로 추정 + 2D 맵 생성
- 동영상/이미지 시퀀스 입력 지원
- result/ 디렉토리에 자동 번호 매기기 (superpoint_01, orb_01, ...)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from slam.visual_slam_2d import VisualSLAM2D
from slam.visual_slam_3d import VisualSLAM3D
from orbslam.orb_slam_2d import ORBSLAM2D


def get_next_subdir(output_dir, prefix):
    """superpoint_01, superpoint_02 같은 다음 디렉토리 번호 생성"""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    existing = list(out_root.glob(f"{prefix}_[0-9][0-9]"))
    if not existing:
        next_num = 1
    else:
        nums = [int(p.name.split("_")[-1]) for p in existing]
        next_num = max(nums) + 1
    
    next_dir = out_root / f"{prefix}_{next_num:02d}"
    next_dir.mkdir(parents=True, exist_ok=True)
    return next_dir


def build_parser():
    parser = argparse.ArgumentParser(description="SuperPoint SLAM")
    parser.add_argument("--mode", type=str, default="slam", choices=["slam", "compare", "3d"], 
                        help="slam: SuperPoint 2D | compare: SuperPoint vs ORB | 3d: 3D 포인트 클라우드")
    parser.add_argument("--input", type=str, required=True, help="동영상 또는 이미지 시퀀스 경로")
    parser.add_argument("--weights", type=str, required=True, help="SuperPoint 가중치 경로")
    parser.add_argument("--resize", nargs=2, type=int, default=[640, 480], 
                        help="입력 리사이즈 [width height]")
    parser.add_argument("--slam_conf_thresh", type=float, default=0.001, 
                        help="특징점 신뢰도 임계값 (낮을수록 특징점 증가)")
    parser.add_argument("--slam_nms_dist", type=int, default=4, help="NMS 거리")
    parser.add_argument("--nn_thresh", type=float, default=0.7, help="매칭 임계값")
    parser.add_argument("--max_kpts", type=int, default=1200, help="프레임당 최대 특징점 개수")
    parser.add_argument("--min_parallax_px", type=float, default=2.0,
                        help="맵 포인트 추가 최소 시차(픽셀)")
    parser.add_argument("--use_subpixel_refine", action=argparse.BooleanOptionalAction, default=True,
                        help="Sub-pixel Refinement 활성화")
    parser.add_argument("--use_uniform_distribution", action=argparse.BooleanOptionalAction, default=True,
                        help="Grid-based NMS / 균일 분포 활성화")
    parser.add_argument("--uniform_grid", nargs=2, type=int, default=[8, 6],
                        help="균일 분포용 그리드 [gx gy]")
    parser.add_argument("--sp_fp16", action=argparse.BooleanOptionalAction, default=False,
                        help="FP16 추론 활성화 (CUDA 전용)")
    parser.add_argument("--output_dir", type=str, default="result", help="출력 루트 디렉토리")
    parser.add_argument("--show_display", action="store_true", default=True, 
                        help="실시간 화면 표시 (기본: 활성화)")
    
    return parser


def run_superpoint_slam(opt):
    """SuperPoint 2D SLAM 실행"""
    out_dir = get_next_subdir(opt.output_dir, "superpoint")
    
    print(f"\n{'='*80}")
    print(f"🚀 SuperPoint 2D SLAM 시작")
    print(f"{'='*80}")
    print(f"📁 입력: {opt.input}")
    print(f"📁 출력: {out_dir}")
    print(f"{'='*80}\n")
    
    slam = VisualSLAM2D(
        weights_path=opt.weights,
        input_path=opt.input,
        nn_thresh=opt.nn_thresh,
        resize=tuple(opt.resize),
        conf_thresh=opt.slam_conf_thresh,
        nms_dist=opt.slam_nms_dist,
        max_kpts=opt.max_kpts,
        min_parallax_px=opt.min_parallax_px,
        use_subpixel_refine=opt.use_subpixel_refine,
        use_uniform_distribution=opt.use_uniform_distribution,
        uniform_grid=tuple(opt.uniform_grid),
        sp_fp16=opt.sp_fp16,
        output_dir=str(out_dir),
        show_display=opt.show_display,
    )
    
    stats = slam.process()
    
    print(f"\n{'='*80}")
    print(f"✅ SuperPoint 완료!")
    print(f"{'='*80}")
    print(f"📊 프레임: {stats.frames}")
    print(f"⏱️  평균 처리: {stats.avg_total_ms:.1f}ms/frame")
    print(f"🎯 특징점: {stats.avg_kpts:.0f}/frame")
    print(f"🔗 매칭: {stats.avg_matches:.1f}/frame")
    print(f"✨ 신뢰도: {stats.avg_inlier_ratio*100:.1f}%")
    print(f"📍 궤적: {stats.trajectory_length:.1f}m")
    print(f"💾 저장: {out_dir}/topdown_map.png")
    print(f"{'='*80}\n")
    
    return stats


def run_orb_slam(opt):
    """ORB SLAM 2D 실행"""
    out_dir = get_next_subdir(opt.output_dir, "orb")
    
    print(f"\n{'='*80}")
    print(f"🚀 ORB SLAM 2D 시작")
    print(f"{'='*80}")
    print(f"📁 입력: {opt.input}")
    print(f"📁 출력: {out_dir}")
    print(f"{'='*80}\n")
    
    orb = ORBSLAM2D(
        input_path=opt.input,
        resize=tuple(opt.resize),
        output_dir=str(out_dir),
        show_display=opt.show_display,
    )
    
    stats = orb.process()
    
    print(f"\n{'='*80}")
    print(f"✅ ORB 완료!")
    print(f"{'='*80}")
    print(f"📊 프레임: {stats.frames}")
    print(f"⏱️  평균 처리: {stats.avg_total_ms:.1f}ms/frame")
    print(f"🎯 특징점: {stats.avg_kpts:.0f}/frame")
    print(f"🔗 매칭: {stats.avg_matches:.1f}/frame")
    print(f"✨ 신뢰도: {stats.avg_inlier_ratio*100:.1f}%")
    print(f"📍 궤적: {stats.trajectory_length:.1f}m")
    print(f"💾 저장: {out_dir}/topdown_map.png")
    print(f"{'='*80}\n")
    
    return stats


def run_3d_slam(opt):
    """SuperPoint 3D SLAM 실행 (포인트 클라우드 시각화)"""
    print(f"\n{'='*80}")
    print(f"🎬 SuperPoint 3D SLAM 시작")
    print(f"{'='*80}")
    print(f"📁 입력: {opt.input}")
    print(f"{'='*80}\n")
    
    slam = VisualSLAM3D(
        weights_path=opt.weights,
        input_path=opt.input,
        nn_thresh=opt.nn_thresh,
        sp_scale=1.0,
        sp_interval=1,
        sp_fp16=False,
    )
    
    slam.process()
    
    print(f"\n{'='*80}")
    print(f"✅ 3D 포인트 클라우드가 생성 및 표시되었습니다!")
    print(f"{'='*80}")
    print(f"💾 저장위치: path_final/final_slam_map.ply")
    print(f"{'='*80}\n")


def main():
    opt = build_parser().parse_args()
    
    if opt.mode == "slam":
        # SuperPoint 2D만 실행
        run_superpoint_slam(opt)
    elif opt.mode == "compare":
        # 2D 비교: SuperPoint vs ORB
        sp_stats = run_superpoint_slam(opt)
        orb_stats = run_orb_slam(opt)
        
        # 비교 결과 출력
        print(f"\n{'='*80}")
        print(f"📊 비교 결과")
        print(f"{'='*80}")
        print(f"{'항목':<20} {'SuperPoint':<20} {'ORB':<20}")
        print(f"{'-'*60}")
        print(f"{'처리시간(ms)':<20} {sp_stats.avg_total_ms:<20.1f} {orb_stats.avg_total_ms:<20.1f}")
        print(f"{'특징점':<20} {sp_stats.avg_kpts:<20.0f} {orb_stats.avg_kpts:<20.0f}")
        print(f"{'매칭수':<20} {sp_stats.avg_matches:<20.1f} {orb_stats.avg_matches:<20.1f}")
        print(f"{'신뢰도(%)':<20} {sp_stats.avg_inlier_ratio*100:<20.1f} {orb_stats.avg_inlier_ratio*100:<20.1f}")
        print(f"{'궤적(m)':<20} {sp_stats.trajectory_length:<20.1f} {orb_stats.trajectory_length:<20.1f}")
        print(f"{'='*60}\n")
    elif opt.mode == "3d":
        # 3D 포인트 클라우드 (Open3D 시각화)
        run_3d_slam(opt)


if __name__ == "__main__":
    main()
