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
from orbslam.orb_slam_3d import ORBSLAM3D


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
    parser.add_argument("--backend", type=str, default="superpoint", choices=["superpoint", "orb"],
                        help="3d 모드에서 사용할 백엔드 선택 (superpoint/orb)")
    parser.add_argument("--input", type=str, required=True, help="동영상 또는 이미지 시퀀스 경로")
    parser.add_argument("--weights", type=str, default=None, help="SuperPoint 가중치 경로")
    parser.add_argument("--resize", nargs=2, type=int, default=[640, 480], 
                        help="입력 리사이즈 [width height]")
    parser.add_argument("--slam_conf_thresh", type=float, default=0.0005, 
                        help="특징점 신뢰도 임계값 (낮을수록 특징점 증가)")
    parser.add_argument("--slam_nms_dist", type=int, default=3, help="NMS 거리")
    parser.add_argument("--nn_thresh", type=float, default=0.7, help="매칭 임계값")
    parser.add_argument("--max_kpts", type=int, default=1500, help="프레임당 최대 특징점 개수(기본: 1500)")
    parser.add_argument("--min_kpts", type=int, default=600, help="프레임당 최소 특징점 개수(기본: 600)")
    parser.add_argument("--min_parallax_px", type=float, default=2.0,
                        help="맵 포인트 추가 최소 시차(픽셀)")
    parser.add_argument("--kpt_display_radius", type=int, default=1,
                        help="화면 출력 특징점 반지름 (픽셀, 낮을수록 작음)")
    parser.add_argument("--aggressive_shadow_filter", action="store_true",
                        help="강한 그림자 억제 프리셋(자동으로 파라미터 조정)")
    parser.add_argument("--use_shadow_filter", action=argparse.BooleanOptionalAction, default=True,
                        help="그림자 기반 특징점 필터 활성화")
    parser.add_argument("--use_top_region_filter", action=argparse.BooleanOptionalAction, default=True,
                        help="최상단 20% 강제 특징점 제거 필터 활성화(기본: True)")
    
    # ROI 강제 마스킹 설정 (자율주행 환경 특화)
    parser.add_argument("--roi_sky", type=float, default=0.35,
                        help="하늘 마스킹 영역(상단 비율, 0.0이면 비활성화, 기본: 0.35)")
    parser.add_argument("--roi_hood", type=float, default=0.85,
                        help="본넷 마스킹 영역(하단 비율, 1.0이면 비활성화, 기본: 0.85)")
    parser.add_argument("--shadow_value_thresh", type=float, default=0.46,
                        help="그림자 명도 임계값(낮을수록 더 어두운 영역만 제거)")
    parser.add_argument("--shadow_saturation_thresh", type=float, default=0.30,
                        help="그림자 채도 임계값(낮을수록 무채색 그림자만 제거)")
    parser.add_argument("--min_shadow_grad", type=float, default=22.0,
                        help="그림자 영역 최소 그래디언트 임계값")
    parser.add_argument("--min_shadow_local_std", type=float, default=10.0,
                        help="그림자 영역 최소 로컬 표준편차 임계값")
    parser.add_argument("--shadow_rel_dark_thresh", type=float, default=0.82,
                        help="주변 대비 상대 명도 임계값(낮을수록 그림자 판정 강화)")
    parser.add_argument("--top_region_ratio", type=float, default=0.38,
                        help="상단 억제 영역 비율(0~1)")
    parser.add_argument("--top_region_min_grad", type=float, default=34.0,
                        help="상단 영역 유지용 최소 그래디언트")
    parser.add_argument("--top_region_min_std", type=float, default=14.0,
                        help="상단 영역 유지용 최소 로컬 표준편차")
    parser.add_argument("--bottom_region_ratio", type=float, default=0.35,
                        help="바닥 영역 비율(0~1, 하혼 영역 높이 비율)")
    parser.add_argument("--filter_floor", action="store_true",
                        help="입력 영상 아래쪽(bottom_region_ratio) 영역을 마스킹하여 특징점 추출 방지")
    parser.add_argument("--use_subpixel_refine", action=argparse.BooleanOptionalAction, default=True,
                        help="Sub-pixel Refinement 활성화")
    parser.add_argument("--use_uniform_distribution", action=argparse.BooleanOptionalAction, default=True,
                        help="Grid-based NMS / 균일 분포 활성화")
    parser.add_argument("--uniform_grid", nargs=2, type=int, default=[8, 6],
                        help="균일 분포용 그리드 [gx gy]")
    parser.add_argument("--sp_fp16", action=argparse.BooleanOptionalAction, default=False,
                        help="FP16 추론 활성화 (CUDA 전용)")
    parser.add_argument("--sp_scale", type=float, default=0.5,
                        help="SuperPoint 입력 이미지 축소 비율 (0.5면 4배 빨라짐, 기본: 0.5)")
    parser.add_argument("--sp_interval", type=int, default=2,
                        help="SuperPoint 추론 주기 (n프레임마다 실행, 기본: 2)")
    parser.add_argument("--orb_nfeatures", type=int, default=1500,
                        help="ORB 3D에서 사용할 특징점 개수")
    parser.add_argument("--orb_max_matches", type=int, default=500,
                        help="ORB 3D에서 프레임당 최대 매칭 수")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False,
                        help="재현성 모드(고정 seed + deterministic 연산) 활성화")
    parser.add_argument("--seed", type=int, default=7,
                        help="재현성 모드에서 사용할 시드")
    parser.add_argument("--output_dir", type=str, default="result", help="출력 루트 디렉토리")
    parser.add_argument("--show_display", action=argparse.BooleanOptionalAction, default=True,
                        help="실시간 화면 표시 (기본: 활성화, --no-show_display로 비활성화)")
    
    return parser


def run_superpoint_slam(opt):
    """SuperPoint 2D SLAM 실행"""
    out_dir = get_next_subdir(opt.output_dir, "superpoint")
    
    # 강한 그림자 억제 프리셋 적용
    if opt.aggressive_shadow_filter:
        opt.shadow_value_thresh = 0.38
        opt.shadow_saturation_thresh = 0.24
        opt.shadow_rel_dark_thresh = 0.72
        opt.min_shadow_grad = 32
        opt.min_shadow_local_std = 16
        opt.top_region_ratio = 0.58
        opt.top_region_min_grad = 52
        opt.top_region_min_std = 20
        print("🔥 강한 그림자 억제 프리셋 적용\n")
    
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
        min_kpts=opt.min_kpts,
        min_parallax_px=opt.min_parallax_px,
        use_subpixel_refine=opt.use_subpixel_refine,
        use_uniform_distribution=opt.use_uniform_distribution,
        uniform_grid=tuple(opt.uniform_grid),
        use_shadow_filter=opt.use_shadow_filter,
        kpt_display_radius=opt.kpt_display_radius,
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
        filter_floor=opt.filter_floor,
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
    """3D SLAM 실행 (SuperPoint 또는 ORB 백엔드)"""
    if opt.backend == "orb":
        out_dir = get_next_subdir(opt.output_dir, "orb_3d")

        print(f"\n{'='*80}")
        print(f"🎬 ORB 3D SLAM 시작")
        print(f"{'='*80}")
        print(f"📁 입력: {opt.input}")
        print(f"📁 출력: {out_dir}")
        print(f"{'='*80}\n")

        slam = ORBSLAM3D(
            input_path=opt.input,
            resize=tuple(opt.resize),
            nfeatures=opt.orb_nfeatures,
            max_matches=opt.orb_max_matches,
            output_dir=str(out_dir),
            show_display=opt.show_display,
        )
        slam.process()
    else:
        out_dir = get_next_subdir(opt.output_dir, "superpoint_3d")

        print(f"\n{'='*80}")
        print(f"🎬 SuperPoint 3D SLAM 시작")
        print(f"{'='*80}")
        print(f"📁 입력: {opt.input}")
        print(f"📁 출력: {out_dir}")
        print(f"{'='*80}\n")

        slam = VisualSLAM3D(
            weights_path=opt.weights,
            input_path=opt.input,
            nn_thresh=opt.nn_thresh,
            conf_thresh=opt.slam_conf_thresh,
            nms_dist=opt.slam_nms_dist,
            max_kpts=opt.max_kpts,
            sp_scale=opt.sp_scale,
            sp_interval=opt.sp_interval,
            sp_fp16=opt.sp_fp16,
            output_dir=str(out_dir),
            roi_sky=opt.roi_sky,
            roi_hood=opt.roi_hood
        )
        
        slam.process()
    
    print(f"\n{'='*80}")
    print(f"✅ 3D 포인트 클라우드 및 2D 비교 맵이 생성되었습니다!")
    print(f"{'='*80}")
    print(f"💾 포인트 클라우드: {out_dir}/final_slam_map.ply")
    print(f"💾 Top-Down 맵: {out_dir}/topdown_map.png")
    print(f"{'='*80}\n")


def main():
    parser = build_parser()
    opt = parser.parse_args()

    needs_superpoint_weights = (
        opt.mode in ("slam", "compare")
        or (opt.mode == "3d" and opt.backend == "superpoint")
    )
    if needs_superpoint_weights and not opt.weights:
        parser.error("--weights 는 SuperPoint 모드(slam/compare/3d --backend superpoint)에서 필수입니다.")
    
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
