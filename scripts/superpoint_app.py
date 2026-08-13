"""
SuperPoint 3D SLAM 앱
- 3D 포인트 클라우드 생성 + 경로 추정
- 동영상/이미지 시퀀스 입력 지원
- result/ 디렉토리에 자동 번호 매기기 (superpoint_3d_01, ...)
"""
import argparse
import copy
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from slam.visual_slam_3d import VisualSLAM3D
from config.slam_config import SLAMConfig


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
    parser.add_argument("--input", type=str, required=True, help="동영상 또는 이미지 시퀀스 경로")
    parser.add_argument("--weights", type=str, required=True, help="SuperPoint 가중치 경로")
    parser.add_argument("--calib", type=str, default=None,
                        help="KITTI calib.txt 경로 (지정 시 실제 카메라 내부 파라미터 사용)")
    parser.add_argument("--resize", nargs=2, type=int, default=None,
                        help="입력 리사이즈 [width height] (미지정 시 종횡비 유지 자동 리사이즈)")
    parser.add_argument("--slam_conf_thresh", type=float, default=0.015, 
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
    parser.add_argument("--min_shadow_grad", type=float, default=15.0,
                        help="그림자 영역 최소 그래디언트 임계값")
    parser.add_argument("--min_shadow_local_std", type=float, default=8.0,
                        help="그림자 영역 최소 로컬 표준편차 임계값")
    parser.add_argument("--shadow_rel_dark_thresh", type=float, default=0.82,
                        help="주변 대비 상대 명도 임계값(낮을수록 그림자 판정 강화)")
    parser.add_argument("--top_region_ratio", type=float, default=0.30,
                        help="상단 억제 영역 비율(0~1)")
    parser.add_argument("--top_region_min_grad", type=float, default=25.0,
                        help="상단 영역 유지용 최소 그래디언트")
    parser.add_argument("--top_region_min_std", type=float, default=10.0,
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
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False,
                        help="재현성 모드(고정 seed + deterministic 연산) 활성화")
    parser.add_argument("--seed", type=int, default=7,
                        help="재현성 모드에서 사용할 시드")
    parser.add_argument("--output_dir", type=str, default="result", help="출력 루트 디렉토리")
    parser.add_argument("--show_display", action=argparse.BooleanOptionalAction, default=True,
                        help="실시간 화면 표시 (기본: 활성화, --no-show_display로 비활성화)")
    parser.add_argument("--use_semantic", action=argparse.BooleanOptionalAction, default=False,
                        help="YOLOv8 기반 동적 객체(차량, 보행자 등) 마스킹 활성화 (3D SLAM 전용)")

    # === 성능 최적화 파라미터 (3D SLAM 전용) ===
    parser.add_argument("--no-viz", action="store_true",
                        help="실시간 2D 시각화 비활성화 (성능 향상: ~15-20ms)")
    parser.add_argument("--topdown-features", action=argparse.BooleanOptionalAction, default=True,
                        help="topdown_map.png에 특징점 표시 (--no-topdown-features면 경로만)")
    parser.add_argument("--sp-scale", type=float, default=1.0,
                        help="SuperPoint 추론 해상도 스케일 (0.5 = 절반 해상도, 기본: 1.0)")
    parser.add_argument("--sp-interval", type=int, default=1,
                        help="SuperPoint 추론 간격 (프레임 단위, 기본: 1 = 매 프레임)")
    parser.add_argument("--flow-win-size", type=int, default=21,
                        help="Optical Flow 윈도우 크기 (기본: 21, 권장: 15)")
    parser.add_argument("--flow-max-level", type=int, default=3,
                        help="Optical Flow 피라미드 레벨 (기본: 3, 권장: 2)")
    parser.add_argument("--flow-fb-thresh", type=float, default=1.0,
                        help="Optical Flow Forward-Backward 일관성 임계값 (픽셀, 기본: 1.0)")
    parser.add_argument("--no-clahe", action="store_true",
                        help="CLAHE 전처리 비활성화 (성능 향상: ~3-5ms)")
    parser.add_argument("--local-map-limit", type=int, default=0,
                        help="로컬 맵 트래킹 MapPoint 개수 제한 (0 = 무제한, 권장: 500)")
    parser.add_argument("--filter-on-sp-only", action="store_true",
                        help="필터를 SuperPoint 추론 프레임에만 적용 (optical flow 프레임은 스킵)")

    # === YAML 설정 파일 ===
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 설정 파일 경로 (미지정 시 기본값 사용, CLI 인자로 개별 오버라이드 가능)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="DEBUG 레벨 로깅 활성화 (프레임별 상세 출력)")

    return parser


def _find_explicit_args(parser, args=None):
    """CLI에서 명시적으로 지정된 인자의 dest 이름 set 반환.
    argparse 기본값을 SUPPRESS로 설정한 사본으로 파싱하여,
    실제로 커맨드라인에서 넘어온 인자만 식별한다.
    """
    p = copy.deepcopy(parser)
    for action in p._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    ns, _ = p.parse_known_args(args)
    return set(vars(ns).keys())


# CLI dest → config dot-notation 매핑 테이블
_CLI_MAP = {
    # superpoint
    "slam_conf_thresh": "superpoint.conf_thresh",
    "nn_thresh": "superpoint.nn_thresh",
    "max_kpts": "superpoint.max_kpts",
    "min_kpts": "superpoint.min_kpts",
    "roi_sky": "superpoint.roi_sky",
    "roi_hood": "superpoint.roi_hood",
    "sp_scale": "superpoint.sp_scale",
    "sp_interval": "superpoint.sp_interval",
    "sp_fp16": "superpoint.sp_fp16",
    # semantic
    "use_semantic": "semantic.enabled",
    # point_filter
    "use_shadow_filter": "point_filter.use_shadow_filter",
    "use_top_region_filter": "point_filter.use_top_region_filter",
    "shadow_value_thresh": "point_filter.shadow_value_thresh",
    "shadow_saturation_thresh": "point_filter.shadow_saturation_thresh",
    "min_shadow_grad": "point_filter.min_shadow_grad",
    "min_shadow_local_std": "point_filter.min_shadow_local_std",
    "shadow_rel_dark_thresh": "point_filter.shadow_rel_dark_thresh",
    "top_region_ratio": "point_filter.top_region_ratio",
    "top_region_min_grad": "point_filter.top_region_min_grad",
    "top_region_min_std": "point_filter.top_region_min_std",
    # optical_flow
    "flow_win_size": "optical_flow.win_size",
    "flow_max_level": "optical_flow.max_level",
    "flow_fb_thresh": "optical_flow.fb_thresh",
    # performance
    "local_map_limit": "performance.local_map_limit",
    "filter_on_sp_only": "performance.filter_on_sp_only",
    # viz
    "topdown_features": "viz.topdown_features",
}


def build_config(opt, parser):
    """CLI 옵션에서 SLAMConfig 생성.
    --config 지정 시 YAML 로드 후 CLI 개별 인자로 오버라이드.
    --config 미지정 시 dataclass 기본값(= default.yaml 동일) 사용.
    """
    # 1. 기본 config 로드
    if opt.config:
        cfg = SLAMConfig.from_yaml(opt.config)
    else:
        cfg = SLAMConfig()

    # 2. aggressive_shadow_filter 프리셋 (개별 CLI보다 먼저 적용)
    if opt.aggressive_shadow_filter:
        cfg.point_filter.shadow_value_thresh = 0.38
        cfg.point_filter.shadow_saturation_thresh = 0.24
        cfg.point_filter.shadow_rel_dark_thresh = 0.72
        cfg.point_filter.min_shadow_grad = 32
        cfg.point_filter.min_shadow_local_std = 16
        cfg.point_filter.top_region_ratio = 0.58
        cfg.point_filter.top_region_min_grad = 52
        cfg.point_filter.top_region_min_std = 20

    # 3. CLI에서 명시적으로 지정된 인자만 override
    explicit = _find_explicit_args(parser)
    overrides = {}
    for cli_dest, config_key in _CLI_MAP.items():
        if cli_dest in explicit:
            overrides[config_key] = getattr(opt, cli_dest)

    # 반전 플래그 처리
    if "no_viz" in explicit:
        overrides["viz.enabled"] = not opt.no_viz
    if "no_clahe" in explicit:
        overrides["clahe.enabled"] = not opt.no_clahe

    if overrides:
        cfg.merge_cli(overrides)

    return cfg


def run_3d_slam(opt, parser):
    """SuperPoint 3D SLAM 실행 (포인트 클라우드 시각화)"""
    out_dir = get_next_subdir(opt.output_dir, "superpoint_3d")

    print(f"\n{'='*80}")
    print(f"🎬 SuperPoint 3D SLAM 시작")
    print(f"{'='*80}")
    print(f"📁 입력: {opt.input}")
    print(f"📁 출력: {out_dir}")
    if opt.config:
        print(f"📄 설정: {opt.config}")
    print(f"{'='*80}\n")

    config = build_config(opt, parser)

    slam = VisualSLAM3D(
        weights_path=opt.weights,
        input_path=opt.input,
        config=config,
        output_dir=str(out_dir),
        resize=opt.resize,
        calib_path=opt.calib,
    )

    slam.process()

    print(f"\n{'='*80}")
    print(f"✅ 3D 포인트 클라우드가 생성되었습니다!")
    print(f"{'='*80}")
    print(f"💾 포인트 클라우드: {out_dir}/final_slam_map.ply")
    print(f"💾 Top-Down 맵: {out_dir}/topdown_map.png")
    print(f"{'='*80}\n")


def main():
    parser = build_parser()
    opt = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if opt.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_3d_slam(opt, parser)


if __name__ == "__main__":
    main()
