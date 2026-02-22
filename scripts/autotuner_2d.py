"""
SuperPoint 2D SLAM 자동 튜닝: 속도/인라이어 Pareto 최적화
그리드서치로 최적 파라미터 조합을 찾아 성능 표를 출력합니다.
"""
import argparse
import json
import sys
from pathlib import Path
from itertools import product

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from slam.visual_slam_2d import VisualSLAM2D
from slam.slam2d_common import mean_or_zero


def run_config(weights_path, input_path, **kwargs):
    """한 가지 설정으로 SLAM 실행 및 지표 수집."""
    try:
        slam = VisualSLAM2D(
            weights_path=weights_path,
            input_path=input_path,
            show_display=False,
            **kwargs,
        )
        stats = slam.process()
        return {
            "config": kwargs,
            "frames": stats.frames,
            "extract_ms": stats.avg_extract_ms,
            "match_ms": stats.avg_match_ms,
            "total_ms": stats.avg_total_ms,
            "inlier_ratio": stats.avg_inlier_ratio,
            "trajectory_length": stats.trajectory_length,
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def auto_tune(weights_path, input_path, output_dir="results_autotuner", grid_search=True):
    """그리드서치로 최적 파라미터 찾기."""
    print("==> SuperPoint 2D SLAM 자동 튜닝")
    print(f"입력: {input_path}")
    print(f"가중치: {weights_path}")
    
    if not grid_search:
        print("(--grid_search 없음: 기본값 한 번만 실행)")
        result = run_config(
            weights_path,
            input_path,
            resize=(640, 480),
            sp_scale=0.75,
            sp_interval=1,
            max_kpts=500,
            ratio_thresh=0.85,
            ransac_thresh=0.8,
            output_dir=output_dir,
        )
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    param_grid = {
        "sp_scale": [0.5, 0.75, 1.0],
        "max_kpts": [300, 500],
        "ratio_thresh": [0.75, 0.85],
        "ransac_thresh": [0.8, 1.2],
    }
    
    print(f"\n그리드서치:")
    print(f"  sp_scale: {param_grid['sp_scale']}")
    print(f"  max_kpts: {param_grid['max_kpts']}")
    print(f"  ratio_thresh: {param_grid['ratio_thresh']}")
    print(f"  ransac_thresh: {param_grid['ransac_thresh']}")
    
    configs = list(product(
        param_grid["sp_scale"],
        param_grid["max_kpts"],
        param_grid["ratio_thresh"],
        param_grid["ransac_thresh"],
    ))
    
    print(f"\n총 {len(configs)}개 설정 테스트 중...")
    
    results = []
    for idx, (sp_scale, max_kpts, ratio_thresh, ransac_thresh) in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] sp_scale={sp_scale}, max_kpts={max_kpts}, ratio={ratio_thresh}, ransac={ransac_thresh}")
        
        result = run_config(
            weights_path,
            input_path,
            resize=(640, 480),
            sp_scale=sp_scale,
            sp_interval=1,
            max_kpts=max_kpts,
            ratio_thresh=ratio_thresh,
            ransac_thresh=ransac_thresh,
            output_dir=f"{output_dir}_{idx:03d}",
        )
        
        if result:
            results.append(result)
            print(f"  inlier_ratio={result['inlier_ratio']:.3f}, total_ms={result['total_ms']:.1f}")
    
    if not results:
        print("ERROR: 모든 설정 실행 실패")
        return
    
    results_sorted = sorted(results, key=lambda r: r["inlier_ratio"], reverse=True)
    
    print("\n==> 결과 (인라이어 비율 순)")
    print(f"{'Rank':<4} {'sp_scale':<8} {'max_kpts':<9} {'ratio':<6} {'ransac':<7} {'inlier':<8} {'speed(ms)':<10}")
    print("-" * 70)
    
    for rank, result in enumerate(results_sorted[:10], 1):
        cfg = result["config"]
        print(
            f"{rank:<4} {cfg['sp_scale']:<8.2f} {cfg['max_kpts']:<9} {cfg['ratio_thresh']:<6.2f} "
            f"{cfg['ransac_thresh']:<7.1f} {result['inlier_ratio']:<8.3f} {result['total_ms']:<10.1f}"
        )
    
    summary = {
        "total_configs": len(configs),
        "successful": len(results),
        "best_inlier": results_sorted[0] if results else None,
        "all_results": results,
    }
    
    out_file = Path(output_dir) / "autotuner_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuperPoint 2D SLAM 자동 튜닝")
    parser.add_argument("--weights", type=str, required=True, help="가중치 경로")
    parser.add_argument("--input", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--output_dir", type=str, default="results_autotuner", help="결과 저장 디렉토리")
    parser.add_argument("--grid_search", action="store_true", help="그리드서치 사용 (활성화시 모든 조합 테스트)")
    opt = parser.parse_args()
    
    auto_tune(opt.weights, opt.input, opt.output_dir, grid_search=opt.grid_search)
