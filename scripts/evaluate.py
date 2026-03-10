"""
SLAM 성능 평가 스크립트
- ATE RMSE (Absolute Trajectory Error)
- Latency / FPS
- Memory Usage
- Inlier Ratio 통계
- 특징점 통계

사용법:
  # SLAM 결과 통계만 보기
  uv run python scripts/evaluate.py --result result/superpoint_3d_36

  # KITTI Ground Truth와 ATE RMSE 비교
  uv run python scripts/evaluate.py --result result/superpoint_3d_36 \
    --gt /path/to/poses/00.txt
"""
import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_slam_trajectory(result_dir):
    """SLAM 결과 디렉토리에서 궤적 로드"""
    traj_file = os.path.join(result_dir, "trajectory_xyz.txt")
    if os.path.exists(traj_file):
        return np.loadtxt(traj_file)
    
    # fallback: XZ 궤적 (2D)
    traj_2d_file = os.path.join(result_dir, "trajectory_xy.txt")
    if os.path.exists(traj_2d_file):
        traj_2d = np.loadtxt(traj_2d_file)
        # X, Z → X, 0, Z 로 확장
        traj_3d = np.zeros((len(traj_2d), 3))
        traj_3d[:, 0] = traj_2d[:, 0]
        traj_3d[:, 2] = traj_2d[:, 1]
        return traj_3d
    
    raise FileNotFoundError(f"궤적 파일을 찾을 수 없습니다: {result_dir}")


def load_kitti_gt(gt_path):
    """KITTI Ground Truth 포즈 파일 로드 (3x4 행렬 → XYZ 궤적)"""
    poses = []
    with open(gt_path, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T[:3, 3])  # 위치만 추출
    return np.array(poses)


def compute_ate_rmse(pred_traj, gt_traj):
    """
    ATE RMSE 계산
    - Umeyama alignment (scale + rotation + translation) 적용
    - 두 궤적의 길이가 다르면 짧은 쪽에 맞춤
    """
    n = min(len(pred_traj), len(gt_traj))
    pred = pred_traj[:n].copy()
    gt = gt_traj[:n].copy()
    
    # Umeyama Alignment (최적 스케일 + 회전 + 이동 정렬)
    pred_aligned = umeyama_alignment(pred, gt)
    
    # 오차 계산
    errors = np.linalg.norm(pred_aligned - gt, axis=1)
    
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    max_err = np.max(errors)
    
    return {
        "ATE_RMSE_m": round(float(rmse), 4),
        "ATE_Mean_m": round(float(mean_err), 4),
        "ATE_Median_m": round(float(median_err), 4),
        "ATE_Max_m": round(float(max_err), 4),
        "num_poses_compared": int(n),
    }


def umeyama_alignment(pred, gt):
    """
    Umeyama alignment: pred를 gt에 최적 정렬 (scale + rotation + translation)
    S. Umeyama, "Least-squares estimation of transformation parameters 
    between two point patterns", IEEE PAMI, 1991
    """
    n = len(pred)
    
    # 중심 이동
    pred_mean = pred.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean
    
    # 공분산 행렬
    W = (gt_centered.T @ pred_centered) / n
    U, D, Vt = np.linalg.svd(W)
    
    # 반사 보정
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    
    # 회전
    R = U @ S @ Vt
    
    # 스케일
    pred_var = np.sum(pred_centered ** 2) / n
    scale = np.sum(D * np.diag(S)) / pred_var
    
    # 이동
    t = gt_mean - scale * R @ pred_mean
    
    # 정렬된 궤적
    aligned = scale * (pred @ R.T) + t
    
    return aligned


def load_stats(result_dir):
    """SLAM 통계 JSON 로드"""
    stats_file = os.path.join(result_dir, "slam_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return None


def print_report(result_dir, ate_result=None, stats=None):
    """종합 평가 보고서 출력"""
    print(f"\n{'='*70}")
    print(f"  📊 SLAM 성능 평가 보고서")
    print(f"{'='*70}")
    print(f"  결과 디렉토리: {result_dir}")
    print(f"{'='*70}\n")
    
    # 1. ATE RMSE
    if ate_result:
        print(f"  📐 ATE (Absolute Trajectory Error)")
        print(f"  {'─'*50}")
        print(f"    RMSE:    {ate_result['ATE_RMSE_m']:.4f} m")
        print(f"    Mean:    {ate_result['ATE_Mean_m']:.4f} m")
        print(f"    Median:  {ate_result['ATE_Median_m']:.4f} m")
        print(f"    Max:     {ate_result['ATE_Max_m']:.4f} m")
        print(f"    비교 포즈: {ate_result['num_poses_compared']}개")
        print()
    
    if stats:
        # 2. Latency / FPS
        if 'latency' in stats:
            lat = stats['latency']
            print(f"  ⏱️  Latency")
            print(f"  {'─'*50}")
            print(f"    Avg Total:       {lat['avg_total_ms']:.2f} ms")
            print(f"    Avg SuperPoint:  {lat['avg_sp_ms']:.2f} ms")
            print(f"    Avg Matching:    {lat['avg_match_ms']:.2f} ms")
            print(f"    FPS:             {lat['fps']:.2f}")
            print(f"    총 프레임:        {lat['total_frames']}개")
            print()
        
        # 3. Memory
        if 'memory' in stats:
            mem = stats['memory']
            print(f"  💾 Memory Usage")
            print(f"  {'─'*50}")
            print(f"    모델 파라미터:   {mem['model_params_M']:.2f} M")
            print(f"    모델 크기:       {mem['model_size_MB']:.2f} MB")
            print(f"    맵 포인트:       {mem['map_points']}개")
            print(f"    키프레임:        {mem['keyframes']}개")
            print()
        
        # 4. Inlier Ratio
        if 'inlier_stats' in stats:
            inl = stats['inlier_stats']
            print(f"  🎯 Inlier Ratio (Essential Matrix)")
            print(f"  {'─'*50}")
            print(f"    평균:   {inl['mean_ratio']:.3f}")
            print(f"    중앙값: {inl['median_ratio']:.3f}")
            print(f"    최소:   {inl['min_ratio']:.3f}")
            print(f"    최대:   {inl['max_ratio']:.3f}")
            print()
        
        # 5. 특징점 통계
        if 'keypoint_stats' in stats:
            kp = stats['keypoint_stats']
            print(f"  🔑 특징점 통계")
            print(f"  {'─'*50}")
            print(f"    평균:   {kp['mean_kpts']:.0f}개/프레임")
            print(f"    최소:   {kp['min_kpts']}개")
            print(f"    최대:   {kp['max_kpts']}개")
            print()
    else:
        print("  ⚠️  slam_stats.json이 없습니다.")
        print("  최신 버전으로 SLAM을 다시 실행하면 자동으로 생성됩니다.\n")
    
    print(f"{'='*70}\n")
    
    # 엑셀 저장을 위한 딕셔너리 생성
    report_dict = {
        "Directory": os.path.basename(result_dir)
    }
    
    if ate_result:
        report_dict.update({
            "ATE RMSE (m)": ate_result['ATE_RMSE_m'],
            "ATE Mean (m)": ate_result['ATE_Mean_m'],
            "ATE Median (m)": ate_result['ATE_Median_m'],
            "ATE Max (m)": ate_result['ATE_Max_m'],
            "Compared Poses": ate_result['num_poses_compared']
        })
        
    if stats:
        if 'latency' in stats:
            lat = stats['latency']
            report_dict.update({
                "FPS": lat['fps'],
                "Avg Total (ms)": lat['avg_total_ms'],
                "Avg SuperPoint (ms)": lat['avg_sp_ms'],
                "Avg Matching (ms)": lat['avg_match_ms'],
                "Total Frames": lat['total_frames']
            })
            
        if 'memory' in stats:
            mem = stats['memory']
            report_dict.update({
                "Model Params (M)": mem['model_params_M'],
                "Model Size (MB)": mem['model_size_MB'],
                "Map Points": mem['map_points'],
                "Keyframes": mem['keyframes']
            })
            
        if 'inlier_stats' in stats:
            inl = stats['inlier_stats']
            report_dict.update({
                "Inlier Mean": inl['mean_ratio'],
                "Inlier Min": inl['min_ratio'],
                "Inlier Max": inl['max_ratio']
            })
            
        if 'keypoint_stats' in stats:
            kp = stats['keypoint_stats']
            report_dict.update({
                "Kpts Mean": kp['mean_kpts'],
                "Kpts Min": kp['min_kpts'],
                "Kpts Max": kp['max_kpts']
            })
            
    return report_dict


def save_to_csv(report_dict, output_path):
    """결과 딕셔너리를 CSV 파일로 저장 (누적 저장 지원)"""
    df_new = pd.DataFrame([report_dict])
    
    try:
        if os.path.exists(output_path):
            df_existing = pd.read_csv(output_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # 중복 실행 방지 (동일 디렉토리명을 가진 가장 최신 행 덮어쓰기)
            df_combined = df_combined.drop_duplicates(subset=['Directory'], keep='last')
            df_combined.to_csv(output_path, index=False)
            print(f"  💾 CSV 파일이 업데이트 되었습니다: {output_path}")
        else:
            df_new.to_csv(output_path, index=False)
            print(f"  💾 CSV 파일이 새로 생성되었습니다: {output_path}")
    except Exception as e:
        print(f"  ⚠️ CSV 저장 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="SLAM 성능 평가")
    parser.add_argument("--result", type=str, required=True,
                        help="SLAM 결과 디렉토리 (예: result/superpoint_3d_36)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground Truth 포즈 파일 (KITTI 형식, 예: poses/00.txt)")
    parser.add_argument("--csv", type=str, default=None,
                        help="결과를 누적 저장할 CSV 파일 경로 (예: VSLab_Performance.csv)")
    opt = parser.parse_args()
    
    # 통계 로드
    stats = load_stats(opt.result)
    
    # ATE 계산
    ate_result = None
    if opt.gt:
        try:
            pred_traj = load_slam_trajectory(opt.result)
            gt_traj = load_kitti_gt(opt.gt)
            ate_result = compute_ate_rmse(pred_traj, gt_traj)
        except Exception as e:
            print(f"  ⚠️ ATE 계산 실패: {e}")
    
    # 보고서 터미널 출력 및 딕셔너리 반환
    report_dict = print_report(opt.result, ate_result, stats)
    
    # CSV 저장
    if opt.csv:
        save_to_csv(report_dict, opt.csv)
    else:
        # 기본 파일명으로 result 디렉토리 안에 생성
        default_csv_path = os.path.join(opt.result, "performance_report.csv")
        save_to_csv(report_dict, default_csv_path)


if __name__ == "__main__":
    main()
