"""
Bundle Adjustment 모듈
g2o를 이용한 키프레임 포즈 최적화 (MapPoint는 고정).
visual_slam_3d.py와 visual_slam_drone.py에서 공통으로 사용.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import g2o
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BAResult:
    """Bundle Adjustment 결과 메트릭"""
    poses_updated: int = 0
    poses_rejected: int = 0
    edges_total: int = 0
    outliers_removed: int = 0
    iterations_used: int = 0
    final_chi2: float = 0.0
    mean_chi2_per_edge: float = 0.0
    avg_translation_drift: float = 0.0
    avg_rotation_drift: float = 0.0

    def to_dict(self) -> dict:
        return {
            "poses_updated": self.poses_updated,
            "poses_rejected": self.poses_rejected,
            "edges_total": self.edges_total,
            "outliers_removed": self.outliers_removed,
            "iterations_used": self.iterations_used,
            "final_chi2": round(self.final_chi2, 3),
            "mean_chi2_per_edge": round(self.mean_chi2_per_edge, 4),
            "avg_translation_drift": round(self.avg_translation_drift, 4),
            "avg_rotation_drift": round(self.avg_rotation_drift, 4),
        }


def run_bundle_adjustment(
    slam_map,
    K: np.ndarray,
    ba_config,
    kf_ids: List[int],
    keyframes_list: list,
    keyframe_indices: list,
    num_iterations: int = 10,
    fix_first: bool = True,
) -> Optional[BAResult]:
    """
    Bundle Adjustment 핵심 엔진.
    지정된 키프레임 ID들과 관측된 MapPoint들을 g2o로 동시 최적화합니다.

    Args:
        slam_map: Map 객체 (keyframes, map_points 포함)
        K: 3x3 카메라 내부 행렬
        ba_config: BAConfig dataclass
        kf_ids: 최적화할 키프레임 ID 리스트
        keyframes_list: self.keyframes (4x4 pose 리스트)
        keyframe_indices: self.keyframe_indices (프레임 인덱스 리스트)
        num_iterations: 최적화 반복 횟수
        fix_first: True면 첫 번째 키프레임 고정

    Returns:
        BAResult 또는 None (최적화 불가 시)
    """
    cfg = ba_config
    focal = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    if len(kf_ids) < 2:
        return None

    # 유효한 키프레임만 필터링
    valid_kfs = [slam_map.keyframes[kf_id] for kf_id in kf_ids if kf_id in slam_map.keyframes]
    if len(valid_kfs) < 2:
        return None

    # 관련 MapPoint 수집 (최소 관측 횟수 필터링)
    kf_id_set = set(kf_ids)
    involved_mps = {}
    for kf in valid_kfs:
        for kpt_idx, mp_id in kf.map_points.items():
            mp = slam_map.get_map_point(mp_id)
            if mp is not None and not mp.is_bad:
                valid_obs = sum(1 for obs_kf in mp.observations if obs_kf in kf_id_set)
                if valid_obs >= cfg.min_observations:
                    involved_mps[mp_id] = mp

    if len(involved_mps) < cfg.min_map_points:
        return None

    # ── g2o 최적화기 구축 ──
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    cam_params = g2o.CameraParameters(focal, [cx, cy], 0)
    cam_params.set_id(0)
    optimizer.add_parameter(cam_params)

    # ID 매핑
    kf_vertex_ids = {}
    mp_vertex_ids = {}
    next_id = 0

    # 1. 카메라 포즈 노드 (VertexSE3Expmap)
    for i, kf in enumerate(valid_kfs):
        v = g2o.VertexSE3Expmap()
        v.set_id(next_id)
        R_wc = kf.pose[:3, :3]
        t_wc = kf.pose[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_wc.T @ t_wc
        se3 = g2o.SE3Quat(R_cw, t_cw)
        v.set_estimate(se3)
        if fix_first and i == 0:
            v.set_fixed(True)
        optimizer.add_vertex(v)
        kf_vertex_ids[kf.id] = next_id
        next_id += 1

    # 2. MapPoint 노드 (VertexPointXYZ)
    for mp_id, mp in involved_mps.items():
        v = g2o.VertexPointXYZ()
        v.set_id(next_id)
        v.set_estimate(mp.pos3d)
        v.set_marginalized(True)
        optimizer.add_vertex(v)
        mp_vertex_ids[mp_id] = next_id
        next_id += 1

    # 3. 재투영 엣지 (EdgeProjectXYZ2UV)
    edges = []
    for mp_id, mp in involved_mps.items():
        if mp_id not in mp_vertex_ids:
            continue
        for obs_kf_id, pixel_uv in mp.observations.items():
            if obs_kf_id not in kf_id_set or obs_kf_id not in kf_vertex_ids:
                continue
            e = g2o.EdgeProjectXYZ2UV()
            e.set_vertex(0, optimizer.vertex(mp_vertex_ids[mp_id]))
            e.set_vertex(1, optimizer.vertex(kf_vertex_ids[obs_kf_id]))
            e.set_measurement(np.array(pixel_uv[:2], dtype=np.float64))
            e.set_information(np.eye(2) * cfg.reprojection_weight)
            kernel = g2o.RobustKernelHuber()
            kernel.set_delta(cfg.huber_delta)
            e.set_robust_kernel(kernel)
            e.set_parameter_id(0, 0)
            optimizer.add_edge(e)
            edges.append(e)

    edge_count = len(edges)
    if edge_count < 5:
        return None

    # ── 4. 최적화 실행 ──
    optimizer.initialize_optimization()

    if cfg.adaptive_termination:
        # 적응형 조기 종료: chi2 수렴 모니터링
        prev_chi2 = float("inf")
        actual_iters = num_iterations
        for i in range(num_iterations):
            optimizer.optimize(1)
            current_chi2 = optimizer.active_chi2()
            if i >= cfg.min_iterations:
                improvement = (prev_chi2 - current_chi2) / max(abs(prev_chi2), 1e-10)
                if improvement < cfg.convergence_threshold:
                    actual_iters = i + 1
                    logger.info("[BA] Early stop at iter %d: improvement %.6f < threshold", i, improvement)
                    break
            prev_chi2 = current_chi2
        else:
            actual_iters = num_iterations
    else:
        optimizer.optimize(num_iterations)
        actual_iters = num_iterations

    # ── 5. 2-pass outlier rejection ──
    outliers_removed = 0
    if cfg.two_pass_rejection:
        for e in edges:
            if e.chi2() > cfg.outlier_chi2_threshold:
                e.set_level(1)
                outliers_removed += 1

        if outliers_removed > 0:
            logger.debug("[BA] Pass 2: removed %d outlier edges, re-optimizing", outliers_removed)
            optimizer.initialize_optimization(0)
            optimizer.optimize(actual_iters)

    # ── 6. chi-squared 모니터링 ──
    final_chi2 = optimizer.active_chi2()
    active_edges = optimizer.active_edges()
    num_active = len(active_edges) if active_edges else 1
    mean_chi2 = final_chi2 / max(num_active, 1)

    if logger.isEnabledFor(logging.DEBUG) and active_edges:
        edge_chi2_vals = [e.chi2() for e in active_edges]
        logger.debug("[BA] Edge chi2: median=%.3f, max=%.3f", np.median(edge_chi2_vals), np.max(edge_chi2_vals))

    # ── 7. 결과 반영: 포즈 업데이트 (translation + rotation drift 검증) ──
    pose_updates = {}
    t_drifts = {}
    r_drifts = {}

    for kf_id, vid in kf_vertex_ids.items():
        v = optimizer.vertex(vid)
        if v is None:
            continue

        se3_opt = v.estimate()
        R_cw_opt = se3_opt.rotation().matrix()
        t_cw_opt = se3_opt.translation()
        R_wc_opt = R_cw_opt.T
        t_wc_opt = -R_cw_opt.T @ t_cw_opt

        pose_opt = np.eye(4)
        pose_opt[:3, :3] = R_wc_opt
        pose_opt[:3, 3] = t_wc_opt

        if kf_id in slam_map.keyframes:
            original_pose = slam_map.keyframes[kf_id].pose
            t_drift = np.linalg.norm(pose_opt[:3, 3] - original_pose[:3, 3])
            t_drifts[kf_id] = t_drift

            # 회전 drift 체크
            R_diff = original_pose[:3, :3].T @ pose_opt[:3, :3]
            trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            r_drift = np.arccos(trace_val)
            r_drifts[kf_id] = r_drift

            if t_drift < cfg.pose_drift_threshold and r_drift < cfg.rotation_drift_threshold:
                pose_updates[kf_id] = pose_opt
            else:
                logger.warning("[BA] KF %d: drift t=%.2fm r=%.2frad, skipping", kf_id, t_drift, r_drift)

    # 검증 통과한 포즈만 반영
    updated_count = 0
    for kf_id, pose_opt in pose_updates.items():
        if kf_id in slam_map.keyframes:
            slam_map.keyframes[kf_id].pose = pose_opt
            updated_count += 1
        if kf_id in keyframe_indices:
            idx = keyframe_indices.index(kf_id)
            if idx < len(keyframes_list):
                keyframes_list[idx] = pose_opt

    rejected_count = len(kf_vertex_ids) - updated_count
    avg_t_drift = np.mean(list(t_drifts.values())) if t_drifts else 0.0
    avg_r_drift = np.mean(list(r_drifts.values())) if r_drifts else 0.0

    result = BAResult(
        poses_updated=updated_count,
        poses_rejected=rejected_count,
        edges_total=edge_count,
        outliers_removed=outliers_removed,
        iterations_used=actual_iters,
        final_chi2=final_chi2,
        mean_chi2_per_edge=mean_chi2,
        avg_translation_drift=avg_t_drift,
        avg_rotation_drift=avg_r_drift,
    )

    logger.info("[BA] %d KFs, %d edges, %d iters | chi2=%.1f (mean=%.3f) | outliers=%d",
                len(valid_kfs), edge_count, actual_iters, final_chi2, mean_chi2, outliers_removed)
    logger.info("[BA] Updated: %d/%d poses (avg t_drift: %.3fm, r_drift: %.3frad)",
                updated_count, len(valid_kfs), avg_t_drift, avg_r_drift)

    return result
