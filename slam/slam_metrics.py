"""SLAM 성능 메트릭 수집 및 보고 모듈."""
import csv
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class SLAMMetrics:
    """프레임별 타이밍, 카운트, CSV 로깅, JSON 통계를 관리합니다."""

    def __init__(self, save_dir: str, logging_cfg):
        self.save_dir = save_dir

        # Timing lists (ms)
        self.sp_ms_list = []
        self.match_ms_list = []
        self.total_ms_list = []
        self.clahe_ms_list = []
        self.flow_ms_list = []
        self.filter_ms_list = []
        self.local_map_ms_list = []
        self.pnp_ms_list = []
        self.triangulation_ms_list = []
        self.visualization_ms_list = []

        # Count lists
        self.kpts_list = []
        self.matches_list = []
        self.inliers_list = []
        self.inlier_ratio_list = []
        self.map_points_added_list = []

        self.ba_result = None

        # CSV 로깅
        self._csv_writer = None
        self._csv_file = None
        if logging_cfg.csv_per_frame:
            csv_path = os.path.join(save_dir, logging_cfg.csv_path)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "frame", "sp_ms", "match_ms", "clahe_ms", "flow_ms", "filter_ms",
                "local_map_ms", "pnp_ms", "tri_ms", "vis_ms", "total_ms",
                "kpts", "matches", "inliers", "inlier_ratio", "method",
            ])

    def record_frame(self, frame_idx, timings: dict, counts: dict, method: str):
        """프레임별 타이밍/카운트를 기록합니다.

        Args:
            frame_idx: 현재 프레임 번호
            timings: dict with keys sp_ms, match_ms, clahe_ms, flow_ms, filter_ms,
                     local_map_ms, pnp_ms, triangulation_ms, vis_ms, total_ms
            counts: dict with keys kpts, matches, inliers, inlier_ratio, map_points_added
            method: "PnP" or "DR" etc.
        """
        self.sp_ms_list.append(timings["sp_ms"])
        self.match_ms_list.append(timings["match_ms"])
        self.total_ms_list.append(timings["total_ms"])
        self.clahe_ms_list.append(timings["clahe_ms"])
        self.flow_ms_list.append(timings["flow_ms"])
        self.filter_ms_list.append(timings["filter_ms"])
        self.local_map_ms_list.append(timings["local_map_ms"])
        self.pnp_ms_list.append(timings["pnp_ms"])
        self.triangulation_ms_list.append(timings["triangulation_ms"])
        self.visualization_ms_list.append(timings["vis_ms"])

        self.kpts_list.append(counts["kpts"])
        self.matches_list.append(counts["matches"])
        self.inliers_list.append(counts["inliers"])
        self.inlier_ratio_list.append(counts["inlier_ratio"])
        self.map_points_added_list.append(counts["map_points_added"])

        if self._csv_writer is not None:
            t = timings
            c = counts
            self._csv_writer.writerow([
                frame_idx,
                f"{t['sp_ms']:.2f}", f"{t['match_ms']:.2f}", f"{t['clahe_ms']:.2f}",
                f"{t['flow_ms']:.2f}", f"{t['filter_ms']:.2f}", f"{t['local_map_ms']:.2f}",
                f"{t['pnp_ms']:.2f}", f"{t['triangulation_ms']:.2f}", f"{t['vis_ms']:.2f}",
                f"{t['total_ms']:.2f}", c["kpts"], c["matches"], c["inliers"],
                f"{c['inlier_ratio']:.3f}", method,
            ])

    def close(self):
        """CSV 파일 핸들을 닫습니다."""
        if self._csv_file is not None:
            self._csv_file.close()
            logger.info("Per-frame CSV saved to: %s", os.path.join(self.save_dir, "frame_log.csv"))

    def print_summary(self, fe_net, all_map_points, keyframes, traj_points, jetson_scale=None):
        """성능 요약을 로깅하고 JSON/궤적 파일을 저장합니다."""
        if not self.total_ms_list:
            return

        avg_total = float(np.mean(self.total_ms_list))
        avg_sp = float(np.mean(self.sp_ms_list))
        avg_match = float(np.mean(self.match_ms_list))
        avg_clahe = float(np.mean(self.clahe_ms_list))
        avg_flow = float(np.mean(self.flow_ms_list))
        avg_filter = float(np.mean(self.filter_ms_list))
        avg_local_map = float(np.mean(self.local_map_ms_list))
        avg_pnp = float(np.mean(self.pnp_ms_list))
        avg_triangulation = float(np.mean(self.triangulation_ms_list))
        avg_vis = float(np.mean(self.visualization_ms_list))
        fps = 1000.0 / max(avg_total, 1e-6)

        accounted = avg_sp + avg_match + avg_clahe + avg_flow + avg_filter + avg_local_map + avg_pnp + avg_triangulation + avg_vis
        other = avg_total - accounted

        logger.info("Performance Summary")
        logger.info("   Avg total:         %.2f ms  (FPS: %.2f)", avg_total, fps)
        logger.info("   Avg SuperPoint:    %.2f ms (%.1f%%)", avg_sp, avg_sp / avg_total * 100)
        logger.info("   Avg Match:         %.2f ms (%.1f%%)", avg_match, avg_match / avg_total * 100)
        logger.info("   Avg CLAHE:         %.2f ms (%.1f%%)", avg_clahe, avg_clahe / avg_total * 100)
        logger.info("   Avg Optical Flow:  %.2f ms (%.1f%%)", avg_flow, avg_flow / avg_total * 100)
        logger.info("   Avg Filter:        %.2f ms (%.1f%%)", avg_filter, avg_filter / avg_total * 100)
        logger.info("   Avg Local Map:     %.2f ms (%.1f%%)", avg_local_map, avg_local_map / avg_total * 100)
        logger.info("   Avg PnP:           %.2f ms (%.1f%%)", avg_pnp, avg_pnp / avg_total * 100)
        logger.info("   Avg Triangulation: %.2f ms (%.1f%%)", avg_triangulation, avg_triangulation / avg_total * 100)
        logger.info("   Avg Visualization: %.2f ms (%.1f%%)", avg_vis, avg_vis / avg_total * 100)
        logger.info("   Other/Overhead:    %.2f ms (%.1f%%)", other, other / avg_total * 100)

        if jetson_scale is not None:
            jetson_fps = fps * jetson_scale
            logger.info("Jetson Nano est. FPS: %.2f (scale=%s)", jetson_fps, jetson_scale)

        # ── 통계 JSON 저장 ──
        model_params = sum(p.numel() for p in fe_net.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in fe_net.parameters()) / (1024 * 1024)

        total_map_pts = sum(len(pts) for pts in all_map_points) if all_map_points else 0

        valid_ratios = [r for r in self.inlier_ratio_list if r > 0]

        stats = {
            "latency": {
                "avg_total_ms": round(avg_total, 2),
                "avg_sp_ms": round(avg_sp, 2),
                "avg_match_ms": round(avg_match, 2),
                "avg_clahe_ms": round(avg_clahe, 2),
                "avg_flow_ms": round(avg_flow, 2),
                "avg_filter_ms": round(avg_filter, 2),
                "avg_local_map_ms": round(avg_local_map, 2),
                "avg_pnp_ms": round(avg_pnp, 2),
                "avg_triangulation_ms": round(avg_triangulation, 2),
                "avg_visualization_ms": round(avg_vis, 2),
                "fps": round(fps, 2),
                "total_frames": len(self.total_ms_list),
            },
            "memory": {
                "model_params_M": round(model_params / 1e6, 2),
                "model_size_MB": round(model_size_mb, 2),
                "map_points": total_map_pts,
                "keyframes": len(keyframes),
            },
            "inlier_stats": {
                "mean_ratio": round(float(np.mean(valid_ratios)), 3) if valid_ratios else 0,
                "median_ratio": round(float(np.median(valid_ratios)), 3) if valid_ratios else 0,
                "min_ratio": round(float(np.min(valid_ratios)), 3) if valid_ratios else 0,
                "max_ratio": round(float(np.max(valid_ratios)), 3) if valid_ratios else 0,
            },
            "keypoint_stats": {
                "mean_kpts": round(float(np.mean(self.kpts_list)), 0) if self.kpts_list else 0,
                "min_kpts": int(np.min(self.kpts_list)) if self.kpts_list else 0,
                "max_kpts": int(np.max(self.kpts_list)) if self.kpts_list else 0,
            },
            "ba_stats": self.ba_result.to_dict() if self.ba_result else {},
        }

        stats_path = os.path.join(self.save_dir, "slam_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 3D 궤적 저장 (ATE 평가용)
        if traj_points:
            traj_3d = np.array(traj_points)
            np.savetxt(os.path.join(self.save_dir, "trajectory_xyz.txt"), traj_3d, fmt="%.4f")
