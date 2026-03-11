"""SLAM 시각화 모듈 — 2D 실시간 뷰 + 3D 최종 결과 렌더링."""
import logging
import os

import cv2
import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    o3d = None
    HAS_O3D = False

from slam.slam_utils import create_camera_frustum, get_height_color

logger = logging.getLogger(__name__)


class SLAMVisualizer:
    """2D 실시간 디버그 뷰 및 3D 최종 결과 렌더링을 담당합니다."""

    def __init__(self, viz_cfg, save_dir: str, W: int, H: int):
        self.cfg = viz_cfg
        self.save_dir = save_dir
        self.W = W
        self.H = H
        self.enabled = viz_cfg.enabled

        if self.enabled:
            try:
                cv2.namedWindow('Processing', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Processing', *viz_cfg.window_size)
            except cv2.error:
                logger.warning("Display not available, disabling visualization")
                self.enabled = False

    def show_live(self, img_curr, kpts, frame_idx) -> float:
        """2D 키포인트 오버레이를 표시합니다. vis_ms를 반환합니다."""
        import time
        vis_t0 = time.perf_counter()
        if self.enabled:
            img_vis = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            for kp in kpts:
                cv2.circle(img_vis, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)
            cv2.putText(img_vis, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Processing', img_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return -1.0  # quit signal
        vis_t1 = time.perf_counter()
        return (vis_t1 - vis_t0) * 1000.0

    def render_final(self, slam_map, keyframes, all_map_points, all_map_colors):
        """3D 포인트 클라우드 + 궤적 + 카메라 프러스텀을 렌더링합니다."""
        if not HAS_O3D:
            logger.warning("open3d not available - 3D visualization skipped")
            return

        # 1. MapPoint로 포인트 클라우드 재구축
        rebuilt_points = []
        rebuilt_colors = []

        active_mps = list(slam_map.map_points.values())
        logger.info("Rendering %d active MapPoints from Map (BA optimized)...", len(active_mps))

        if len(active_mps) > 0:
            world_pts = np.array([mp.pos3d for mp in active_mps])
            cols = get_height_color(world_pts[:, 1])
            rebuilt_points.append(world_pts)
            rebuilt_colors.append(cols)

        if not rebuilt_points:
            if not all_map_points:
                logger.warning("No points generated.")
                return
            points = np.vstack(all_map_points)
            colors = np.vstack(all_map_colors)
        else:
            points = np.vstack(rebuilt_points)
            colors = np.vstack(rebuilt_colors)
            logger.info("Rendering %d points from Map across %d keyframes.", len(points), len(keyframes))

        # 2. 포인트 클라우드 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=self.cfg.voxel_size)

        # 3. 경로선
        traj_pts = np.array([pose[:3, 3] for pose in keyframes])
        traj_pts[:, 1] = 0.0
        if len(traj_pts) < 2:
            return

        lines = [[i, i + 1] for i in range(len(traj_pts) - 1)]
        traj_line = o3d.geometry.LineSet()
        traj_line.points = o3d.utility.Vector3dVector(traj_pts)
        traj_line.lines = o3d.utility.Vector2iVector(lines)
        traj_line.paint_uniform_color([0, 1, 0])

        # 4. 키프레임 카메라 프러스텀
        vis_geoms = [pcd, traj_line]
        logger.info("Generating %d Keyframes...", len(keyframes))

        for pose in keyframes:
            frustum = create_camera_frustum(scale=0.5, color=[0, 0, 1])
            frustum.transform(pose)
            vis_geoms.append(frustum)

        # 5. 최종 뷰어 실행
        logger.info("Visualization Ready!")
        vis = o3d.visualization.Visualizer()
        vis.create_window("SuperPoint SLAM Result", width=1280, height=720)

        vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.05])
        vis.get_render_option().point_size = self.cfg.point_size

        for geom in vis_geoms:
            vis.add_geometry(geom)

        ctr = vis.get_view_control()
        ctr.set_lookat(traj_pts[len(traj_pts) // 2])
        ctr.set_front([-0.5, -1.0, -0.5])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.5)

        # PLY 저장
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "final_slam_map.ply"), pcd)
        logger.info("Point Cloud saved to: %s", os.path.join(self.save_dir, "final_slam_map.ply"))

        # 2D Top-Down Map
        try:
            from .slam2d_common import render_topdown_map
            traj_2d = traj_pts[:, [0, 2]]
            map_3d = np.asarray(pcd.points)
            map_2d = map_3d[:, [0, 2]] if len(map_3d) > 0 else np.empty((0, 2))
            map_img = render_topdown_map(traj_2d, map_2d)
            cv2.imwrite(os.path.join(self.save_dir, "topdown_map.png"), map_img)
            np.savetxt(os.path.join(self.save_dir, "trajectory_xy.txt"), traj_2d, fmt="%.4f")
            logger.info("Topdown Map saved to: %s", os.path.join(self.save_dir, "topdown_map.png"))
        except Exception as e:
            logger.error("Could not save Top-down map: %s", e)

        vis.run()
        vis.destroy_window()

    def close(self):
        """OpenCV 윈도우를 정리합니다."""
        cv2.destroyAllWindows()
