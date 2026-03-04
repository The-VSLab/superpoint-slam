from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from slam.slam2d_common import render_topdown_map


@dataclass
class ORBSLAM3DStats:
    frames: int
    points: int
    output_dir: str


class ORBSLAM3D:
    def __init__(
        self,
        input_path: str,
        resize: tuple[int, int] = (640, 480),
        nfeatures: int = 1500,
        max_matches: int = 500,
        output_dir: str = "path_final",
        show_display: bool = False,
    ):
        self.input_path = str(input_path)
        self.width = int(resize[0])
        self.height = int(resize[1])
        self.nfeatures = int(nfeatures)
        self.max_matches = int(max_matches)
        self.output_dir = str(output_dir)
        self.show_display = bool(show_display)

        self.focal = max(self.width, self.height) * 0.8
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.K = np.array(
            [[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self.extractor = cv2.ORB_create(nfeatures=self.nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        os.makedirs(self.output_dir, exist_ok=True)

    def _triangulate(self, p1: np.ndarray, p2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t.reshape(3, 1)))
        pts_4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        pts_3d = (pts_4d[:3] / (pts_4d[3] + 1e-8)).T
        return pts_3d

    def _show_final_view(self, points: np.ndarray, traj_3d_arr: np.ndarray):
        vis_geoms = []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(points) > 0:
            pcd = pcd.voxel_down_sample(voxel_size=0.1)
            vis_geoms.append(pcd)

        if len(traj_3d_arr) >= 2:
            lines = [[i, i + 1] for i in range(len(traj_3d_arr) - 1)]
            traj_line = o3d.geometry.LineSet()
            traj_line.points = o3d.utility.Vector3dVector(traj_3d_arr)
            traj_line.lines = o3d.utility.Vector2iVector(lines)
            traj_line.paint_uniform_color([0.0, 1.0, 0.0])
            vis_geoms.append(traj_line)

        if len(vis_geoms) == 0:
            return

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("ORB SLAM 3D Result", width=1280, height=720)
        vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.05])
        vis.get_render_option().point_size = 3.0

        for geom in vis_geoms:
            vis.add_geometry(geom)

        ctr = vis.get_view_control()
        if len(traj_3d_arr) > 0:
            ctr.set_lookat(traj_3d_arr[len(traj_3d_arr) // 2])
        ctr.set_front([-0.5, -1.0, -0.5])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.5)

        def _close(_):
            vis.close()
            return False

        vis.register_key_callback(ord("Q"), _close)
        vis.register_key_callback(ord("q"), _close)

        print("==> Visualization Ready! (q를 누르면 종료)")
        vis.run()
        vis.destroy_window()

    def process(self) -> ORBSLAM3DStats:
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {self.input_path}")

        prev_kpts = np.empty((0, 2), dtype=np.float64)
        prev_desc = None
        prev_pose = np.eye(4, dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)

        map_points_world = []
        map_colors = []
        traj_3d = [pose[:3, 3].copy()]

        frame_idx = 0
        if self.show_display:
            cv2.namedWindow("ORB 3D SLAM", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ORB 3D SLAM", 960, 540)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.width, self.height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv_kpts, desc = self.extractor.detectAndCompute(gray, None)
            curr_kpts = (
                np.array([kp.pt for kp in cv_kpts], dtype=np.float64)
                if cv_kpts
                else np.empty((0, 2), dtype=np.float64)
            )

            inliers = 0
            pairs = []
            if prev_desc is not None and desc is not None and len(prev_kpts) > 0 and len(curr_kpts) > 0:
                raw = self.matcher.match(prev_desc, desc)
                raw = sorted(raw, key=lambda m: m.distance)
                pairs = raw[: self.max_matches]

            if len(pairs) >= 8:
                idx_prev = np.array([m.queryIdx for m in pairs], dtype=np.int32)
                idx_curr = np.array([m.trainIdx for m in pairs], dtype=np.int32)
                p1 = prev_kpts[idx_prev].astype(np.float64)
                p2 = curr_kpts[idx_curr].astype(np.float64)

                E, _ = cv2.findEssentialMat(
                    p1,
                    p2,
                    self.K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0,
                )

                if E is not None:
                    _, R, t, pose_mask = cv2.recoverPose(E, p1, p2, self.K)
                    if pose_mask is not None:
                        inlier_mask = pose_mask.reshape(-1) > 0
                        inliers = int(np.count_nonzero(inlier_mask))
                    else:
                        inlier_mask = np.ones((len(p1),), dtype=bool)

                    if inliers >= 8:
                        p1_in = p1[inlier_mask]
                        p2_in = p2[inlier_mask]

                        pts_prev = self._triangulate(p1_in, p2_in, R, t[:, 0])
                        valid_depth = np.isfinite(pts_prev).all(axis=1) & (pts_prev[:, 2] > 0)

                        if np.any(valid_depth):
                            pts_prev = pts_prev[valid_depth]
                            p2_vis = np.round(p2_in[valid_depth]).astype(np.int32)
                            p2_vis[:, 0] = np.clip(p2_vis[:, 0], 0, self.width - 1)
                            p2_vis[:, 1] = np.clip(p2_vis[:, 1], 0, self.height - 1)

                            world = (prev_pose[:3, :3] @ pts_prev.T).T + prev_pose[:3, 3]
                            colors = frame[p2_vis[:, 1], p2_vis[:, 0], ::-1].astype(np.float32) / 255.0

                            map_points_world.append(world)
                            map_colors.append(colors)

                        rel = np.eye(4, dtype=np.float64)
                        rel[:3, :3] = R
                        rel[:3, 3] = t[:, 0]
                        prev_pose = pose.copy()
                        pose = pose @ rel

            traj_3d.append(pose[:3, 3].copy())

            if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                print(
                    f"[Frame {frame_idx + 1:4d}] Keypts: {len(curr_kpts):4d} | "
                    f"Matches: {len(pairs):3d} | Inliers: {inliers:3d}"
                )

            if self.show_display:
                vis = frame.copy()
                cv2.drawKeypoints(frame, cv_kpts[:250], vis, color=(255, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
                cv2.putText(vis, f"Frame: {frame_idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"Matches: {len(pairs)} Inliers: {inliers}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
                cv2.imshow("ORB 3D SLAM", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_kpts = curr_kpts
            prev_desc = desc
            frame_idx += 1

        cap.release()
        if self.show_display:
            cv2.destroyAllWindows()

        if len(map_points_world) > 0:
            points = np.vstack(map_points_world)
            colors = np.vstack(map_colors)
        else:
            points = np.empty((0, 3), dtype=np.float64)
            colors = np.empty((0, 3), dtype=np.float64)

        traj_3d_arr = np.asarray(traj_3d, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(colors) == len(points) and len(points) > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, "final_slam_map.ply"), pcd)

        traj_xz = traj_3d_arr[:, [0, 2]] if len(traj_3d_arr) > 0 else np.empty((0, 2), dtype=np.float64)
        map_xz = points[:, [0, 2]] if len(points) > 0 else np.empty((0, 2), dtype=np.float64)
        map_img = render_topdown_map(traj_xz, map_xz)
        cv2.imwrite(os.path.join(self.output_dir, "topdown_map.png"), map_img)
        np.savetxt(os.path.join(self.output_dir, "trajectory_xy.txt"), traj_xz, fmt="%.4f")

        print(" -> Point Cloud saved to:", os.path.join(self.output_dir, "final_slam_map.ply"))
        print(" -> Topdown Map saved to:", os.path.join(self.output_dir, "topdown_map.png"))

        self._show_final_view(points, traj_3d_arr)

        return ORBSLAM3DStats(frames=frame_idx, points=int(len(points)), output_dir=self.output_dir)
