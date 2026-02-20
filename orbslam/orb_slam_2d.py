from __future__ import annotations

import time

import cv2
import numpy as np

from slam.slam2d_common import (
    Slam2DStats,
    compose_pose2d,
    mean_or_zero,
    path_length,
    render_topdown_map,
    save_run_artifacts,
)


class ORBSLAM2D:
    def __init__(
        self,
        input_path: str,
        resize: tuple[int, int] = (640, 480),
        nfeatures: int = 1200,
        max_matches: int = 500,
        motion_scale: float = 1.0,
        output_dir: str = "results_orbslam_2d",
        show_display: bool = True,
    ):
        self.input_path = str(input_path)
        self.width = int(resize[0])
        self.height = int(resize[1])
        self.nfeatures = int(nfeatures)
        self.max_matches = int(max_matches)
        self.motion_scale = float(motion_scale)
        self.output_dir = output_dir
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

    def process(self) -> Slam2DStats:
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {self.input_path}")

        prev_kpts = None
        prev_desc = None

        pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        trajectory = [pose[:2].copy()]
        map_points = []

        extract_ms = []
        match_ms = []
        total_ms = []
        kpts_count = []
        matches_count = []
        inliers_count = []
        inlier_ratio_list = []

        frame_idx = 0
        if self.show_display:
            cv2.namedWindow("ORB 2D SLAM", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ORB 2D SLAM", 960, 540)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            frame = cv2.resize(frame, (self.width, self.height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            fe_t0 = time.perf_counter()
            cv_kpts, desc = self.extractor.detectAndCompute(gray, None)
            fe_t1 = time.perf_counter()

            curr_kpts = np.array([kp.pt for kp in cv_kpts], dtype=np.float64) if cv_kpts else np.empty((0, 2), dtype=np.float64)

            m_t0 = time.perf_counter()
            pairs = []
            if prev_desc is not None and desc is not None and len(prev_kpts) > 0 and len(curr_kpts) > 0:
                raw = self.matcher.match(prev_desc, desc)
                raw = sorted(raw, key=lambda m: m.distance)
                pairs = raw[: self.max_matches]
            m_t1 = time.perf_counter()

            inliers = 0
            inlier_ratio = 0.0

            if len(pairs) >= 8:
                idx_prev = np.array([m.queryIdx for m in pairs], dtype=np.int32)
                idx_curr = np.array([m.trainIdx for m in pairs], dtype=np.int32)

                p1 = prev_kpts[idx_prev].astype(np.float64)
                p2 = curr_kpts[idx_curr].astype(np.float64)

                E, emask = cv2.findEssentialMat(
                    p2,
                    p1,
                    self.K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0,
                )
                if E is not None:
                    _, R, t, pose_mask = cv2.recoverPose(E, p2, p1, self.K)
                    inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0
                    inlier_ratio = inliers / max(len(pairs), 1)

                    t = t[:, 0]
                    if np.isfinite(t).all():
                        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
                        delta_local = np.array([t[0], t[2]], dtype=np.float64)
                        norm = np.linalg.norm(delta_local)
                        if norm > 1e-6:
                            delta_local = (delta_local / norm) * self.motion_scale
                        pose = compose_pose2d(pose, delta_local, yaw)
                        trajectory.append(pose[:2].copy())

                        sample = p2[::2] if len(p2) > 120 else p2
                        centered = sample - np.array([self.cx, self.cy], dtype=np.float64)
                        local = np.stack([centered[:, 0], centered[:, 1] * 0.0], axis=1) / max(self.width, 1)
                        c = np.cos(pose[2])
                        s = np.sin(pose[2])
                        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
                        world = (rot @ local.T).T + pose[:2]
                        map_points.append(world)
                else:
                    trajectory.append(pose[:2].copy())
            else:
                trajectory.append(pose[:2].copy())

            t1 = time.perf_counter()

            extract_ms.append((fe_t1 - fe_t0) * 1000.0)
            match_ms.append((m_t1 - m_t0) * 1000.0)
            total_ms.append((t1 - t0) * 1000.0)
            kpts_count.append(int(len(curr_kpts)))
            matches_count.append(int(len(pairs)))
            inliers_count.append(inliers)
            inlier_ratio_list.append(inlier_ratio)

            if self.show_display:
                vis = frame.copy()
                cv2.drawKeypoints(frame, cv_kpts[:250], vis, color=(0, 255, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
                cv2.putText(vis, f"Frame: {frame_idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"Matches: {len(pairs)} Inliers: {inliers}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
                cv2.imshow("ORB 2D SLAM", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_kpts = curr_kpts
            prev_desc = desc
            frame_idx += 1

        cap.release()
        if self.show_display:
            cv2.destroyAllWindows()

        traj_xy = np.asarray(trajectory, dtype=np.float64)
        map_xy = np.vstack(map_points) if len(map_points) > 0 else np.empty((0, 2), dtype=np.float64)
        topdown = render_topdown_map(traj_xy, map_xy)

        stats = Slam2DStats(
            name="orbslam_2d",
            frames=frame_idx,
            avg_extract_ms=mean_or_zero(extract_ms),
            avg_match_ms=mean_or_zero(match_ms),
            avg_total_ms=mean_or_zero(total_ms),
            avg_kpts=mean_or_zero(kpts_count),
            avg_matches=mean_or_zero(matches_count),
            avg_inliers=mean_or_zero(inliers_count),
            avg_inlier_ratio=mean_or_zero(inlier_ratio_list),
            trajectory_length=path_length(traj_xy),
            output_dir=self.output_dir,
        )
        save_run_artifacts(self.output_dir, stats, traj_xy, topdown)
        return stats
