from dataclasses import dataclass
import logging
import numpy as np
import cv2

from config.slam_config import LoopClosureConfig

logger = logging.getLogger(__name__)


@dataclass
class LoopClosureResult:
    match_index: int
    transform: np.ndarray
    inliers: int
    inlier_ratio: float
    matches: int
    scale: float  # PnP에서 구한 스케일 (Essential Matrix fallback 시 1.0)


class LoopClosureManager:
    def __init__(self, matcher, K, config: LoopClosureConfig = None):
        cfg = config or LoopClosureConfig()
        self.matcher = matcher
        self.K = K
        self.min_frame_gap = cfg.min_frame_gap
        self.top_k = cfg.top_k
        self.min_inliers = cfg.min_inliers
        self.min_inlier_ratio = cfg.min_inlier_ratio
        self.descriptor_similarity = cfg.descriptor_similarity
        self.verify_min_inliers = cfg.verify_min_inliers
        self.verify_min_inlier_ratio = cfg.verify_min_inlier_ratio
        self.verify_min_3d_points = cfg.verify_min_3d_points
        self.verify_reprojection_error = cfg.verify_reprojection_error
        self.keyframes = []

    def add_keyframe(self, frame_idx, kpts, desc, pts_3d=None):
        global_desc = self._global_descriptor(desc)
        self.keyframes.append(
            {
                "frame_idx": int(frame_idx),
                "kpts": kpts,
                "desc": desc,
                "pts_3d": pts_3d,
                "global_desc": global_desc,
            }
        )

    def find_loop(self, frame_idx, kpts, desc):
        if len(self.keyframes) < 2:
            return None

        global_desc = self._global_descriptor(desc)
        if global_desc is None:
            return None

        candidates = []
        for i, kf in enumerate(self.keyframes):
            if frame_idx - kf["frame_idx"] < self.min_frame_gap:
                continue
            if kf["global_desc"] is None:
                continue
            sim = float(np.dot(global_desc, kf["global_desc"]))
            candidates.append((sim, i))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for c in candidates if c[0] > self.descriptor_similarity]

        for sim, cand_idx in candidates[: self.top_k]:
            result = self._verify_candidate(cand_idx, kpts, desc)
            if result is not None:
                return result

        return None

    def _global_descriptor(self, desc):
        if desc is None or desc.size == 0:
            return None
        gdesc = np.mean(desc, axis=1)
        norm = np.linalg.norm(gdesc)
        if norm < 1e-6:
            return None
        return gdesc / norm

    def _verify_candidate(self, cand_idx, kpts, desc):
        cand = self.keyframes[cand_idx]
        if cand["desc"] is None or desc is None:
            return None

        matches = self.matcher.match(cand["desc"], desc)
        if matches.shape[0] < 8:
            return None

        # PnP를 위한 3D-2D 매칭 구성
        if cand["pts_3d"] is not None:
            p1_3d = cand["pts_3d"][matches[:, 0]]
            p2_2d = kpts[matches[:, 1], :2].astype(np.float64)

            valid_3d_mask = ~np.isnan(p1_3d[:, 0])
            obj_pts_c = p1_3d[valid_3d_mask].astype(np.float32)
            img_pts_c = p2_2d[valid_3d_mask].astype(np.float32)

            if len(obj_pts_c) >= self.verify_min_3d_points:

                dist_coeffs = np.zeros(4, dtype=np.float32)

                success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                    obj_pts_c, img_pts_c, self.K, dist_coeffs,
                    iterationsCount=1000,
                    reprojectionError=self.verify_reprojection_error,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success and inliers_pnp is not None:
                    inliers = len(inliers_pnp)
                    inlier_ratio = inliers / max(len(obj_pts_c), 1)

                    if inliers >= self.verify_min_inliers and inlier_ratio >= self.verify_min_inlier_ratio:
                        R, _ = cv2.Rodrigues(rvec)
                        transform = np.eye(4)
                        transform[:3, :3] = R.T
                        transform[:3, 3] = -R.T @ tvec[:, 0]

                        scale = np.linalg.norm(transform[:3, 3])
                        logger.info("[LOOP FOUND (PnP)] Frame %d <-> Curr | Inliers: %d | Scale: %.3f", cand["frame_idx"], inliers, scale)

                        return LoopClosureResult(
                            match_index=cand_idx,
                            transform=transform,
                            inliers=inliers,
                            inlier_ratio=inlier_ratio,
                            matches=int(matches.shape[0]),
                            scale=scale,
                        )

        return None
