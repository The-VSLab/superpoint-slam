from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class LoopClosureResult:
    match_index: int
    transform: np.ndarray
    inliers: int
    inlier_ratio: float
    matches: int


class LoopClosureManager:
    def __init__(
        self,
        matcher,
        K,
        min_frame_gap=30,
        top_k=5,
        min_inliers=30,
        min_inlier_ratio=0.25,
    ):
        self.matcher = matcher
        self.K = K
        self.min_frame_gap = int(min_frame_gap)
        self.top_k = int(top_k)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.keyframes = []

    def add_keyframe(self, frame_idx, kpts, desc):
        global_desc = self._global_descriptor(desc)
        self.keyframes.append(
            {
                "frame_idx": int(frame_idx),
                "kpts": kpts,
                "desc": desc,
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
        for _, cand_idx in candidates[: self.top_k]:
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

        p1 = cand["kpts"][matches[:, 0], :2].astype(np.float64)
        p2 = kpts[matches[:, 1], :2].astype(np.float64)

        E, mask = cv2.findEssentialMat(
            p2,
            p1,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None or mask is None:
            return None

        _, R, t, mask = cv2.recoverPose(E, p2, p1, self.K)
        inliers = int(mask.ravel().sum()) if mask is not None else 0
        inlier_ratio = inliers / max(matches.shape[0], 1)

        if inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return None

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t[:, 0]

        return LoopClosureResult(
            match_index=cand_idx,
            transform=transform,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            matches=int(matches.shape[0]),
        )
