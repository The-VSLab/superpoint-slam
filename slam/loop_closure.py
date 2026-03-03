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
    scale: float  # PnP에서 구한 스케일 (Essential Matrix fallback 시 1.0)


class LoopClosureManager:
    def __init__(
        self,
        matcher,
        K,
        min_frame_gap=15,
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
        # 0.95 이상인 유력 후보만 필터링 (불필요한 기하 연산 방지)
        candidates = [c for c in candidates if c[0] > 0.90]

        for sim, cand_idx in candidates[: self.top_k]:
            print(f"  [Loop Search] Testing Frame {self.keyframes[cand_idx]['frame_idx']} (sim: {sim:.3f})...")
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
            
            valid_mask = ~np.isnan(p1_3d[:, 0])
            obj_pts = p1_3d[valid_mask].astype(np.float32)
            img_pts = p2_2d[valid_mask].astype(np.float32)
            
            if len(obj_pts) >= 15:
                # [중요] OpenCV C++ 바인딩 오류 방지를 위해 명시적 형태 정의
                obj_pts_c = np.ascontiguousarray(obj_pts).reshape(-1, 1, 3)
                img_pts_c = np.ascontiguousarray(img_pts).reshape(-1, 1, 2)
                
                print(f"  [Loop PnP Debug] Valid 3D points: {len(obj_pts)}")
                
                dist_coeffs = np.zeros(4, dtype=np.float32)
                
                success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                    obj_pts_c, img_pts_c, self.K, dist_coeffs, 
                    iterationsCount=1000,
                    reprojectionError=15.0,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success and inliers_pnp is not None:
                    inliers = len(inliers_pnp)
                    inlier_ratio = inliers / max(len(obj_pts), 1)
                    
                    # 진짜 루프만 수락 (inlier 10개 이상, 적당한 비율)
                    if inliers >= 10 and inlier_ratio >= 0.05:
                        R, _ = cv2.Rodrigues(rvec)
                        transform = np.eye(4)
                        transform[:3, :3] = R.T
                        transform[:3, 3] = -R.T @ tvec[:, 0]
                        
                        scale = np.linalg.norm(transform[:3, 3])
                        print(f"\n🟢 [LOOP FOUND (PnP)] Frame {cand['frame_idx']} <-> Curr | Inliers: {inliers} | Scale: {scale:.3f}")
                        
                        return LoopClosureResult(
                            match_index=cand_idx,
                            transform=transform,
                            inliers=inliers,
                            inlier_ratio=inlier_ratio,
                            matches=int(matches.shape[0]),
                            scale=scale,
                        )
                    else:
                        print(f"  [Loop PnP Debug] Rejected by Thresholds: inliers={inliers}/5, ratio={inlier_ratio:.2f}/0.01")
                else:
                    print(f"  [Loop PnP Debug] solvePnPRansac failed mathematically. success={success}")
        
        # 3D 맵포인트가 불충분하거나 PnP가 실패한 경우, 스케일이 없는 Essential Matrix로는 
        # 올바른 루프 클로저(SE3) 엣지를 생성할 수 없으므로 루프를 기각합니다.
        return None
