from dataclasses import dataclass
import logging
import numpy as np
import cv2

from config.slam_config import LoopClosureConfig

logger = logging.getLogger(__name__)


@dataclass
class LoopClosureResult:
    match_index: int
    transform: np.ndarray   # 후보(from)→현재(to) 상대 포즈. app 규약: T_wc_curr = T_wc_cand @ transform
    inliers: int
    inlier_ratio: float
    matches: int
    scale: float            # 상대 병진 크기(로깅용). PGO 측정 스케일은 항상 1.0
    method: str = "pnp"     # 'pnp'(metric 병진 신뢰) 또는 'ess'(회전만 신뢰, 병진 스케일 불명)


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
        self.check_interval = max(1, cfg.check_interval)
        self.min_distinctiveness = cfg.min_distinctiveness
        self.verify_min_inliers = cfg.verify_min_inliers
        self.verify_min_inlier_ratio = cfg.verify_min_inlier_ratio
        self.verify_min_3d_points = cfg.verify_min_3d_points
        self.verify_reprojection_error = cfg.verify_reprojection_error
        self.keyframes = []
        self._kf_call_count = 0  # check_interval 카운터
        self.stats = {
            'attempts': 0,
            'above_thresh': 0,
            'pnp_tried': 0,
            'found': 0,
            'found_pnp': 0,
            'found_ess': 0,
            'max_sim': 0.0,
            'fail_few_matches': 0,
            'fail_essential': 0,       # Essential Matrix 자체 실패
            'fail_inliers': 0,         # inliers < verify_min_inliers
            'pnp_no3d': 0,             # 매칭 중 유효 3D가 verify_min_3d_points 미만
            'pnp_3d_max': 0,           # 검증 시 관측된 최대 유효 3D 개수 (진단용)
            'skipped_interval': 0,     # check_interval로 건너뜀
            'skipped_distinct': 0,     # min_distinctiveness 미달로 건너뜀
        }

    def add_keyframe(self, frame_idx, kpts, desc, pts_3d=None, pose=None):
        global_desc = self._global_descriptor(desc)
        self.keyframes.append(
            {
                "frame_idx": int(frame_idx),
                "kpts": kpts,
                "desc": desc,
                "pts_3d": pts_3d,
                "global_desc": global_desc,
                "pose": None if pose is None else np.asarray(pose, dtype=np.float64),
            }
        )

    def find_loop(self, frame_idx, kpts, desc):
        if len(self.keyframes) < 2:
            return None

        # check_interval: N 키프레임마다 한 번만 실제 검색 수행
        self._kf_call_count += 1
        if self._kf_call_count % self.check_interval != 0:
            self.stats['skipped_interval'] += 1
            return None

        global_desc = self._global_descriptor(desc)
        if global_desc is None:
            return None

        # centering: 저장된 raw mean 벡터들의 평균(mu)을 빼고 정규화해야
        # KITTI류 장면의 공통 성분이 제거되어 유사도에 변별력이 생긴다.
        valid = [(i, kf) for i, kf in enumerate(self.keyframes) if kf["global_desc"] is not None]
        if len(valid) < 2:
            return None
        G = np.stack([kf["global_desc"] for _, kf in valid])
        mu = G.mean(axis=0)
        Gc = G - mu
        Gc /= (np.linalg.norm(Gc, axis=1, keepdims=True) + 1e-9)
        q = global_desc - mu
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return None
        q /= q_norm
        sims = Gc @ q

        candidates = []
        for k, (i, kf) in enumerate(valid):
            if frame_idx - kf["frame_idx"] < self.min_frame_gap:
                continue
            candidates.append((float(sims[k]), i))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        self.stats['attempts'] += 1

        top_sim, top_idx = candidates[0]
        if top_sim > self.stats['max_sim']:
            self.stats['max_sim'] = top_sim

        logger.debug("[LoopClosure] frame %d: top_sim=%.4f (thresh=%.2f), total_cands=%d",
                     frame_idx, top_sim, self.descriptor_similarity, len(candidates))

        # min_distinctiveness: top1이 '시간적으로 떨어진' 2위 대비 충분히 높아야 함.
        # 진짜 재방문 구간에서는 top1의 이웃 키프레임들도 같이 높게 나오므로,
        # top1 근처(min_frame_gap/4 이내) 후보는 2위 비교에서 제외한다.
        if self.min_distinctiveness > 0.0 and len(candidates) >= 2:
            top_frame = self.keyframes[top_idx]["frame_idx"]
            exclusion = max(20, self.min_frame_gap // 4)
            second_sim = 0.0
            for s, i in candidates[1:]:
                if abs(self.keyframes[i]["frame_idx"] - top_frame) > exclusion:
                    second_sim = s
                    break
            if second_sim > 1e-6 and (top_sim / second_sim) < self.min_distinctiveness:
                self.stats['skipped_distinct'] += 1
                return None

        candidates = [c for c in candidates if c[0] > self.descriptor_similarity]
        if not candidates:
            return None

        self.stats['above_thresh'] += 1

        # top_k 후보 중 PnP(metric 병진+스케일)로 검증된 것을 우선한다.
        # Essential(회전만)은 어떤 후보도 PnP를 통과하지 못했을 때의 폴백.
        # (예: KITTI 07의 진짜 루프가 3D가 없는 KF 0과 매칭돼도, 이웃 후보
        #  KF 8/12에는 3D가 있어 PnP 루프를 얻을 수 있음)
        ess_result = None
        for sim, cand_idx in candidates[: self.top_k]:
            self.stats['pnp_tried'] += 1
            result = self._verify_candidate(cand_idx, kpts, desc)
            if result is None:
                continue
            if result.method == "pnp":
                self.stats['found'] += 1
                return result
            if ess_result is None:
                ess_result = result

        if ess_result is not None:
            self.stats['found'] += 1
            return ess_result
        return None

    def _global_descriptor(self, desc):
        """프레임 전체 키포인트 디스크립터 [256, N] → 원시(raw) mean-pooling 벡터.

        max-pooling은 수백 개 키포인트에서 차원별 포화로 모든 프레임 쌍의
        유사도가 ~1.0이 되어 변별력이 없음(KITTI 07 실측). mean-pooling도
        공통 성분이 지배하므로, 비교 시점에 저장된 키프레임 평균(mu)을 빼는
        centering을 적용한다 (find_loop 참조). 여기서는 raw mean만 저장."""
        if desc is None or desc.size == 0:
            return None
        gdesc = np.mean(desc, axis=1)
        if np.linalg.norm(gdesc) < 1e-9:
            return None
        return gdesc

    def _verify_candidate(self, cand_idx, kpts, desc):
        cand = self.keyframes[cand_idx]
        if cand["desc"] is None or desc is None:
            return None

        matches = self.matcher.match(cand["desc"], desc)
        if matches.shape[0] < 8:
            self.stats['fail_few_matches'] += 1
            return None

        pts1 = cand["kpts"][matches[:, 0], :2].astype(np.float64)
        pts2 = kpts[matches[:, 1], :2].astype(np.float64)

        # 1차: PnP (충분한 3D 포인트가 있을 때, 스케일 추정 가능)
        if cand["pts_3d"] is not None:
            p1_3d = cand["pts_3d"][matches[:, 0]]
            valid_3d_mask = ~np.isnan(p1_3d[:, 0])
            obj_pts = p1_3d[valid_3d_mask].astype(np.float32)
            img_pts = pts2[valid_3d_mask].astype(np.float32)

            n_valid = int(len(obj_pts))
            n_stored = int(np.count_nonzero(~np.isnan(cand["pts_3d"][:, 0])))
            if n_valid > self.stats['pnp_3d_max']:
                self.stats['pnp_3d_max'] = n_valid
            logger.debug("[LoopClosure] PnP check: cand kf has %d 3D pts, matches=%d, matched-with-3D=%d (need %d)",
                         n_stored, len(matches), n_valid, self.verify_min_3d_points)
            if n_valid < self.verify_min_3d_points:
                self.stats['pnp_no3d'] += 1

            if len(obj_pts) >= self.verify_min_3d_points:
                success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                    obj_pts, img_pts, self.K, np.zeros(4, dtype=np.float32),
                    iterationsCount=1000,
                    reprojectionError=self.verify_reprojection_error,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not success or inliers_pnp is None:
                    logger.debug("[LoopClosure] PnP solve failed (success=%s) with %d 3D pts", success, n_valid)
                if success and inliers_pnp is not None:
                    inliers = len(inliers_pnp)
                    inlier_ratio = inliers / max(len(obj_pts), 1)
                    if inliers < self.verify_min_inliers or inlier_ratio < self.verify_min_inlier_ratio:
                        logger.debug("[LoopClosure] PnP gate fail: inliers=%d/%d (need %d, ratio %.2f<%.2f)",
                                     inliers, n_valid, self.verify_min_inliers, inlier_ratio, self.verify_min_inlier_ratio)
                    if inliers >= self.verify_min_inliers and inlier_ratio >= self.verify_min_inlier_ratio:
                        # PnP는 후보의 world 3D점 기준으로 현재 카메라의 절대 포즈(T_wc_curr)를 추정.
                        R, _ = cv2.Rodrigues(rvec)
                        T_wc_curr = np.eye(4)
                        T_wc_curr[:3, :3] = R.T
                        T_wc_curr[:3, 3] = -R.T @ tvec[:, 0]
                        # PGO 엣지는 후보(from)→현재(to) 상대 포즈를 요구하므로 후보 포즈로 relativize.
                        # T_rel 규약: T_wc_curr = T_wc_cand @ T_rel  →  T_rel = inv(T_wc_cand) @ T_wc_curr
                        cand_pose = cand.get("pose")
                        if cand_pose is not None:
                            transform = np.linalg.inv(cand_pose) @ T_wc_curr
                        else:
                            transform = T_wc_curr  # 포즈 미저장 시 폴백(하위호환)
                        scale = float(np.linalg.norm(transform[:3, 3]))
                        logger.info("[LOOP FOUND (PnP)] Frame %d | Inliers: %d | RelDist: %.3f",
                                    cand["frame_idx"], inliers, scale)
                        self.stats['found_pnp'] += 1
                        return LoopClosureResult(
                            match_index=cand_idx, transform=transform,
                            inliers=inliers, inlier_ratio=inlier_ratio,
                            matches=int(matches.shape[0]), scale=scale, method="pnp",
                        )

        # 2차: Essential Matrix (스케일 드리프트에 강건, 3D 포인트 불필요)
        E, mask_E = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=2.0,
        )
        if E is None or mask_E is None:
            self.stats['fail_essential'] += 1
            return None

        inliers_e = int(np.count_nonzero(mask_E))
        inlier_ratio = inliers_e / max(len(matches), 1)

        if inliers_e < self.verify_min_inliers or inlier_ratio < self.verify_min_inlier_ratio:
            self.stats['fail_inliers'] += 1
            logger.debug("[LoopClosure] Essential fail: inliers=%d (need %d), ratio=%.2f (need %.2f)",
                         inliers_e, self.verify_min_inliers, inlier_ratio, self.verify_min_inlier_ratio)
            return None

        # recoverPose는 후보(cand)→현재(curr) 상대 회전 + '단위' 병진(스케일 불명)을 반환.
        # transform은 후보→현재 상대 포즈(T_rel). 병진 크기는 신뢰 불가 → PGO에서 회전만 반영.
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask_E)
        transform = np.eye(4)
        transform[:3, :3] = R.T
        transform[:3, 3] = (-R.T @ t)[:, 0]

        logger.info("[LOOP FOUND (Ess)] Frame %d | Inliers: %d | Ratio: %.2f (rotation-only)",
                    cand["frame_idx"], inliers_e, inlier_ratio)
        self.stats['found_ess'] += 1
        return LoopClosureResult(
            match_index=cand_idx, transform=transform,
            inliers=inliers_e, inlier_ratio=inlier_ratio,
            matches=int(matches.shape[0]), scale=1.0, method="ess",
        )
