"""포즈 추정 모듈 — PnP RANSAC + Essential/Homography 폴백."""
import logging
from typing import Optional

import cv2
import numpy as np

from slam.slam_utils import recover_scale

logger = logging.getLogger(__name__)


class PoseResult:
    """포즈 추정 결과."""
    __slots__ = ('success', 'R', 't_vec', 'inliers', 'inlier_ratio', 'mask', 'method')

    def __init__(self, success=False, R=None, t_vec=None, inliers=0,
                 inlier_ratio=0.0, mask=None, method="DR"):
        self.success = success
        self.R = R
        self.t_vec = t_vec
        self.inliers = inliers
        self.inlier_ratio = inlier_ratio
        self.mask = mask
        self.method = method


class PoseEstimator:
    """PnP RANSAC + Essential Matrix + Homography 기반 포즈 추정기."""

    def __init__(self, K: np.ndarray, pnp_cfg, motion_cfg, epipolar_cfg):
        self.K = K
        self.pnp_cfg = pnp_cfg
        self.motion_cfg = motion_cfg
        self.epipolar_cfg = epipolar_cfg

    def estimate_pnp(
        self,
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        cur_pose: np.ndarray,
        match_count: int,
        p1: Optional[np.ndarray],
        recent_speeds: list,
        last_t_vec: np.ndarray,
        frame_idx: int,
    ) -> PoseResult:
        """PnP RANSAC으로 포즈를 추정합니다.

        Args:
            obj_pts: (N, 3) 3D 월드 좌표
            img_pts: (N, 2) 2D 이미지 좌표
            cur_pose: 현재 4x4 카메라 포즈
            match_count: 전체 매칭 수 (inlier_ratio 계산용)
            p1: 이전 프레임 특징점 (마스크 생성용)
            recent_speeds: 최근 속도 기록
            last_t_vec: 이전 이동 벡터
            frame_idx: 현재 프레임 번호 (로깅용)

        Returns:
            PoseResult
        """
        cfg = self.pnp_cfg
        mcfg = self.motion_cfg

        if len(obj_pts) < cfg.min_points:
            return PoseResult()

        success, rvec, tvec_pnp, inliers_pnp = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, None,
            flags=cv2.USAC_MAGSAC,
            reprojectionError=cfg.reprojection_error,
            confidence=0.999,
            iterationsCount=1000,
        )

        if not success or inliers_pnp is None or len(inliers_pnp) <= cfg.min_inliers:
            return PoseResult()

        R_pnp, _ = cv2.Rodrigues(rvec)

        # PnP 반환값은 T_CW → T_rel (Prev→Curr)로 변환
        R_WC_new = R_pnp.T
        t_WC_new = -R_pnp.T @ tvec_pnp
        T_WC_new = np.eye(4)
        T_WC_new[:3, :3] = R_WC_new
        T_WC_new[:3, 3] = t_WC_new[:, 0]

        T_WC_prev_inv = np.linalg.inv(cur_pose)
        T_rel_mat = T_WC_prev_inv @ T_WC_new

        R_rel = T_rel_mat[:3, :3]
        t_rel = T_rel_mat[:3, 3]
        tvec_len = np.linalg.norm(t_rel)

        # 필터 1: 절대 상한
        if not (cfg.min_translation < tvec_len < cfg.max_translation):
            return PoseResult()

        # 필터 2: 속도 기반 점프 감지
        if len(recent_speeds) >= 3:
            median_speed = np.median(recent_speeds[-10:])
            speed_thresh = max(median_speed * mcfg.speed_jump_multiplier, mcfg.min_speed_thresh)
            if tvec_len > speed_thresh:
                logger.warning(
                    "frame %d: [PnP REJECTED] speed jump %.3fm > thresh %.3fm (median_speed=%.3f)",
                    frame_idx, tvec_len, speed_thresh, median_speed,
                )
                return PoseResult()

        # 필터 3: 방향 반전 체크
        last_t_norm = np.linalg.norm(last_t_vec)
        if last_t_norm > 1e-4:
            cos_angle = np.dot(t_rel, last_t_vec) / (tvec_len * last_t_norm + 1e-8)
            if cos_angle < mcfg.direction_reversal_cos:
                logger.warning("frame %d: [PnP REJECTED] direction reversal cos=%.3f", frame_idx, cos_angle)
                return PoseResult()

        inliers = len(inliers_pnp)
        inlier_ratio = inliers / max(match_count, 1)

        # PnP 성공 시 삼각측량용 마스크 생성
        mask = None
        if p1 is not None and len(p1) > 0:
            mask = np.ones(len(p1), dtype=np.uint8)

        logger.debug(
            "frame %d: [PnP] matches=%d, inliers=%d, ratio=%.3f, scale=%.4f, t=%s",
            frame_idx, match_count, inliers, inlier_ratio, tvec_len, t_rel.round(3),
        )

        return PoseResult(
            success=True, R=R_rel, t_vec=t_rel,
            inliers=inliers, inlier_ratio=inlier_ratio,
            mask=mask, method="PnP",
        )

    def estimate_epipolar(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        match_count: int,
        recent_speeds: list,
        last_t_vec: np.ndarray,
        frame_idx: int,
    ) -> PoseResult:
        """Essential Matrix + Homography 폴백으로 포즈를 추정합니다.

        Args:
            p1: 이전 프레임 특징점 (N, 2)
            p2: 현재 프레임 특징점 (N, 2)
            match_count: 전체 매칭 수
            recent_speeds: 최근 속도 기록
            last_t_vec: 이전 이동 벡터
            frame_idx: 현재 프레임 번호

        Returns:
            PoseResult
        """
        ecfg = self.epipolar_cfg

        E, mask_E = cv2.findEssentialMat(
            p2, p1, self.K, method=cv2.RANSAC, prob=0.999,
            threshold=ecfg.essential_threshold,
        )
        H, mask_H = cv2.findHomography(p2, p1, cv2.RANSAC, ecfg.homography_threshold)

        inliers_E = np.count_nonzero(mask_E) if mask_E is not None else 0
        inliers_H = np.count_nonzero(mask_H) if mask_H is not None else 0

        # Homography 시도
        use_homography = (
            inliers_H > inliers_E * ecfg.homography_selection_ratio
            and inliers_H > ecfg.homography_min_inliers
        )

        if use_homography and H is not None:
            result = self._solve_homography(
                H, p1, p2, mask_H, inliers_H, match_count,
                recent_speeds, last_t_vec, frame_idx,
            )
            if result.success:
                return result

        # Essential Matrix 폴백
        if E is not None:
            return self._solve_essential(
                E, p1, p2, mask_E, inliers_E, match_count,
                recent_speeds, last_t_vec, frame_idx,
            )

        return PoseResult()

    def _solve_homography(
        self, H, p1, p2, mask_H, inliers_H, match_count,
        recent_speeds, last_t_vec, frame_idx,
    ) -> PoseResult:
        """Homography 분해로 포즈를 추정합니다."""
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, self.K)
        if num_solutions == 0:
            return PoseResult()

        # Chirality 기반 해 선택
        try:
            _, filtered = cv2.filterHomographyDecompByVisibleRefpoints(rotations, normals, p1, p2)
            valid_indices = [i for i in range(num_solutions) if filtered[i] > 0]
        except Exception:
            logger.debug("filterHomographyDecompByVisibleRefpoints failed, using all solutions")
            valid_indices = list(range(num_solutions))

        if not valid_indices:
            valid_indices = list(range(num_solutions))

        # 유효 해 중 전진 방향(t[2] > 0) 우선 선택
        best_idx = valid_indices[0]
        for i in valid_indices:
            if translations[i][2][0] > 0:
                best_idx = i
                break

        R = rotations[best_idx]
        t = translations[best_idx]
        t_vec = t[:, 0]
        t_vec = recover_scale(t_vec, recent_speeds, last_t_vec)

        inlier_ratio = inliers_H / max(match_count, 1)

        logger.debug(
            "frame %d: [Homography] matches=%d, inliers=%d, ratio=%.3f, t=%s",
            frame_idx, match_count, inliers_H, inlier_ratio, t_vec.round(3),
        )

        if np.isfinite(t_vec).all() and inliers_H > 10:
            return PoseResult(
                success=True, R=R, t_vec=t_vec,
                inliers=inliers_H, inlier_ratio=inlier_ratio,
                mask=mask_H, method="Homo",
            )
        return PoseResult()

    def _solve_essential(
        self, E, p1, p2, mask_E, inliers_E, match_count,
        recent_speeds, last_t_vec, frame_idx,
    ) -> PoseResult:
        """Essential Matrix로 포즈를 추정합니다."""
        _, R, t, mask = cv2.recoverPose(E, p2, p1, self.K, mask=mask_E)
        t_vec = t[:, 0]
        t_vec = recover_scale(t_vec, recent_speeds, last_t_vec)

        inlier_ratio = inliers_E / max(match_count, 1)

        logger.debug(
            "frame %d: [Ess] matches=%d, inliers=%d, ratio=%.3f, t=%s",
            frame_idx, match_count, inliers_E, inlier_ratio, t_vec.round(3),
        )

        if np.isfinite(t_vec).all() and inliers_E > 10:
            return PoseResult(
                success=True, R=R, t_vec=t_vec,
                inliers=inliers_E, inlier_ratio=inlier_ratio,
                mask=mask, method="Ess",
            )
        return PoseResult()
