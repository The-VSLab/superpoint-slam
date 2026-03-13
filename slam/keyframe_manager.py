"""키프레임 관리 모듈 — 키프레임 필요 여부 결정."""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class KeyframeManager:
    """키프레임 생성 시점을 결정합니다.

    Parallax(시차), Inlier 감소, 프레임 간격 기반으로
    키프레임 필요 여부를 판단합니다.
    """

    def __init__(self, kf_cfg):
        self.cfg = kf_cfg
        self.accumulated_parallax = 0.0

    def should_create_keyframe(
        self,
        p1: Optional[np.ndarray],
        p2: Optional[np.ndarray],
        inliers: int,
        inlier_history: list,
        frame_idx: int,
        last_keyframe_idx: int,
    ) -> bool:
        """키프레임이 필요한지 판단합니다.

        Args:
            p1: 이전 프레임 매칭 특징점 (N, 2) or None
            p2: 현재 프레임 매칭 특징점 (N, 2) or None
            inliers: 현재 프레임 인라이어 수
            inlier_history: 최근 인라이어 기록 리스트
            frame_idx: 현재 프레임 번호
            last_keyframe_idx: 마지막 키프레임 프레임 번호

        Returns:
            True이면 키프레임 생성 필요
        """
        cfg = self.cfg
        need = False

        # 1. Parallax (시차) 기반
        if p1 is not None and p2 is not None and len(p1) > 0:
            disp = np.linalg.norm(p2 - p1, axis=1)
            median_disp = np.median(disp) if len(disp) > 0 else 0
            self.accumulated_parallax += median_disp

            if self.accumulated_parallax >= cfg.parallax_threshold:
                need = True

        # 2. Inlier 감소 기반 (적응형 임계값)
        if p1 is not None and p2 is not None:
            recent = inlier_history[-30:] if len(inlier_history) > 0 else [80]
            adaptive_thresh = max(
                int(np.median(recent) * cfg.adaptive_inlier_ratio),
                cfg.min_adaptive_inliers,
            )
            if inliers < adaptive_thresh and (frame_idx - last_keyframe_idx) >= cfg.min_gap:
                need = True

        # 3. 일정 프레임 이상 경과 시 강제 추가 (Fallback)
        if (frame_idx - last_keyframe_idx) >= cfg.max_gap:
            need = True

        return need

    def reset_parallax(self):
        """키프레임 생성 후 parallax를 리셋합니다."""
        self.accumulated_parallax = 0.0
