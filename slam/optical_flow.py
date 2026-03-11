"""Optical Flow 트래킹 모듈 — LK Forward-Backward 일관성 검증."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FlowResult:
    """Optical Flow 결과."""
    __slots__ = ('flow_p1', 'flow_p2', 'status')

    def __init__(self, flow_p1=None, flow_p2=None, status=None):
        self.flow_p1 = flow_p1
        self.flow_p2 = flow_p2
        self.status = status


class OpticalFlowTracker:
    """LK Forward-Backward Optical Flow 트래커."""

    def __init__(self, flow_cfg):
        self.cfg = flow_cfg
        self.lk_params = dict(
            winSize=(flow_cfg.win_size, flow_cfg.win_size),
            maxLevel=flow_cfg.max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                flow_cfg.termination_max_iter,
                flow_cfg.termination_epsilon,
            ),
        )

    def track(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_kpts: np.ndarray,
    ) -> FlowResult:
        """이전 프레임에서 현재 프레임으로 특징점을 추적합니다.

        Args:
            prev_frame: 이전 그레이스케일 프레임
            curr_frame: 현재 그레이스케일 프레임
            prev_kpts: 이전 프레임 특징점 (N, 2)

        Returns:
            FlowResult
        """
        if prev_kpts is None or len(prev_kpts) == 0:
            return FlowResult()

        prev_pts = prev_kpts.astype(np.float32).reshape(-1, 1, 2)

        # Forward: prev → curr
        curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_pts, None, **self.lk_params
        )

        if status_fwd is None or curr_pts is None:
            return FlowResult()

        # Backward: curr → prev (Forward-Backward 일관성 검증)
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_frame, prev_frame, curr_pts, None, **self.lk_params
        )

        status_fwd = status_fwd.reshape(-1).astype(bool)
        status_bwd = status_bwd.reshape(-1).astype(bool)

        if back_pts is not None:
            fb_error = np.linalg.norm(
                prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            status = status_fwd & status_bwd & (fb_error < self.cfg.fb_thresh)
        else:
            status = status_fwd & status_bwd

        if not status.any():
            return FlowResult()

        flow_p1 = prev_kpts[status]
        flow_p2 = curr_pts.reshape(-1, 2)[status]

        return FlowResult(flow_p1=flow_p1, flow_p2=flow_p2, status=status)
