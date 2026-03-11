"""삼각측량 모듈 — 포인트 삼각측량 + 필터링 + 월드 변환."""
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_EMPTY_3D = np.empty((0, 3))
_EMPTY_2D = np.empty((0, 2))
_EMPTY_IDX = np.empty(0, dtype=int)


class TriangulationResult:
    """삼각측량 결과."""
    __slots__ = ('local_pts', 'world_pts', 'valid_indices', 'p1_filtered', 'p2_filtered')

    def __init__(self, local_pts=None, world_pts=None, valid_indices=None,
                 p1_filtered=None, p2_filtered=None):
        self.local_pts = local_pts if local_pts is not None else _EMPTY_3D
        self.world_pts = world_pts if world_pts is not None else _EMPTY_3D
        self.valid_indices = valid_indices if valid_indices is not None else _EMPTY_IDX
        self.p1_filtered = p1_filtered if p1_filtered is not None else _EMPTY_2D
        self.p2_filtered = p2_filtered if p2_filtered is not None else _EMPTY_2D

    @property
    def count(self) -> int:
        return len(self.local_pts)


class Triangulator:
    """삼각측량 + 필터링 + 월드 좌표 변환.

    process() 루프 내의 삼각측량 로직을 캡슐화합니다.
    """

    def __init__(self, K: np.ndarray, tri_cfg):
        self.K = K
        self.cfg = tri_cfg

    def triangulate(
        self, R: np.ndarray, t: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    ) -> np.ndarray:
        """두 뷰에서 3D 포인트를 삼각측량합니다.

        Args:
            R: 3x3 회전 행렬 (curr→prev)
            t: 3x1 이동 벡터
            p1: (N, 2) 이전 프레임 특징점
            p2: (N, 2) 현재 프레임 특징점

        Returns:
            (N, 3) 로컬 3D 포인트 (현재 카메라 프레임 기준)
        """
        P1 = self.K @ np.hstack((R, t))
        P2 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        pts_4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        pts_3d = pts_4d[:3] / (pts_4d[3] + 1e-8)
        return pts_3d.T

    def triangulate_and_filter(
        self,
        R_orig: np.ndarray,
        t_orig: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p2_idx: Optional[np.ndarray],
        mask: np.ndarray,
        cur_pose: np.ndarray,
        curr_3d_pts_world: np.ndarray,
    ) -> TriangulationResult:
        """삼각측량 + 필터링 + 월드 변환을 수행합니다.

        Args:
            R_orig: 원본 회전 행렬 (안정화 전)
            t_orig: 원본 이동 벡터
            p1: 매칭된 이전 프레임 특징점
            p2: 매칭된 현재 프레임 특징점
            p2_idx: 현재 프레임 특징점 인덱스 (curr_3d_pts_world 매핑용)
            mask: 인라이어 마스크
            cur_pose: 4x4 현재 카메라 포즈
            curr_3d_pts_world: (M, 3) 월드 좌표 배열 (업데이트됨)

        Returns:
            TriangulationResult
        """
        if p1 is None or mask is None:
            return TriangulationResult()

        mask = mask.ravel().astype(bool)
        valid_len = min(len(p1), len(mask))
        p1_m = p1[:valid_len][mask[:valid_len]]
        p2_m = p2[:valid_len][mask[:valid_len]]

        if p2_idx is not None:
            p2_idx_m = p2_idx[:valid_len][mask[:valid_len]]
        else:
            p2_idx_m = np.arange(len(p2_m))

        if len(p1_m) == 0:
            return TriangulationResult()

        local_pts = self.triangulate(R_orig, t_orig.reshape(3, 1), p1_m, p2_m)

        # 필터링 (비상식적인 점 제거)
        tri = self.cfg
        valid = (
            (local_pts[:, 2] > tri.min_depth)
            & (local_pts[:, 2] < tri.max_depth)
            & (np.abs(local_pts[:, 0]) < tri.max_x_extent)
            & (np.abs(local_pts[:, 1]) < tri.max_y_extent)
        )
        local_pts = local_pts[valid]
        p1_m = p1_m[valid]
        p2_m = p2_m[valid]

        if len(p2_idx_m) == len(valid):
            valid_indices = p2_idx_m[valid]
        else:
            valid_indices = np.arange(len(local_pts))

        # 월드 좌표 변환
        world_pts = (cur_pose[:3, :3] @ local_pts.T).T + cur_pose[:3, 3]

        # curr_3d_pts_world 범위 체크
        bounds_ok = valid_indices < len(curr_3d_pts_world)
        valid_indices = valid_indices[bounds_ok]
        world_pts = world_pts[bounds_ok]
        local_pts = local_pts[bounds_ok]
        p1_m = p1_m[bounds_ok]
        p2_m = p2_m[bounds_ok]

        # curr_3d_pts_world 인플레이스 업데이트
        curr_3d_pts_world[valid_indices] = world_pts

        return TriangulationResult(
            local_pts=local_pts,
            world_pts=world_pts,
            valid_indices=valid_indices,
            p1_filtered=p1_m,
            p2_filtered=p2_m,
        )
