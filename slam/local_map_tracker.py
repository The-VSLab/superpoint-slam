"""로컬 맵 트래킹 모듈 — 맵포인트 재투영 + 디스크립터 매칭."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LocalMapResult:
    """로컬 맵 트래킹 결과."""
    __slots__ = ('extra_obj_pts', 'extra_img_pts', 'extra_mp_ids', 'extra_kpt_idxs')

    def __init__(self):
        self.extra_obj_pts = []
        self.extra_img_pts = []
        self.extra_mp_ids = []
        self.extra_kpt_idxs = []

    @property
    def count(self) -> int:
        return len(self.extra_obj_pts)

    def as_arrays(self):
        """리스트를 numpy 배열로 변환하여 (obj_pts, img_pts)를 반환합니다."""
        if self.count == 0:
            return None, None
        return (
            np.array(self.extra_obj_pts, dtype=np.float32),
            np.array(self.extra_img_pts, dtype=np.float32),
        )


class LocalMapTracker:
    """로컬 맵포인트를 현재 뷰로 투영하여 추가 2D-3D 대응을 확보합니다.

    PnP RANSAC의 강건성을 높이기 위해, 프레임 간 매칭 외에도
    최근 키프레임들의 맵포인트를 현재 프레임에 재투영하여 매칭합니다.
    """

    def __init__(self, K: np.ndarray, matcher, perf_cfg):
        self.K = K
        self.matcher = matcher
        self.local_map_limit = perf_cfg.local_map_limit
        self.local_map_keyframes = perf_cfg.local_map_keyframes
        self.projection_dist = perf_cfg.local_map_projection_dist

    def track(
        self,
        slam_map,
        cur_pose: np.ndarray,
        kpts: np.ndarray,
        desc,
        curr_kpt_to_mp: dict,
        frame_shape: tuple,
        frame_idx: int,
    ) -> LocalMapResult:
        """로컬 맵포인트를 현재 프레임에 매칭합니다.

        Args:
            slam_map: Map 객체
            cur_pose: 4x4 현재 카메라 포즈
            kpts: (N, 2) 현재 프레임 특징점
            desc: 현재 프레임 디스크립터
            curr_kpt_to_mp: 이미 매칭된 kpt→MapPoint 딕셔너리
            frame_shape: (H, W) 프레임 크기
            frame_idx: 현재 프레임 번호 (로깅용)

        Returns:
            LocalMapResult
        """
        result = LocalMapResult()

        if desc is None:
            return result

        local_map_pts_dict = slam_map.get_recent_map_points(
            num_recent_keyframes=self.local_map_keyframes
        )

        if len(local_map_pts_dict) == 0:
            return result

        # MapPoint 개수 제한
        if self.local_map_limit > 0 and len(local_map_pts_dict) > self.local_map_limit:
            sorted_items = sorted(local_map_pts_dict.items(), key=lambda x: x[0], reverse=True)
            local_map_pts_dict = dict(sorted_items[:self.local_map_limit])

        # 1. 맵포인트들을 현재 카메라 뷰로 투영
        mp_positions = np.array(
            [mp.pos3d for mp in local_map_pts_dict.values()], dtype=np.float32
        )
        proj_pts, _ = cv2.projectPoints(
            mp_positions,
            cv2.Rodrigues(cur_pose[:3, :3].T)[0],
            -cur_pose[:3, :3].T @ cur_pose[:3, 3],
            self.K, None,
        )
        proj_pts = proj_pts.reshape(-1, 2)

        # 2. 이미지 범위 필터링
        h, w = frame_shape
        in_image = (
            (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < w)
            & (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < h)
        )

        visible_mps = [mp for i, mp in enumerate(local_map_pts_dict.values()) if in_image[i]]
        visible_proj_pts = proj_pts[in_image]

        # 3. 디스크립터 매칭 (descriptor가 있는 맵포인트만 필터링)
        valid_indices = [i for i, mp in enumerate(visible_mps) if mp.descriptor is not None]
        if len(valid_indices) == 0:
            return result

        valid_mps = [visible_mps[i] for i in valid_indices]
        valid_proj_pts = visible_proj_pts[valid_indices]

        mp_descs = np.array([mp.descriptor for mp in valid_mps]).T
        if mp_descs.shape[1] == 0:
            return result

        knn_matches = self.matcher.match(mp_descs, desc)

        for m in knn_matches:
            mp_idx = m[0]
            kpt_idx = m[1]

            if kpt_idx not in curr_kpt_to_mp:
                mp = valid_mps[mp_idx]
                pt_proj = valid_proj_pts[mp_idx]
                dist = np.linalg.norm(pt_proj - kpts[kpt_idx][:2])

                if dist < self.projection_dist:
                    result.extra_obj_pts.append(mp.pos3d)
                    result.extra_img_pts.append(kpts[kpt_idx][:2])
                    result.extra_mp_ids.append(mp)
                    result.extra_kpt_idxs.append(kpt_idx)

        if result.count > 0:
            logger.debug(
                "frame %d: [Local Map Tracking] Added %d extra correspondences.",
                frame_idx, result.count,
            )

        return result
