"""특징점 추출 모듈 — SuperPoint + Semantic/Point 필터 파이프라인."""
import logging
import time
from typing import Optional

import cv2
import numpy as np

from slam.slam_utils import desc_to_numpy, desc_filter_by_mask

logger = logging.getLogger(__name__)

_EMPTY_KPTS = np.empty((0, 2))


class ExtractionResult:
    """특징점 추출 결과."""
    __slots__ = ('kpts', 'desc', 'sp_ms', 'filter_ms')

    def __init__(self, kpts=None, desc=None, sp_ms=0.0, filter_ms=0.0):
        self.kpts = kpts if kpts is not None else _EMPTY_KPTS
        self.desc = desc
        self.sp_ms = sp_ms
        self.filter_ms = filter_ms


class FeatureExtractor:
    """SuperPoint 추론 + Semantic/Point 필터 파이프라인.

    CLAHE 전처리, SuperPoint 추론, Semantic Filter, Point Filter를
    하나의 extract() 호출로 통합합니다.
    """

    def __init__(
        self,
        fe,  # SuperPointFrontend
        point_filter,  # PointFilter
        semantic_filter,  # SemanticFilter or None
        *,
        sp_cfg,
        point_filter_cfg,
        use_cuda: bool = False,
        use_semantic: bool = False,
        filter_on_sp_only: bool = False,
        W: int = 640,
        H: int = 360,
    ):
        self.fe = fe
        self.point_filter = point_filter
        self.semantic_filter = semantic_filter
        self.sp_scale = float(sp_cfg.sp_scale)
        self.use_cuda = use_cuda
        self.use_semantic = use_semantic
        self.filter_on_sp_only = filter_on_sp_only
        self.W = W
        self.H = H

        # Point filter config
        self.pf_cfg = point_filter_cfg

        # 캐시된 Semantic mask (OF 프레임에서 재사용)
        self._cached_dyn_mask: Optional[np.ndarray] = None

    def extract(
        self,
        img_masked: np.ndarray,
        img_color: np.ndarray,
        frame_color_resized: np.ndarray,
        run_infer: bool,
    ) -> ExtractionResult:
        """특징점을 추출합니다.

        Args:
            img_masked: 대시보드 마스킹된 그레이스케일 이미지
            img_color: 원본 컬러 이미지 (point filter HSV 변환용)
            frame_color_resized: 리사이즈된 컬러 프레임 (semantic filter용)
            run_infer: True면 SuperPoint 추론 실행, False면 건너뜀

        Returns:
            ExtractionResult
        """
        if not run_infer:
            return ExtractionResult()

        # SuperPoint 입력 준비
        if self.sp_scale != 1.0:
            sp_w = max(int(self.W * self.sp_scale), 8)
            sp_h = max(int(self.H * self.sp_scale), 8)
            img_sp = cv2.resize(img_masked, (sp_w, sp_h))
        else:
            img_sp = img_masked

        # Semantic Filter (YOLO 마스크 생성)
        dyn_mask = None
        if self.use_semantic and self.semantic_filter is not None and self.semantic_filter.enabled:
            dyn_mask = self.semantic_filter.get_dynamic_mask(frame_color_resized)
            self._cached_dyn_mask = dyn_mask

        img_fe = img_sp.astype(np.float32) / 255.0

        # SuperPoint 추론
        sp_t0 = time.perf_counter()
        pts, desc, _ = self.fe.run(img_fe)
        sp_t1 = time.perf_counter()

        kpts = pts[:2, :].T if pts.shape[1] > 0 else np.empty((0, 2))
        if self.sp_scale != 1.0 and len(kpts) > 0:
            kpts = kpts / self.sp_scale

        # CUDA: desc를 GPU 텐서로 유지 / MPS·CPU: 즉시 numpy 변환
        if desc is not None and not self.use_cuda:
            desc = desc_to_numpy(desc)

        # Semantic Filter 적용 (동적 객체 제거)
        if dyn_mask is not None and len(kpts) > 0:
            kpts, desc = self._apply_dynamic_mask(kpts, desc, dyn_mask)

        # Point Filter (하늘 + 그림자 + 상단 영역)
        filter_t0 = time.perf_counter()
        should_filter = (not self.filter_on_sp_only) or run_infer
        if len(kpts) > 0 and should_filter:
            pf = self.pf_cfg
            combined_mask = self.point_filter.filter_shadow_and_top_combined(
                img_color,
                kpts,
                use_sky_filter=True,
                use_shadow_filter=pf.use_shadow_filter,
                use_top_region_filter=pf.use_top_region_filter,
                shadow_value_thresh=pf.shadow_value_thresh,
                shadow_saturation_thresh=pf.shadow_saturation_thresh,
                min_shadow_grad=pf.min_shadow_grad,
                min_shadow_local_std=pf.min_shadow_local_std,
                shadow_rel_dark_thresh=pf.shadow_rel_dark_thresh,
                top_region_ratio=pf.top_region_ratio,
                top_region_min_grad=pf.top_region_min_grad,
                top_region_min_std=pf.top_region_min_std,
            )
            kpts = kpts[combined_mask]
            desc = desc_filter_by_mask(desc, combined_mask)
        filter_t1 = time.perf_counter()

        return ExtractionResult(
            kpts=kpts,
            desc=desc,
            sp_ms=(sp_t1 - sp_t0) * 1000.0,
            filter_ms=(filter_t1 - filter_t0) * 1000.0,
        )

    def filter_flow_by_dynamic_mask(self, kpts: np.ndarray) -> np.ndarray:
        """OF 프레임에서 캐시된 dyn_mask로 동적 객체를 제거합니다.

        Args:
            kpts: (N, 2) 키포인트

        Returns:
            유효한 키포인트의 boolean 마스크
        """
        if self._cached_dyn_mask is None or len(kpts) == 0:
            return np.ones(len(kpts), dtype=bool)

        kpts_int = kpts.astype(int)
        kx = np.clip(kpts_int[:, 0], 0, self.W - 1)
        ky = np.clip(kpts_int[:, 1], 0, self.H - 1)
        return ~self._cached_dyn_mask[ky, kx]

    def _apply_dynamic_mask(self, kpts, desc, dyn_mask):
        """동적 마스크를 적용하여 동적 객체 위의 키포인트를 제거합니다."""
        kpts_int = kpts.astype(int)
        kx = np.clip(kpts_int[:, 0], 0, self.W - 1)
        ky = np.clip(kpts_int[:, 1], 0, self.H - 1)
        valid_mask = ~dyn_mask[ky, kx]
        return kpts[valid_mask], desc_filter_by_mask(desc, valid_mask)
