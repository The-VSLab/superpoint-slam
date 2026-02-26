from __future__ import annotations

import random
import time

import cv2
import numpy as np
import torch

from matcher_module import BTMatcher
from frontend.superpoint_frontend import SuperPointFrontend
from tracking import PointFilter
from .slam2d_common import (
    Slam2DStats,
    compose_pose2d,
    mean_or_zero,
    path_length,
    render_topdown_map,
    save_run_artifacts,
)


class VisualSLAM2D:
    def __init__(
        self,
        weights_path: str,
        input_path: str,
        nn_thresh: float = 0.7,
        resize: tuple[int, int] = (640, 480),
        conf_thresh: float = 0.001,
        nms_dist: int = 4,
        mask_car: bool = False,
        motion_scale: float = 1.0,
        output_dir: str = "results_superpoint_2d",
        show_display: bool = True,
        sp_scale: float = 0.5,
        sp_interval: int = 1,
        sp_fp16: bool = True,
        max_kpts: int = 1500,
        min_kpts: int = 600,
        uniform_grid: tuple[int, int] = (8, 6),
        kpt_display_radius: int = 1,
        use_subpixel_refine: bool = True,
        use_uniform_distribution: bool = True,
        use_hybrid_matching: bool = False,
        ratio_thresh: float = 0.85,
        ransac_thresh: float = 0.8,
        min_parallax_px: float = 2.0,
        com_radius: int = 2,
        use_shadow_filter: bool = True,
        use_top_region_filter: bool = True,
        shadow_value_thresh: float = 0.46,
        shadow_saturation_thresh: float = 0.30,
        min_shadow_grad: float = 22.0,
        min_shadow_local_std: float = 10.0,
        shadow_rel_dark_thresh: float = 0.82,
        top_region_ratio: float = 0.38,
        top_region_min_grad: float = 34.0,
        top_region_min_std: float = 14.0,
        bottom_region_ratio: float = 0.35,
        filter_floor: bool = False,
        deterministic: bool = False,
        seed: int = 0,
    ):
        self.weights_path = str(weights_path)
        self.input_path = str(input_path)
        self.nn_thresh = float(nn_thresh)
        self.width = int(resize[0])
        self.height = int(resize[1])
        self.conf_thresh = float(conf_thresh)
        self.nms_dist = int(nms_dist)
        self.mask_car = bool(mask_car)
        self.motion_scale = float(motion_scale)
        self.output_dir = output_dir
        self.show_display = bool(show_display)
        self.sp_scale = float(sp_scale)
        self.sp_interval = max(int(sp_interval), 1)
        self.sp_fp16 = bool(sp_fp16)
        self.max_kpts = int(max_kpts)
        self.min_kpts = int(min_kpts)
        if self.max_kpts < self.min_kpts:
            self.max_kpts = self.min_kpts
        self.uniform_grid = (int(uniform_grid[0]), int(uniform_grid[1]))
        self.kpt_display_radius = max(int(kpt_display_radius), 1)
        self.use_subpixel_refine = bool(use_subpixel_refine)
        self.use_uniform_distribution = bool(use_uniform_distribution)
        self.use_hybrid_matching = bool(use_hybrid_matching)
        self.ratio_thresh = float(ratio_thresh)
        self.ransac_thresh = float(ransac_thresh)
        self.min_parallax_px = float(min_parallax_px)
        self.com_radius = max(int(com_radius), 1)
        self.use_shadow_filter = bool(use_shadow_filter)
        self.use_top_region_filter = bool(use_top_region_filter)
        self.shadow_value_thresh = float(shadow_value_thresh)
        self.shadow_saturation_thresh = float(shadow_saturation_thresh)
        self.min_shadow_grad = float(min_shadow_grad)
        self.min_shadow_local_std = float(min_shadow_local_std)
        self.shadow_rel_dark_thresh = float(shadow_rel_dark_thresh)
        self.top_region_ratio = float(top_region_ratio)
        self.top_region_min_grad = float(top_region_min_grad)
        self.top_region_min_std = float(top_region_min_std)
        self.bottom_region_ratio = float(bottom_region_ratio)
        self.filter_floor = bool(filter_floor)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        if self.deterministic:
            self._configure_determinism()

        if not (0.25 <= self.sp_scale <= 1.0):
            raise ValueError("sp_scale must be in [0.25, 1.0]")
        if self.max_kpts < 100:
            raise ValueError("max_kpts must be >= 100")
        if self.uniform_grid[0] < 1 or self.uniform_grid[1] < 1:
            raise ValueError("uniform_grid must be positive")
        self.use_cuda = torch.cuda.is_available()

        self.focal = max(self.width, self.height) * 0.8
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.K = np.array(
            [[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self.frontend = SuperPointFrontend(
            weights_path=self.weights_path,
            nms_dist=self.nms_dist,
            conf_thresh=self.conf_thresh,
            nn_thresh=self.nn_thresh,
            cuda=self.use_cuda,
        )
        self.matcher = BTMatcher(
            nn_thresh=self.nn_thresh, 
            use_cuda=self.use_cuda, 
            mutual=True,
            ratio_thresh=self.ratio_thresh
        )
        
        # 포인트 필터 초기화 (구름/전선/노이즈 제거)
        self.point_filter = PointFilter(frame_h=self.height, frame_w=self.width)

    def _configure_determinism(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        cv2.setRNGSeed(self.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    def _apply_mask(self, gray: np.ndarray) -> np.ndarray:
        """Optionally apply car/floor masking to the grayscale frame.

        - mask_car: zeros out lower ~8% of image (car hood)
        - filter_floor: zeros out the bottom region defined by
          ``bottom_region_ratio`` to prevent feature extraction there.
        """
        masked = gray
        if self.mask_car:
            masked = masked.copy()
            masked[int(self.height * 0.92) :, :] = 0
        if self.filter_floor and self.bottom_region_ratio > 0:
            if masked is gray:
                masked = masked.copy()
            bottom_y = int(self.height * (1.0 - self.bottom_region_ratio))
            masked[bottom_y:, :] = 0
        return masked

    def _refine_subpixel_com(self, kpts: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        if heatmap is None or len(kpts) == 0:
            return kpts

        h, w = heatmap.shape[:2]
        refined = kpts.astype(np.float64).copy()
        r = self.com_radius

        for idx in range(len(refined)):
            x, y = refined[idx]
            xi = int(round(x))
            yi = int(round(y))
            x0 = max(0, xi - r)
            x1 = min(w - 1, xi + r)
            y0 = max(0, yi - r)
            y1 = min(h - 1, yi + r)
            patch = heatmap[y0 : y1 + 1, x0 : x1 + 1]
            if patch.size == 0:
                continue

            patch = np.exp(patch - np.max(patch))
            total = float(np.sum(patch))
            if total < 1e-8:
                continue

            ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            refined[idx, 0] = float(np.sum(xs * patch) / total)
            refined[idx, 1] = float(np.sum(ys * patch) / total)

        return refined

    def _select_uniform_keypoints(
        self,
        kpts: np.ndarray,
        desc: np.ndarray,
        heatmap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(kpts) == 0 or desc is None:
            return kpts, desc

        gx, gy = self.uniform_grid
        per_cell = max(1, self.max_kpts // (gx * gy))
        cell_w = self.width / gx
        cell_h = self.height / gy

        if heatmap is None:
            scores = np.ones((len(kpts),), dtype=np.float64)
        else:
            xi = np.clip(np.round(kpts[:, 0]).astype(np.int32), 0, heatmap.shape[1] - 1)
            yi = np.clip(np.round(kpts[:, 1]).astype(np.int32), 0, heatmap.shape[0] - 1)
            scores = heatmap[yi, xi].astype(np.float64)

        selected = []
        for gy_i in range(gy):
            y_min = gy_i * cell_h
            y_max = (gy_i + 1) * cell_h
            for gx_i in range(gx):
                x_min = gx_i * cell_w
                x_max = (gx_i + 1) * cell_w
                cell_mask = (
                    (kpts[:, 0] >= x_min)
                    & (kpts[:, 0] < x_max)
                    & (kpts[:, 1] >= y_min)
                    & (kpts[:, 1] < y_max)
                )
                idxs = np.where(cell_mask)[0]
                if len(idxs) == 0:
                    continue
                order = idxs[np.argsort(-scores[idxs])]
                selected.extend(order[:per_cell].tolist())

        if len(selected) < self.max_kpts:
            global_order = np.argsort(-scores)
            selected_set = set(selected)
            for idx in global_order:
                if idx not in selected_set:
                    selected.append(int(idx))
                    selected_set.add(int(idx))
                if len(selected) >= self.max_kpts:
                    break

        sel = np.array(selected[: self.max_kpts], dtype=np.int32)
        return kpts[sel], desc[:, sel]

    def _match_with_ratio_test(self, prev_desc: np.ndarray, desc: np.ndarray) -> np.ndarray:
        if prev_desc is None or desc is None or prev_desc.shape[1] == 0 or desc.shape[1] == 0:
            return np.empty((0, 2), dtype=np.int32)

        d1 = prev_desc.T.astype(np.float32)
        d2 = desc.T.astype(np.float32)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = matcher.knnMatch(d1, d2, k=2)

        filtered = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance and m.distance < self.nn_thresh:
                filtered.append((m.queryIdx, m.trainIdx))

        if not filtered:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(filtered, dtype=np.int32)

    def _supplement_keypoints_to_min(
        self,
        kpts: np.ndarray,
        desc: np.ndarray,
        candidate_kpts: np.ndarray,
        candidate_desc: np.ndarray,
        heatmap: np.ndarray,
        target_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if target_count <= 0:
            return kpts, desc
        if candidate_kpts is None or candidate_desc is None or len(candidate_kpts) == 0:
            return kpts, desc
        if desc is None or candidate_desc is None:
            return kpts, desc

        if len(kpts) >= target_count:
            return kpts, desc

        if heatmap is None:
            cand_scores = np.ones((len(candidate_kpts),), dtype=np.float64)
        else:
            xi = np.clip(np.round(candidate_kpts[:, 0]).astype(np.int32), 0, heatmap.shape[1] - 1)
            yi = np.clip(np.round(candidate_kpts[:, 1]).astype(np.int32), 0, heatmap.shape[0] - 1)
            cand_scores = heatmap[yi, xi].astype(np.float64)

        existing = set()
        for pt in kpts:
            existing.add((int(round(pt[0])), int(round(pt[1]))))

        needed = int(target_count - len(kpts))
        if needed <= 0:
            return kpts, desc

        selected_idx = []
        for idx in np.argsort(-cand_scores):
            p = candidate_kpts[int(idx)]
            key = (int(round(p[0])), int(round(p[1])))
            if key in existing:
                continue
            selected_idx.append(int(idx))
            existing.add(key)
            if len(selected_idx) >= needed:
                break

        if not selected_idx:
            return kpts, desc

        add_kpts = candidate_kpts[selected_idx]
        add_desc = candidate_desc[:, selected_idx]
        out_kpts = np.vstack([kpts, add_kpts])
        out_desc = np.concatenate([desc, add_desc], axis=1)

        if len(out_kpts) > target_count:
            if heatmap is None:
                out_scores = np.ones((len(out_kpts),), dtype=np.float64)
            else:
                xi = np.clip(np.round(out_kpts[:, 0]).astype(np.int32), 0, heatmap.shape[1] - 1)
                yi = np.clip(np.round(out_kpts[:, 1]).astype(np.int32), 0, heatmap.shape[0] - 1)
                out_scores = heatmap[yi, xi].astype(np.float64)
            keep = np.argsort(-out_scores)[:target_count]
            out_kpts = out_kpts[keep]
            out_desc = out_desc[:, keep]

        return out_kpts, out_desc

    def _geometric_filter_matches(
        self,
        prev_kpts: np.ndarray,
        curr_kpts: np.ndarray,
        matches: np.ndarray,
        threshold: float = 1.5,
    ) -> np.ndarray:
        if matches is None or len(matches) < 8:
            return matches

        p1 = prev_kpts[matches[:, 0]].astype(np.float64)
        p2 = curr_kpts[matches[:, 1]].astype(np.float64)
        _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, threshold, 0.99)
        if mask is None:
            return matches
        mask = mask.ravel().astype(bool)
        filtered = matches[mask]
        return filtered if len(filtered) >= 8 else matches

    def _select_match_set(
        self,
        prev_kpts: np.ndarray,
        curr_kpts: np.ndarray,
        mutual_matches: np.ndarray,
        ratio_matches: np.ndarray,
    ) -> np.ndarray:
        candidates = []
        if mutual_matches is not None and len(mutual_matches) > 0:
            candidates.append(mutual_matches)
        if ratio_matches is not None and len(ratio_matches) > 0:
            candidates.append(ratio_matches)
        if mutual_matches is not None and ratio_matches is not None and len(mutual_matches) > 0 and len(ratio_matches) > 0:
            inter = np.array(
                list(set((int(i), int(j)) for i, j in mutual_matches) & set((int(i), int(j)) for i, j in ratio_matches)),
                dtype=np.int32,
            )
            if len(inter) > 0:
                candidates.append(inter)

        if not candidates:
            return np.empty((0, 2), dtype=np.int32)

        best = candidates[0]
        best_score = -1.0
        for cand in candidates:
            if len(cand) < 8:
                continue
            filtered = self._geometric_filter_matches(prev_kpts, curr_kpts, cand)
            score = len(filtered) / max(len(cand), 1)
            if score > best_score:
                best = filtered
                best_score = score

        return best

    def process(self) -> Slam2DStats:
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {self.input_path}")

        prev_kpts = None
        prev_desc = None

        pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        trajectory = [pose[:2].copy()]
        map_points = []

        extract_ms = []
        match_ms = []
        total_ms = []
        kpts_count = []
        matches_count = []
        inliers_count = []
        inlier_ratio_list = []

        frame_idx = 0
        if self.show_display:
            cv2.namedWindow("SuperPoint 2D SLAM", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("SuperPoint 2D SLAM", 960, 540)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            frame = cv2.resize(frame, (self.width, self.height))
            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 2:
                gray = frame
            else:
                print(f"ERROR: Unexpected frame shape: {frame.shape}")
                break
            masked = self._apply_mask(gray)

            run_infer = (frame_idx == 0) or (frame_idx % self.sp_interval == 0)

            fe_t0 = time.perf_counter()
            if run_infer:
                if self.sp_scale != 1.0:
                    sw = max(int(self.width * self.sp_scale), 64)
                    sh = max(int(self.height * self.sp_scale), 64)
                    sp_input = cv2.resize(masked, (sw, sh), interpolation=cv2.INTER_AREA)
                else:
                    sp_input = masked

                pts, desc, heatmap = self.frontend.run(sp_input.astype(np.float32) / 255.0)
                kpts = pts[:2, :].T if pts.shape[1] > 0 else np.empty((0, 2), dtype=np.float64)

                if self.sp_scale != 1.0 and len(kpts) > 0:
                    kpts = kpts / self.sp_scale
                if self.sp_scale != 1.0 and heatmap is not None:
                    heatmap = cv2.resize(
                        heatmap.astype(np.float32),
                        (self.width, self.height),
                        interpolation=cv2.INTER_CUBIC,
                    )

                # remove floor points post-detection as a safety net
                if self.filter_floor and len(kpts) > 0:
                    bottom_y = self.height * (1.0 - self.bottom_region_ratio)
                    valid = kpts[:, 1] < bottom_y
                    if not np.all(valid):
                        kpts = kpts[valid]
                        if desc is not None:
                            # descriptor shape may be (dim,N) or (N,dim)
                            if desc.ndim == 2 and desc.shape[1] == len(valid):
                                desc = desc[:, valid]
                            elif desc.ndim == 2 and desc.shape[0] == len(valid):
                                desc = desc[valid, :]

                raw_kpts = kpts.copy()
                raw_desc = desc.copy() if desc is not None else None

                # *** 포인트 필터링 적용 ***
                # 구름/하늘, 전선, 통계적 이상치, 낮은 신뢰도 제거
                if len(kpts) > 0 and desc is not None:
                    kpts_filtered, desc_filtered = self.point_filter.apply_all_filters(
                        frame,
                        kpts,
                        desc,
                        use_shadow_filter=self.use_shadow_filter,
                        use_top_region_filter=self.use_top_region_filter,
                        shadow_value_thresh=self.shadow_value_thresh,
                        shadow_saturation_thresh=self.shadow_saturation_thresh,
                        min_shadow_grad=self.min_shadow_grad,
                        min_shadow_local_std=self.min_shadow_local_std,
                        shadow_rel_dark_thresh=self.shadow_rel_dark_thresh,
                        top_region_ratio=self.top_region_ratio,
                        top_region_min_grad=self.top_region_min_grad,
                        top_region_min_std=self.top_region_min_std,
                    )
                    kpts = kpts_filtered
                    desc = desc_filtered
                else:
                    # 필터링할 포인트 없음
                    pass

                if self.use_subpixel_refine:
                    kpts = self._refine_subpixel_com(kpts, heatmap)
                if self.use_uniform_distribution:
                    kpts, desc = self._select_uniform_keypoints(kpts, desc, heatmap)
                # Ensure minimum keypoints
                if len(kpts) < self.min_kpts:
                    target = min(self.min_kpts, self.max_kpts)
                    before = len(kpts)

                    # 보충 후보도 하늘/구름/상단 약구조 포인트는 제외해 재유입 방지
                    cand_mask = self.point_filter.filter_sky_points(frame, raw_kpts)
                    if self.use_shadow_filter:
                        cand_mask &= self.point_filter.filter_shadow_points(
                            frame,
                            raw_kpts,
                            shadow_value_thresh=self.shadow_value_thresh,
                            shadow_saturation_thresh=self.shadow_saturation_thresh,
                            min_shadow_grad=self.min_shadow_grad,
                            min_shadow_local_std=self.min_shadow_local_std,
                            shadow_rel_dark_thresh=self.shadow_rel_dark_thresh,
                        )
                    if self.use_top_region_filter:
                        cand_mask &= self.point_filter.filter_top_region_points(
                            frame,
                            raw_kpts,
                            top_region_ratio=self.top_region_ratio,
                            top_region_min_grad=self.top_region_min_grad,
                            top_region_min_std=self.top_region_min_std,
                        )

                    cand_kpts = raw_kpts[cand_mask]
                    cand_desc = raw_desc[:, cand_mask] if raw_desc is not None else None

                    kpts, desc = self._supplement_keypoints_to_min(
                        kpts,
                        desc,
                        cand_kpts,
                        cand_desc,
                        heatmap,
                        target,
                    )
                    if len(kpts) < self.min_kpts:
                        shortage = self.min_kpts - len(kpts)
                        print(f"⚠️ Keypoints {before} -> {len(kpts)} (target {self.min_kpts}), still short by {shortage}")
                    else:
                        print(f"✅ Keypoints supplemented: {before} -> {len(kpts)} (target {self.min_kpts})")
            else:
                kpts = prev_kpts if prev_kpts is not None else np.empty((0, 2), dtype=np.float64)
                desc = prev_desc
            fe_t1 = time.perf_counter()

            matches = np.empty((0, 2), dtype=np.int32)
            inliers = 0
            inlier_ratio = 0.0

            m_t0 = time.perf_counter()
            if prev_desc is not None and desc is not None and len(kpts) > 0 and len(prev_kpts) > 0:
                if self.use_hybrid_matching:
                    mutual_matches = self.matcher.match(prev_desc, desc)
                    ratio_matches = self._match_with_ratio_test(prev_desc, desc)
                    matches = self._select_match_set(prev_kpts, kpts, mutual_matches, ratio_matches)
                else:
                    matches = self.matcher.match(prev_desc, desc)
            m_t1 = time.perf_counter()

            if len(matches) >= 8:
                p1 = prev_kpts[matches[:, 0]].astype(np.float64)
                p2 = kpts[matches[:, 1]].astype(np.float64)

                E, emask = cv2.findEssentialMat(
                    p2,
                    p1,
                    self.K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=self.ransac_thresh,
                )
                if E is not None:
                    _, R, t, pose_mask = cv2.recoverPose(E, p2, p1, self.K)
                    inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0
                    inlier_ratio = inliers / max(len(matches), 1)

                    t = t[:, 0]
                    if np.isfinite(t).all():
                        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
                        delta_local = np.array([t[0], t[2]], dtype=np.float64)
                        norm = np.linalg.norm(delta_local)
                        if norm > 1e-6:
                            delta_local = (delta_local / norm) * self.motion_scale
                        pose = compose_pose2d(pose, delta_local, yaw)
                        trajectory.append(pose[:2].copy())

                        # === 특징점을 맵에 추가 (깊이 추정 기반) ===
                        # Y 좌표 기반 깊이 추정: 하단(가까움) ~ 상단(멀리)
                        # 예: 화면 하단 20% → 2m, 중앙 → 10m, 상단 30% → 30m
                        y_normalized = (self.cy - p2[:, 1]) / (self.cy + 1e-6)  # -1 (하단) ~ +1 (상단)
                        y_normalized = np.clip(y_normalized, -1.0, 1.0)
                        
                        # 깊이 매핑: 하단(2m) ~ 상단(30m)  
                        depth_min = 2.0
                        depth_max = 30.0
                        depth = depth_min + (depth_max - depth_min) * (y_normalized + 1.0) / 2.0
                        
                        # X 방향 오프셋 (수평 위치 기반)
                        x_offset = (p2[:, 0] - self.cx) / (self.focal + 1e-6)
                        
                        # 로컬 좌표계 (X: 좌우, Y: 전방 깊이)
                        local_x = x_offset * depth
                        local_y = depth
                        local = np.stack([local_x, local_y], axis=1)
                        
                        # 월드 좌표 변환 (회전 + 이동)
                        c = np.cos(pose[2])
                        s = np.sin(pose[2])
                        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
                        world = (rot @ local.T).T + pose[:2]
                        
                        # inlier + 최소 시차 필터링 + 바닥/사물 구분
                        bottom_y = int(self.height * (1.0 - self.bottom_region_ratio))
                        if pose_mask is not None:
                            inlier_mask = pose_mask.flatten() > 0
                            parallax = np.linalg.norm((p2 - p1), axis=1)
                            valid = inlier_mask & (parallax >= self.min_parallax_px)
                            inlier_pts = world[valid]
                            is_floor_array = p2[valid, 1] >= bottom_y  # p2의 y >= bottom_y면 바닥
                            if len(inlier_pts) > 0:
                                map_points.append((inlier_pts, True, is_floor_array))
                        else:
                            parallax = np.linalg.norm((p2 - p1), axis=1)
                            valid = parallax >= self.min_parallax_px
                            inlier_pts = world[valid]
                            is_floor_array = p2[valid, 1] >= bottom_y  # p2의 y >= bottom_y면 바닥
                            if len(inlier_pts) > 0:
                                map_points.append((inlier_pts, True, is_floor_array))
                else:
                    trajectory.append(pose[:2].copy())
            else:
                trajectory.append(pose[:2].copy())

            t1 = time.perf_counter()

            extract_ms.append((fe_t1 - fe_t0) * 1000.0)
            match_ms.append((m_t1 - m_t0) * 1000.0)
            total_ms.append((t1 - t0) * 1000.0)
            kpts_count.append(int(len(kpts)))
            matches_count.append(int(len(matches)))
            inliers_count.append(inliers)
            inlier_ratio_list.append(inlier_ratio)
            
            # 진행 상황 출력 (터미널에서 확인 가능)
            if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                elapsed_ms = (t1 - t0) * 1000.0
                print(f"[Frame {frame_idx+1:4d}] "
                      f"Keypts: {len(kpts):4d} | "
                      f"Matches: {len(matches):3d} | "
                      f"Inliers: {inliers:3d} ({inlier_ratio*100:5.1f}%) | "
                      f"Time: {elapsed_ms:6.1f}ms")

            if self.show_display:
                vis = frame.copy()
                bottom_y = int(self.height * (1.0 - self.bottom_region_ratio))
                # only draw non-floor keypoints (skip points below threshold)
                # display every 5th keypoint to reduce visual clutter
                for pt in kpts[::5]:
                    x, y = int(pt[0]), int(pt[1])
                    if y < bottom_y:
                        cv2.circle(vis, (x, y), self.kpt_display_radius, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                # Draw bottom region threshold line for reference
                cv2.line(vis, (0, bottom_y), (self.width, bottom_y), (100, 100, 100), 1)
                cv2.putText(vis, f"Frame: {frame_idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"Matches: {len(matches)} Inliers: {inliers}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
                # no floor/keypoint legend since floor points are hidden
                cv2.imshow("SuperPoint 2D SLAM", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if run_infer:
                prev_kpts = kpts
                prev_desc = desc
            frame_idx += 1

        cap.release()
        if self.show_display:
            cv2.destroyAllWindows()

        traj_xy = np.asarray(trajectory, dtype=np.float64)
        
        # 맵 포인트 결합 (바닥/사물 구분 정보 포함)
        all_pts = []
        all_is_floor = []
        for item in map_points:
            if isinstance(item, tuple) and len(item) == 3:
                pts, _, is_floor = item  # (pts, is_inlier, is_floor)
                all_pts.append(pts)
                all_is_floor.append(is_floor)
            elif isinstance(item, tuple) and len(item) == 2:
                pts, _ = item
                all_pts.append(pts)
                all_is_floor.append(np.zeros(len(pts), dtype=bool))  # 기본값: 사물
            else:
                all_pts.append(item)
                all_is_floor.append(np.zeros(len(item), dtype=bool))
        
        map_xy = np.vstack(all_pts) if all_pts else np.empty((0, 2), dtype=np.float64)
        is_floor_array = np.concatenate(all_is_floor) if all_is_floor else np.empty(0, dtype=bool)
        
        # 2D 맵 렌더링 (경로 + 특징점: 파란색/빨간색 구분)
        topdown = render_topdown_map(traj_xy, map_xy, is_floor_array=is_floor_array)

        stats = Slam2DStats(
            name="superpoint_2d",
            frames=frame_idx,
            avg_extract_ms=mean_or_zero(extract_ms),
            avg_match_ms=mean_or_zero(match_ms),
            avg_total_ms=mean_or_zero(total_ms),
            avg_kpts=mean_or_zero(kpts_count),
            avg_matches=mean_or_zero(matches_count),
            avg_inliers=mean_or_zero(inliers_count),
            avg_inlier_ratio=mean_or_zero(inlier_ratio_list),
            trajectory_length=path_length(traj_xy),
            output_dir=self.output_dir,
        )
        save_run_artifacts(self.output_dir, stats, traj_xy, topdown)
        return stats
