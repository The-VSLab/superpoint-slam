from __future__ import annotations

import os
import time

import cv2
import numpy as np
import torch

from matcher_module import BTMatcher
from scripts.py_superpoint import SuperPointFrontend
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
        sp_backend: str = "torch",
        trt_engine: str | None = None,
        trt_onnx: str | None = None,
        trt_build: bool = False,
        max_kpts: int = 1200,
        uniform_grid: tuple[int, int] = (8, 6),
        use_subpixel_refine: bool = True,
        use_uniform_distribution: bool = True,
        use_hybrid_matching: bool = False,
        ratio_thresh: float = 0.85,
        ransac_thresh: float = 0.8,
        com_radius: int = 2,
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
        self.sp_backend = str(sp_backend).lower()
        self.trt_engine = trt_engine
        self.trt_onnx = trt_onnx
        self.trt_build = bool(trt_build)
        self.max_kpts = int(max_kpts)
        self.uniform_grid = (int(uniform_grid[0]), int(uniform_grid[1]))
        self.use_subpixel_refine = bool(use_subpixel_refine)
        self.use_uniform_distribution = bool(use_uniform_distribution)
        self.use_hybrid_matching = bool(use_hybrid_matching)
        self.ratio_thresh = float(ratio_thresh)
        self.ransac_thresh = float(ransac_thresh)
        self.com_radius = max(int(com_radius), 1)

        if not (0.25 <= self.sp_scale <= 1.0):
            raise ValueError("sp_scale must be in [0.25, 1.0]")
        if self.max_kpts < 100:
            raise ValueError("max_kpts must be >= 100")
        if self.uniform_grid[0] < 1 or self.uniform_grid[1] < 1:
            raise ValueError("uniform_grid must be positive")
        if self.sp_backend not in ("torch", "trt"):
            raise ValueError("sp_backend must be one of: torch, trt")

        self.use_cuda = torch.cuda.is_available()

        self.focal = max(self.width, self.height) * 0.8
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.K = np.array(
            [[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        if self.sp_backend == "trt":
            from scripts.superpoint_trt import SuperPointTRTFrontend, build_engine_from_onnx

            if not self.trt_engine:
                raise ValueError("trt backend requires trt_engine path")

            if not os.path.exists(self.trt_engine):
                if self.trt_onnx and self.trt_build:
                    print("==> Building TensorRT engine from ONNX...")
                    build_engine_from_onnx(self.trt_onnx, self.trt_engine, fp16=self.sp_fp16)
                else:
                    raise FileNotFoundError(
                        f"TensorRT engine not found: {self.trt_engine}. "
                        "Provide --trt_engine or use --trt_build with --trt_onnx."
                    )

            self.frontend = SuperPointTRTFrontend(
                engine_path=self.trt_engine,
                nms_dist=self.nms_dist,
                conf_thresh=self.conf_thresh,
                nn_thresh=self.nn_thresh,
                use_fp16=self.sp_fp16,
            )
        else:
            self.frontend = SuperPointFrontend(
                weights_path=self.weights_path,
                nms_dist=self.nms_dist,
                conf_thresh=self.conf_thresh,
                nn_thresh=self.nn_thresh,
                cuda=self.use_cuda,
                head_hidden=256,
                descriptor_dim=128,
                use_fp16=self.sp_fp16,
                top_k=self.max_kpts,
            )
        self.matcher = BTMatcher(nn_thresh=self.nn_thresh, use_cuda=self.use_cuda, mutual=True)
        
        # 포인트 필터 초기화 (구름/전선/노이즈 제거)
        self.point_filter = PointFilter(frame_h=self.height, frame_w=self.width)

    def _apply_mask(self, gray: np.ndarray) -> np.ndarray:
        if not self.mask_car:
            return gray
        masked = gray.copy()
        masked[int(self.height * 0.92) :, :] = 0
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

                # *** 포인트 필터링 적용 ***
                # 구름/하늘, 전선, 통계적 이상치, 낮은 신뢰도 제거
                if len(kpts) > 0 and desc is not None:
                    kpts_filtered, desc_filtered = self.point_filter.apply_all_filters(
                        frame, kpts, desc
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

                        # 특징점을 맵에 추가 (2D)
                        centered = p2 - np.array([self.cx, self.cy], dtype=np.float64)
                        local = np.stack([centered[:, 0], centered[:, 1] * 0.0], axis=1) / max(self.width, 1)
                        c = np.cos(pose[2])
                        s = np.sin(pose[2])
                        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
                        world = (rot @ local.T).T + pose[:2]
                        
                        # inlier 점들은 신뢰도 높음으로 표시
                        if pose_mask is not None:
                            inlier_mask = pose_mask.flatten() > 0
                            inlier_pts = world[inlier_mask]
                            outlier_pts = world[~inlier_mask]
                            map_points.append((inlier_pts, True))   # (points, is_inlier)
                            map_points.append((outlier_pts, False))
                        else:
                            map_points.append((world, True))
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
                for pt in kpts[::3]:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(vis, f"Frame: {frame_idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"Matches: {len(matches)} Inliers: {inliers}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
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
        
        # 맵 포인트 결합 (모든 특징점을 표시)
        all_pts = []
        for item in map_points:
            if isinstance(item, tuple):
                pts, _ = item  # is_inlier 무시
                all_pts.append(pts)
            else:
                all_pts.append(item)
        
        map_xy = np.vstack(all_pts) if all_pts else np.empty((0, 2), dtype=np.float64)
        
        # 2D 맵 렌더링 (경로 + 특징점만 간단하게)
        topdown = render_topdown_map(traj_xy, map_xy)

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
