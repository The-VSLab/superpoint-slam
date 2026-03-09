import argparse
import csv
import logging
import numpy as np
import cv2
import torch
import open3d as o3d
import  g2o
import os
import time
import sys

logger = logging.getLogger(__name__)

# 기존 모듈
from frontend.superpoint_frontend import SuperPointFrontend
from matcher_module import BTMatcher
from slam.loop_closure import LoopClosureManager
from tracking.point_filter import PointFilter
from tracking.semantic_filter import SemanticFilter
from slam.map_elements import Map, MapPoint, KeyFrame
from slam.bundle_adjustment import run_bundle_adjustment, BAResult
from config.slam_config import SLAMConfig

# --- 환경별 장치 자동 설정 함수 추가 ---
def get_optimal_device():
    """
    NVIDIA GPU(CUDA), Apple Silicon(MPS), CPU 중 최적의 장치를 반환
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def create_camera_frustum(scale=1.0, color=[0, 0, 1]):
    """카메라 위치를 나타내는 피라미드(Frustum) 생성"""
    points = [
        [0, 0, 0],  # 0: Camera Center (Tip)
        [-scale, -scale, scale*2], # 1: Top-Left
        [scale, -scale, scale*2],  # 2: Top-Right
        [scale, scale, scale*2],   # 3: Bottom-Right
        [-scale, scale, scale*2]   # 4: Bottom-Left
    ]
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4], # Tip to corners
        [1, 2], [2, 3], [3, 4], [4, 1]  # Base rectangle
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set

def get_height_color(y_vals, y_min=-5.0, y_max=2.0):
    """높이 기반 컬러링 (Jet Style)"""
    y_vals = np.atleast_1d(y_vals)
    norm = np.clip((y_vals - y_min) / (y_max - y_min), 0.0, 1.0)
    colors = np.zeros((len(y_vals), 3))
    # Jet Colormap Approximation
    colors[:, 0] = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0) # R
    colors[:, 1] = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0) # G
    colors[:, 2] = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0) # B
    return colors

class VisualSLAM3D:
    def __init__(
        self,
        weights_path,
        input_path,
        config: SLAMConfig = None,
        output_dir="path_final",
        resize=None,
        highway_mode=False,
    ):
        # --- 설정 객체 ---
        self.cfg = config or SLAMConfig()
        c = self.cfg  # 축약 참조

        # 1. 장치 결정
        self.highway_mode = highway_mode
        self.device = get_optimal_device()
        self.use_cuda = (self.device == "cuda")

        # 설정 값 → 인스턴스 변수 (기존 코드 호환)
        self.use_shadow_filter = c.point_filter.use_shadow_filter
        self.use_top_region_filter = c.point_filter.use_top_region_filter
        self.shadow_value_thresh = c.point_filter.shadow_value_thresh
        self.shadow_saturation_thresh = c.point_filter.shadow_saturation_thresh
        self.min_shadow_grad = c.point_filter.min_shadow_grad
        self.min_shadow_local_std = c.point_filter.min_shadow_local_std
        self.shadow_rel_dark_thresh = c.point_filter.shadow_rel_dark_thresh
        self.top_region_ratio = c.point_filter.top_region_ratio
        self.top_region_min_grad = c.point_filter.top_region_min_grad
        self.top_region_min_std = c.point_filter.top_region_min_std

        self.enable_viz = c.viz.enabled
        self.flow_win_size = c.optical_flow.win_size
        self.flow_max_level = c.optical_flow.max_level
        self.flow_fb_thresh = c.optical_flow.fb_thresh
        self.use_clahe = c.clahe.enabled
        self.local_map_limit = c.performance.local_map_limit
        self.filter_on_sp_only = c.performance.filter_on_sp_only

        logger.info("Running on device: %s", self.device.upper())
        if not self.enable_viz:
            logger.info("Visualization DISABLED (performance mode)")

        self.input_path = input_path

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise ValueError(f"Error: {input_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if resize is not None:
            self.W = (resize[0] // 8) * 8
            self.H = (resize[1] // 8) * 8
        else:
            target_long = c.camera.resize_target
            if orig_w >= orig_h:
                scale_factor = target_long / orig_w
            else:
                scale_factor = target_long / orig_h
            self.W = (int(orig_w * scale_factor) // 8) * 8
            self.H = (int(orig_h * scale_factor) // 8) * 8

        self.W = max(self.W, c.camera.min_dimension)
        self.H = max(self.H, c.camera.min_dimension)

        logger.info("Original: %dx%d → Resized: %dx%d (aspect preserved, 8x aligned)", orig_w, orig_h, self.W, self.H)

        self.sp_scale = float(c.superpoint.sp_scale)
        self.sp_interval = max(int(c.superpoint.sp_interval), 1)
        if not (0.1 <= self.sp_scale <= 1.0):
            raise ValueError("sp_scale must be in [0.1, 1.0]")

        logger.info("Resolution: %dx%d", self.W, self.H)
        logger.info("SuperPoint config: fp16=%s, sp_scale=%s, sp_interval=%d", c.superpoint.sp_fp16, self.sp_scale, self.sp_interval)
        logger.info("Semantic SLAM (YOLO): %s", c.semantic.enabled)

        # 카메라 파라미터
        self.focal = max(self.W, self.H) * c.camera.focal_multiplier
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.K = np.array([[self.focal, 0, self.cx], [0, self.focal, self.cy], [0, 0, 1]])

        logger.info("Loading SuperPoint...")
        self.fe = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=4,  # 아키텍처 상수
            conf_thresh=c.superpoint.conf_thresh,
            nn_thresh=c.superpoint.nn_thresh,
            cuda=self.use_cuda,
            roi_sky=c.superpoint.roi_sky,
            roi_hood=c.superpoint.roi_hood
        )

        # CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=c.clahe.clip_limit,
            tileGridSize=tuple(c.clahe.tile_grid_size)
        )

        # Semantic Filter
        self.use_semantic = c.semantic.enabled
        if self.use_semantic:
            self.semantic_filter = SemanticFilter(conf_thresh=c.semantic.yolo_conf)
        else:
            self.semantic_filter = None
        self.matcher = BTMatcher(
            nn_thresh=c.superpoint.nn_thresh,
            use_cuda=self.use_cuda,
            mutual=c.matcher.mutual,
            ratio_thresh=c.matcher.ratio_thresh
        )
        self.loop_closure = LoopClosureManager(
            matcher=self.matcher,
            K=self.K,
            config=c.loop_closure,
        )

        self.prev_frame = None
        self.prev_kpts = None
        self.prev_desc = None
        self.prev_3d_pts = None # Initialize prev_3d_pts (로컬 좌표 - 프레임간 트래킹용)
        self.prev_3d_pts_world = None # 월드 좌표 (루프 클로저용)
        self.cur_pose = np.eye(4)
        
        # 포인트 필터 (하늘/구름 제거)
        self.point_filter = PointFilter(frame_h=self.H, frame_w=self.W)

        # 데이터 저장소
        self.all_map_points = []
        self.all_map_colors = []
        self.keyframes = [] # keyframe poses for visualization
        self.keyframe_indices = []
        self.keyframe_local_pts = []  # 키프레임별 로컬 3D 포인트 (재투영용)
        self.keyframe_original_poses = []  # 최적화 전 원본 포즈
        self.traj_points = []
        self.last_t_vec = np.array([0.0, 0.0, 1.0]) 
        self.accumulated_parallax = 0.0
        self.last_keyframe_idx = 0
        self.dead_reckoning_count = 0  # [Fix 1] 연속 추적 실패 횟수
        self.recent_speeds = []  # [Fix 4] 최근 이동 속도 기록 (스케일 복구용)
        
        # 새로운 3D Map Point 시스템
        self.map = Map()
        self.recent_kf_ids = []  # 매칭 대상으로 삼기 위한 최근 키프레임들 목록
        self.prev_kpt_to_mp = {} # 이전 프레임의 특징점 인덱스 -> MapPoint 매핑
        
        # g2o Sim3 Pose Graph 최적화기 설정
        self.g2o_optimizer = None
        self.g2o_edges = []
        self.jetson_scale = None  # legacy

        self.save_dir = str(output_dir)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        # 사용된 설정을 출력 디렉토리에 저장 (재현성)
        self.cfg.to_yaml(os.path.join(self.save_dir, "config_used.yaml"))

        # 실시간 2D 확인창 - 최적화: enable_viz=True인 경우에만 생성
        if self.enable_viz:
            cv2.namedWindow('Processing', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Processing', *self.cfg.viz.window_size)

        # =========================
        # Benchmark / Stats
        # =========================
        self.sp_ms_list = []           # SuperPoint 특징점 추출 시간 (ms)
        self.match_ms_list = []        # 특징점 매칭 시간 (ms)
        self.total_ms_list = []        # 전체 프레임 처리 시간 (ms)
        self.kpts_list = []            # 프레임별 검출된 특징점 수
        self.matches_list = []         # 프레임별 매칭된 특징점 쌍의 수
        self.inliers_list = []         # RANSAC 후 남은 유효한 매칭 수
        self.inlier_ratio_list = []    # 전체 매칭 중 유효한 매칭의 비율
        self.map_points_added_list = [] # 각 프레임에서 추가된 3D 맵 포인트 수

        # Detailed profiling (새로운 세부 타이밍)
        self.clahe_ms_list = []
        self.flow_ms_list = []
        self.filter_ms_list = []
        self.local_map_ms_list = []
        self.pnp_ms_list = []
        self.triangulation_ms_list = []
        self.visualization_ms_list = []
        self.ba_result = None  # BAResult (GBA 완료 후 채워짐)

        # 프레임별 CSV 로깅
        self._csv_writer = None
        self._csv_file = None
        if self.cfg.logging.csv_per_frame:
            csv_path = os.path.join(self.save_dir, self.cfg.logging.csv_path)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "frame", "sp_ms", "match_ms", "clahe_ms", "flow_ms", "filter_ms",
                "local_map_ms", "pnp_ms", "tri_ms", "vis_ms", "total_ms",
                "kpts", "matches", "inliers", "inlier_ratio", "method",
            ])

    def triangulate(self, R, t, p1, p2):
        # R, t map from curr to prev (X_prev = R * X_curr + t)
        # We want to find X in curr frame!
        P1 = self.K @ np.hstack((R, t))  # Projection for p1 (prev frame)
        P2 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection for p2 (curr frame)
        pts_4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        pts_3d = pts_4d[:3] / (pts_4d[3] + 1e-8)
        return pts_3d.T

    def mask_car(self, img):
        # 대시보드(차체) 마스킹
        h, w = img.shape[:2]
        img[int(h * self.cfg.camera.dashboard_mask_ratio):, :] = 0 
        return img

    def process(self):
        cap = cv2.VideoCapture(self.input_path)
        logger.info("Starting Analysis... (Visualization will appear at the end)")
        
        frame_idx = 0
        self.force_keyframe = False
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            img_curr = cv2.resize(frame, (self.W, self.H))
            img_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

            # CLAHE 적용 (그림자 구역 대비 강화) - 최적화: 선택적 적용
            clahe_t0 = time.perf_counter()
            if self.use_clahe:
                img_gray = self.clahe.apply(img_gray)
            clahe_t1 = time.perf_counter()

            img_masked = self.mask_car(img_gray.copy())
            frame_t0 = time.perf_counter()

            run_infer = (
                self.prev_frame is None
                or frame_idx % self.sp_interval == 0
                or self.force_keyframe
            )
            self.force_keyframe = False

            desc = None
            kpts = np.empty((0, 2))

            if run_infer:
                if self.sp_scale != 1.0:
                    sp_w = max(int(self.W * self.sp_scale), 8)
                    sp_h = max(int(self.H * self.sp_scale), 8)
                    img_sp = cv2.resize(img_masked, (sp_w, sp_h))
                else:
                    img_sp = img_masked

                # Semantic Filter (Dynamic Object Masking) - 사전 준비
                dyn_mask = None
                if self.use_semantic and self.semantic_filter is not None and self.semantic_filter.enabled:
                    # Resize original colored frame for YOLO
                    frame_resized = cv2.resize(frame, (self.W, self.H))
                    dyn_mask = self.semantic_filter.get_dynamic_mask(frame_resized)

                img_fe = (img_sp.astype(np.float32) / 255.0)

                # Superpoint 추론 시간 측정
                # perf_counter가 time보다 짧은 구간 측정에서 더 정확
                sp_t0 = time.perf_counter()
                pts, desc, _ = self.fe.run(img_fe)
                sp_t1 = time.perf_counter()
                kpts = pts[:2, :].T if pts.shape[1] > 0 else np.empty((0, 2))
                if self.sp_scale != 1.0 and len(kpts) > 0:
                    kpts = kpts / self.sp_scale

                # CUDA: desc를 GPU 텐서로 유지하여 Matcher까지 직통 전달
                # MPS/CPU: 소규모 텐서 연산은 CPU numpy가 빠름 — 즉시 변환
                if desc is not None and not self.use_cuda:
                    desc = desc.cpu().numpy()

                # Semantic Filter 적용 (가짜 코너 방지) - 추출 후 제거
                if dyn_mask is not None and len(kpts) > 0:
                    kpts_int = kpts.astype(int)
                    kx = np.clip(kpts_int[:, 0], 0, self.W - 1)
                    ky = np.clip(kpts_int[:, 1], 0, self.H - 1)
                    # dyn_mask가 True인 곳은 동적 객체이므로 제거
                    valid_mask = ~dyn_mask[ky, kx]
                    kpts = kpts[valid_mask]
                    if desc is not None:
                        if isinstance(desc, torch.Tensor):
                            desc = desc[:, torch.from_numpy(valid_mask).to(desc.device)]
                        else:
                            desc = desc[:, valid_mask]

                # 하늘 + 그림자 + 상단 영역 통합 필터 적용 (HSV 연산 1회로 통합)
                # 최적화: filter_on_sp_only=True인 경우 SuperPoint 추론 시에만 적용
                filter_t0 = time.perf_counter()
                should_apply_filter = (not self.filter_on_sp_only) or run_infer
                if len(kpts) > 0 and should_apply_filter:
                    combined_mask = self.point_filter.filter_shadow_and_top_combined(
                        img_curr,
                        kpts,
                        use_sky_filter=True,
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
                    kpts = kpts[combined_mask]
                    if desc is not None:
                        if isinstance(desc, torch.Tensor):
                            desc = desc[:, torch.from_numpy(combined_mask).to(desc.device)]
                        else:
                            desc = desc[:, combined_mask]
                filter_t1 = time.perf_counter()
            else:
                sp_t0 = time.perf_counter()
                sp_t1 = sp_t0
                filter_t0 = time.perf_counter()
                filter_t1 = filter_t0

            flow_t0 = time.perf_counter()
            flow_p1 = None
            flow_p2 = None
            if self.prev_frame is not None and self.prev_kpts is not None and len(self.prev_kpts) > 0:
                prev_pts = self.prev_kpts.astype(np.float32).reshape(-1, 1, 2)
                lk_params = dict(
                    winSize=(self.flow_win_size, self.flow_win_size),
                    maxLevel=self.flow_max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             self.cfg.optical_flow.termination_max_iter,
                             self.cfg.optical_flow.termination_epsilon),
                )

                # Forward: prev → curr
                curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, img_gray, prev_pts, None, **lk_params
                )

                # Backward: curr → prev (Forward-Backward 일관성 검증)
                if status_fwd is not None and curr_pts is not None:
                    back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                        img_gray, self.prev_frame, curr_pts, None, **lk_params
                    )

                    status_fwd = status_fwd.reshape(-1).astype(bool)
                    status_bwd = status_bwd.reshape(-1).astype(bool)

                    if back_pts is not None:
                        fb_error = np.linalg.norm(
                            prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
                        )
                        status = status_fwd & status_bwd & (fb_error < self.flow_fb_thresh)
                    else:
                        status = status_fwd & status_bwd

                    if status.any():
                        flow_p1 = self.prev_kpts[status]
                        flow_p2 = curr_pts.reshape(-1, 2)[status]
            flow_t1 = time.perf_counter()

            if not run_infer and flow_p2 is not None:
                kpts = flow_p2

            if self.prev_frame is None:
                self.prev_frame, self.prev_kpts, self.prev_desc = img_gray, kpts, desc
                self.prev_3d_pts = np.full((len(kpts), 3), np.nan)
                self.prev_3d_pts_world = np.full((len(kpts), 3), np.nan)
                self.curr_kpt_to_mp = {}  # Initialize empty match list for the first frame
                # 첫 프레임 키프레임 추가
                self.traj_points.append([0,0,0])
                self.add_keyframe(frame_idx, kpts, desc)
                frame_idx += 1
                continue

            max_pts = max(len(kpts), len(self.prev_kpts) if self.prev_kpts is not None else 0)
            curr_3d_pts_world = np.full((max_pts, 3), np.nan)  # 월드 좌표 병렬 추적
            self.curr_kpt_to_mp = {}

            # Matcher 시간 측정
            match_t0 = time.perf_counter()
            matches = np.empty((0, 2), dtype=int)
            use_flow = flow_p1 is not None and flow_p2 is not None
            if not use_flow and self.prev_desc is not None and desc is not None:
                matches = self.matcher.match(self.prev_desc, desc)
            match_t1 = time.perf_counter()

            # 프레임별 기본값 초기화(recoverPose 성공시에만 값이 갱신/ 실패하면 0 유지)
            valid_step = False
            inliers = 0
            inlier_ratio = 0.0
            map_points_added = 0

            # 디버깅: 특징점 검출 및 매칭 상태 모니터링
            # - desc_dim: descriptor 차원 수
            # - kpts: 현재 프레임에서 검출된 특징점 수
            # - matches: 이전 프레임과 현재 프레임 사이의 매칭된 특징점 쌍의 수
            match_count = len(flow_p1) if use_flow else len(matches)
            logger.debug("frame %d: desc_dim=%s, kpts=%d, matches=%d", frame_idx, None if desc is None else desc.shape[0], len(kpts), match_count)

            if use_flow and len(flow_p1) > 8:
                p1_idx = np.where(status)[0]
                p2_idx = np.arange(len(flow_p2))
                p1 = flow_p1.astype(np.float64)
                p2 = flow_p2.astype(np.float64)
                # Inherit 3D points (월드 동시 상속)
                curr_3d_pts_world[p2_idx] = self.prev_3d_pts_world[p1_idx]
                # Inherit MapPoint associations
                for i, prev_kpt_idx in enumerate(p1_idx):
                    if prev_kpt_idx in self.prev_kpt_to_mp:
                        self.curr_kpt_to_mp[p2_idx[i]] = self.prev_kpt_to_mp[prev_kpt_idx]
            elif len(matches) > 8:
                p1_idx = matches[:, 0]
                p2_idx = matches[:, 1]
                p1 = self.prev_kpts[p1_idx, :2].astype(np.float64)
                p2 = kpts[p2_idx, :2].astype(np.float64)
                # Inherit 3D points (월드 동시 상속)
                curr_3d_pts_world[p2_idx] = self.prev_3d_pts_world[p1_idx]
                # Inherit MapPoint associations
                for i, prev_kpt_idx in enumerate(p1_idx):
                    if prev_kpt_idx in self.prev_kpt_to_mp:
                        self.curr_kpt_to_mp[p2_idx[i]] = self.prev_kpt_to_mp[prev_kpt_idx]
            else:
                p1 = None
                p2 = None
                
            # [NEW] 로컬 맵 트래킹 강화 (PnP 강건성 확보)
            # flow나 단순 이전 프레임 매칭으로 확보한 curr_3d_pts_world 외에도,
            # 현재 카메라 포즈 주변의 로컬 맵포인트들을 불러와 추가로 매칭을 시도합니다.
            local_map_t0 = time.perf_counter()
            local_map_pts_dict = self.map.get_recent_map_points(num_recent_keyframes=self.cfg.performance.local_map_keyframes)

            # 최적화: local_map_limit이 설정된 경우 MapPoint 개수 제한
            if self.local_map_limit > 0 and len(local_map_pts_dict) > self.local_map_limit:
                # 최근 MapPoint만 유지 (ID 기준 정렬)
                sorted_items = sorted(local_map_pts_dict.items(), key=lambda x: x[0], reverse=True)
                local_map_pts_dict = dict(sorted_items[:self.local_map_limit])
            
            # 매칭을 수행할 추가적인 MapPoint 후보들 추려내기
            extra_obj_pts = []
            extra_img_pts = []
            extra_mp_ids = []
            extra_kpt_idxs = []
            
            # [FIX] optical flow 사용 시에도 local map tracking 활성화 (PnP 강건성 향상)
            # use_flow가 True여도 desc가 있으면 추가 매칭을 시도합니다
            if len(local_map_pts_dict) > 0 and desc is not None:
                # 1. 맵포인트들을 현재 카메라(추정) 뷰로 투영
                proj_pts, _ = cv2.projectPoints(
                    np.array([mp.pos3d for mp in local_map_pts_dict.values()]).astype(np.float32),
                    cv2.Rodrigues(self.cur_pose[:3, :3].T)[0],
                    -self.cur_pose[:3, :3].T @ self.cur_pose[:3, 3],
                    self.K, None
                )
                proj_pts = proj_pts.reshape(-1, 2)
                
                # 2. 이미지 범위를 벗어난 곳 필터링
                h, w = self.prev_frame.shape if self.prev_frame is not None else (360, 640)
                in_image = (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < w) & \
                           (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < h)
                           
                visible_mps = [mp for i, mp in enumerate(local_map_pts_dict.values()) if in_image[i]]
                visible_proj_pts = proj_pts[in_image]
                
                # 3. 디스크립터 매칭 수행 (보이는 맵포인트 vs 현재 프레임 kpts)
                valid_mask = [mp.descriptor is not None for mp in visible_mps]
                valid_mps = [mp for i, mp in enumerate(visible_mps) if valid_mask[i]]
                valid_proj_pts = visible_proj_pts[valid_mask]
                
                if len(valid_mps) > 0:
                    mp_descs = np.array([mp.descriptor for mp in valid_mps]).T
                    if mp_descs.shape[1] > 0:
                        # 매칭 수행
                        knn_matches = self.matcher.match(mp_descs, desc)
                        
                        for m in knn_matches:
                            mp_idx = m[0]
                            kpt_idx = m[1]
                            
                            # 이미 매칭된 kpt이거나 mp인지 중복 검사
                            if kpt_idx not in self.curr_kpt_to_mp:
                                mp = visible_mps[mp_idx]
                                # 위치 재검증 (이미 계산된 투영 픽셀 사용)
                                pt_proj = visible_proj_pts[mp_idx]
                                
                                dist = np.linalg.norm(pt_proj - kpts[kpt_idx][:2])
                                if dist < self.cfg.performance.local_map_projection_dist:
                                    extra_obj_pts.append(mp.pos3d)
                                    extra_img_pts.append(kpts[kpt_idx][:2])
                                    extra_mp_ids.append(mp)
                                    extra_kpt_idxs.append(kpt_idx)
                                    
            if len(extra_obj_pts) > 0:
                logger.debug("frame %d: [Local Map Tracking] Added %d extra correspondences.", frame_idx, len(extra_obj_pts))
            local_map_t1 = time.perf_counter()

            # --- 1. PnP 시도 (충분한 3D 점이 있을 때) ---
            pnp_t0 = time.perf_counter()
            obj_pts_list = []
            img_pts_list = []
            
            # 기존 P1-P2 추적분 추가
            if "p2_idx" in locals() and p2_idx is not None and len(p2_idx) > 0:
                valid_3d_mask = ~np.isnan(curr_3d_pts_world[p2_idx, 0])
                if np.any(valid_3d_mask):
                    obj_pts_list.append(curr_3d_pts_world[p2_idx][valid_3d_mask].astype(np.float32))
                    img_pts_list.append(p2[valid_3d_mask].astype(np.float32))
            
            # 신규 로컬 맵 매칭분 추가
            if len(extra_obj_pts) > 0:
                obj_pts_list.append(np.array(extra_obj_pts, dtype=np.float32))
                img_pts_list.append(np.array(extra_img_pts, dtype=np.float32))
                
                # 매칭된 MapPoint 연결 끊어지지 않게 저장
                for i, kpt_idx in enumerate(extra_kpt_idxs):
                    self.curr_kpt_to_mp[kpt_idx] = extra_mp_ids[i]
                    # 역으로 curr_3d_pts_world 에도 꽂아넣기 (나중에 삼각측량 시 중복 안되게)
                    curr_3d_pts_world[kpt_idx] = extra_mp_ids[i].pos3d
            
            pnp_success = False
            # mask와 inliers를 전방 선언하여 NameError 방지
            mask = None
            inliers = 0

            # 1. PnP 시도 (충분한 3D 점이 있을 때)
            if len(obj_pts_list) > 0:
                obj_pts = np.vstack(obj_pts_list)
                img_pts = np.vstack(img_pts_list)
                
                if len(obj_pts) >= self.cfg.pnp.min_points:
                    success, rvec, tvec_pnp, inliers_pnp = cv2.solvePnPRansac(
                        obj_pts, img_pts, self.K, None,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        reprojectionError=self.cfg.pnp.reprojection_error
                    )

                    if success and inliers_pnp is not None and len(inliers_pnp) > self.cfg.pnp.min_inliers:
                        R_pnp, _ = cv2.Rodrigues(rvec)
                        t_pnp = tvec_pnp
                        
                        # PnP 반환값은 T_CW (World to Camera). 이를 T_rel (Prev Camera to Curr Camera)로 변환
                        R_WC_new = R_pnp.T
                        t_WC_new = -R_pnp.T @ t_pnp
                        T_WC_new = np.eye(4)
                        T_WC_new[:3, :3] = R_WC_new
                        T_WC_new[:3, 3] = t_WC_new[:, 0]
                        
                        T_WC_prev_inv = np.linalg.inv(self.cur_pose)
                        T_rel_mat = T_WC_prev_inv @ T_WC_new
                        
                        R_rel = T_rel_mat[:3, :3]
                        t_rel = T_rel_mat[:3, 3]
                        
                        tvec_len = np.linalg.norm(t_rel)
                        # 필터 1: 절대 상한
                        if self.cfg.pnp.min_translation < tvec_len < self.cfg.pnp.max_translation:
                            # 필터 2: 속도 기반 점프 감지 — 최근 이동 속도 대비 과도한 이동 거부
                            speed_rejected = False
                            if len(self.recent_speeds) >= 3:
                                median_speed = np.median(self.recent_speeds[-10:])
                                speed_thresh = max(median_speed * self.cfg.motion.speed_jump_multiplier,
                                                   self.cfg.motion.min_speed_thresh)
                                if tvec_len > speed_thresh:
                                    logger.warning("frame %d: [PnP REJECTED] speed jump %.3fm > thresh %.3fm (median_speed=%.3f)", frame_idx, tvec_len, speed_thresh, median_speed)
                                    speed_rejected = True

                            if speed_rejected:
                                pnp_success = False
                            else:
                                # [Fix 3] PnP 각도 일관성 체크 — 이전 방향과 120° 이상 반전이면 거부
                                last_t_norm = np.linalg.norm(self.last_t_vec)
                                if last_t_norm > 1e-4:
                                    cos_angle = np.dot(t_rel, self.last_t_vec) / (tvec_len * last_t_norm + 1e-8)
                                    if cos_angle < self.cfg.motion.direction_reversal_cos:
                                        logger.warning("frame %d: [PnP REJECTED] direction reversal cos=%.3f", frame_idx, cos_angle)
                                        pnp_success = False
                                    else:
                                        pnp_success = True
                                else:
                                    pnp_success = True
                            
                            if pnp_success:
                                R = R_rel
                                t_vec = t_rel
                                inliers = len(inliers_pnp)
                                inlier_ratio = inliers / max(match_count, 1)
                                
                                # PnP 성공 시, 삼각측량용 마스크 생성
                                if p1 is not None and len(p1) > 0:
                                    mask = np.zeros(len(p1), dtype=np.uint8)
                                    mask[:] = 1
                                
                                logger.debug("frame %d: [PnP] matches=%d, inliers=%d, ratio=%.3f, scale=%.4f, t=%s", frame_idx, match_count, inliers, inlier_ratio, tvec_len, t_vec.round(3))

            # 2. PnP 실패 시 Essential Matrix 또는 Homography (부트스트랩 및 평면/전진 모션 강건화)
            if not pnp_success and p1 is not None and p2 is not None and len(p1) >= 8:
                E, mask_E = cv2.findEssentialMat(p2, p1, self.K, method=cv2.RANSAC, prob=0.999,
                                                threshold=self.cfg.epipolar.essential_threshold)
                H, mask_H = cv2.findHomography(p2, p1, cv2.RANSAC, self.cfg.epipolar.homography_threshold)
                
                inliers_E = np.count_nonzero(mask_E) if mask_E is not None else 0
                inliers_H = np.count_nonzero(mask_H) if mask_H is not None else 0

                use_homography = False
                # Homography inlier가 Essential inlier와 비슷하거나 더 많으면 Planar로 간주 (고속도로/전진모션 특화)
                if inliers_H > inliers_E * self.cfg.epipolar.homography_selection_ratio and inliers_H > self.cfg.epipolar.homography_min_inliers:
                    use_homography = True
                
                if use_homography and H is not None:
                    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, self.K)
                    if num_solutions > 0:
                        # Chirality 기반 해 선택: 매칭된 점들이 카메라 앞에 있는 해만 선택
                        # filterHomographyDecompByVisibleRefpoints는 유효한 해의 인덱스를 반환
                        try:
                            _, filtered = cv2.filterHomographyDecompByVisibleRefpoints(
                                rotations, normals, p1, p2
                            )
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
                        inliers = inliers_H
                        inlier_ratio = inliers / max(match_count, 1)
                        t_vec = t[:, 0]
                        # 3D 맵 스케일 유지: 단위 벡터에 이전 프레임 이동 거리(속도)를 곱하여 스케일 복원
                        vec_norm = np.linalg.norm(t_vec) + 1e-6
                        # [Fix 4] 이동 평균 스케일 복구 (단일 프레임 의존 제거)
                        if len(self.recent_speeds) > 0:
                            prev_speed = np.median(self.recent_speeds[-10:])
                        else:
                            prev_speed = np.linalg.norm(self.last_t_vec)
                        t_vec = (t_vec / vec_norm) * (prev_speed if prev_speed > 0 else 1.0)
                        
                        mask = mask_H
                        
                        logger.debug("frame %d: [Homography] matches=%d, inliers=%d, ratio=%.3f, t=%s", frame_idx, match_count, inliers, inlier_ratio, t_vec.round(3))
                        if np.isfinite(t_vec).all() and inliers > 10:
                            pnp_success = True

                if not pnp_success and E is not None:
                    _, R, t, mask = cv2.recoverPose(E, p2, p1, self.K, mask=mask_E)
                    inliers = inliers_E
                    inlier_ratio = inliers / max(match_count, 1)
                    t_vec = t[:, 0]
                    # 3D 맵 스케일 유지: 단위 벡터에 이전 프레임 이동 거리(속도)를 곱하여 스케일 복원
                    vec_norm = np.linalg.norm(t_vec) + 1e-6
                    # [Fix 4] 이동 평균 스케일 복구 (단일 프레임 의존 제거)
                    if len(self.recent_speeds) > 0:
                        prev_speed = np.median(self.recent_speeds[-10:])
                    else:
                        prev_speed = np.linalg.norm(self.last_t_vec)
                    t_vec = (t_vec / vec_norm) * (prev_speed if prev_speed > 0 else 1.0)
                    # recoverPose는 내부적으로 chirality check를 수행하여 올바른 부호를 반환하므로
                    # 방향 가드 없이 결과를 그대로 신뢰합니다.
                    
                    logger.debug("frame %d: [Ess] matches=%d, inliers=%d, ratio=%.3f, t=%s", frame_idx, match_count, inliers, inlier_ratio, t_vec.round(3))
                    if np.isfinite(t_vec).all() and inliers > 10:
                        pnp_success = True
            pnp_t1 = time.perf_counter()

            # 3. 최적 포즈 적용 및 지도 업데이트
            triangulation_t0 = time.perf_counter()
            if pnp_success:
                # --- [안정화 로직: IMU 부재로 인한 Y축 및 각도 드리프트 억제] ---
                
                # [핵심] R(회전행렬)은 절대 수정하지 않음 → Yaw(커브/회전) 보존
                # 기존 R.T 반전 완전 제거, 코사인 방향 락도 제거
                # recoverPose / decomposeHomography / PnP 모두 chirality check로 올바른 부호 반환
                
                # 원본 이동 속도 보존 (X, Y 억제로 인한 스케일 축소 방지)
                t_speed = np.linalg.norm(t_vec)

                # 삼각측량용 순수 R, t 보존 (댐핑 전)
                R_orig = R.copy()
                t_orig = t_vec.copy()

                # 2. 횡이동(X축) 억제 (Turn이 작으면 X 이동 억제)
                if self.highway_mode:
                    turn_amount = np.abs(np.arctan2(R[0,2], R[2,2]))
                    damp_x = np.clip(turn_amount * 10.0, 0.1, 1.0)
                    t_vec[0] *= damp_x 
                    
                # 3. Y축(상하 높이) 이동 극단적 억제 (IMU 역할: 평지 주행 가정)
                t_vec[1] *= self.cfg.stabilization.y_damping
                
                # 댐핑 후 스케일 복원 (속도 감소 방지)
                t_len_after = np.linalg.norm(t_vec)
                if t_len_after > 1e-6 and t_speed > 0:
                    t_vec = (t_vec / t_len_after) * t_speed
                
                # [Fix 2] 카메라 Pitch/Roll 억제 실제 구현 (기존 Dead Code 수정)
                # 단안 카메라 특성상 전진 시 바닥을 보면 위로 올라가는것처럼 착각함
                pitch_roll_damp = self.cfg.stabilization.pitch_roll_damping
                euler_ret = cv2.RQDecomp3x3(R)
                euler_angles = np.array(euler_ret[0])  # (Pitch, Yaw, Roll) in degrees
                Qx, Qy, Qz = euler_ret[1], euler_ret[2], euler_ret[3]  # 분해된 기본 회전행렬
                
                # Pitch(X축)와 Roll(Z축) 억제 적용
                pitch_rad = np.deg2rad(euler_angles[0] * pitch_roll_damp)
                yaw_rad = np.deg2rad(euler_angles[1])  # Yaw는 그대로 유지
                roll_rad = np.deg2rad(euler_angles[2] * pitch_roll_damp)
                
                # 억제된 Euler 각도로 Rotation Matrix 재구성 (Rz * Ry * Rx)
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                               [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
                Ry = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                               [0, 1, 0],
                               [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
                Rz = np.array([[np.cos(roll_rad), -np.sin(roll_rad), 0],
                               [np.sin(roll_rad), np.cos(roll_rad), 0],
                               [0, 0, 1]])
                R_damped = Rz @ Ry @ Rx
                
                # 5. [Fix 2] 적응적 관성 혼합 (방향 변화에 따라 가중치 조절)
                last_t_norm = np.linalg.norm(self.last_t_vec)
                t_norm = np.linalg.norm(t_vec)
                if last_t_norm > 1e-6 and t_norm > 1e-6:
                    cos_sim = np.dot(t_vec, self.last_t_vec) / (t_norm * last_t_norm)
                    # 같은 방향(cos~1): 관성 0.4, 반대 방향(cos~-1): 관성 0.0
                    inertia_w = np.clip(cos_sim * self.cfg.stabilization.inertia_weight, 0.0, self.cfg.stabilization.inertia_weight)
                else:
                    inertia_w = 0.0
                mixed_t = t_vec * (1.0 - inertia_w) + self.last_t_vec * inertia_w
                mixed_len = np.linalg.norm(mixed_t)
                if mixed_len > 1e-6 and t_speed > 0:
                    t_vec = (mixed_t / mixed_len) * t_speed
                else:
                    t_vec = mixed_t
                    
                self.last_t_vec = t_vec.copy()
                self.dead_reckoning_count = 0  # [Fix 1] 추적 성공 시 카운터 리셋
                # [Fix 4] 속도 기록
                self.recent_speeds.append(float(np.linalg.norm(t_vec)))
                if len(self.recent_speeds) > self.cfg.stabilization.speed_history_len:
                    self.recent_speeds = self.recent_speeds[-self.cfg.stabilization.speed_history_len:]
                
                # Pose Update (Relative to absolute)
                T_rel = np.eye(4)
                T_rel[:3, :3] = R_damped
                T_rel[:3, 3] = t_vec
                self.cur_pose = self.cur_pose @ T_rel
                
                # --- [절대 Y축 높이 강제 고정 (Hard Clip)] --- 
                # 차량이 하늘로 날아가거나 땅굴을 파고 들어가는 현상 원천 차단
                self.cur_pose[1, 3] = np.clip(self.cur_pose[1, 3], *self.cfg.stabilization.y_clip)
                
                valid_step = True
                
                if p1 is not None and "mask" in locals() and mask is not None:
                    mask = mask.ravel().astype(bool)
                    # Avoid IndexError by ensuring mask is not longer than p1
                    valid_len = min(len(p1), len(mask))
                    p1_m = p1[:valid_len][mask[:valid_len]]
                    p2_m = p2[:valid_len][mask[:valid_len]]
                    
                    if "p2_idx" in locals() and p2_idx is not None:
                        p2_idx_m = p2_idx[:valid_len][mask[:valid_len]]
                    else:
                        p2_idx_m = np.arange(len(p2_m)) # flow가 안 쓰였을 경우 fallback
                        
                    if len(p1_m) > 0:
                        local_pts = self.triangulate(R_orig, t_orig.reshape(3,1), p1_m, p2_m)
                    
                        # 필터링 (너무 멀거나 비상식적인 점 제거 - 주변 건물/도로 환경 고려하여 반경 확대)
                        tri = self.cfg.triangulation
                        valid = (local_pts[:, 2] > tri.min_depth) & (local_pts[:, 2] < tri.max_depth) & \
                                (np.abs(local_pts[:, 0]) < tri.max_x_extent) & (np.abs(local_pts[:, 1]) < tri.max_y_extent)
                        local_pts = local_pts[valid]
                        p1_m_filtered = p1_m[valid]
                        p2_m_filtered = p2_m[valid]
                        # p2_idx_m과 valid 크기 불일치 방지
                        if len(p2_idx_m) == len(valid):
                            valid_indices = p2_idx_m[valid]
                        else:
                            valid_indices = np.arange(len(local_pts))
                        
                        # 월드 좌표로 즉시 변환하여 병렬 저장 (루프 클로저용)
                        world_pts_for_lc = (self.cur_pose[:3, :3] @ local_pts.T).T + self.cur_pose[:3, 3]
                        
                        # valid_indices가 curr_3d_pts_world 범위를 넘지 않도록 필터링
                        bounds_ok = valid_indices < len(curr_3d_pts_world)
                        valid_indices = valid_indices[bounds_ok]
                        world_pts_for_lc = world_pts_for_lc[bounds_ok]
                        local_pts = local_pts[bounds_ok]
                        p1_m_filtered = p1_m_filtered[bounds_ok]
                        p2_m_filtered = p2_m_filtered[bounds_ok]
                        
                        curr_3d_pts_world[valid_indices] = world_pts_for_lc
                        
                        # [NEW] 삼각측량된 새 포인트들을 MapPoint 객체로 등록
                        for i, pt3d in enumerate(world_pts_for_lc):
                            kpt_idx = valid_indices[i]
                            if kpt_idx not in self.curr_kpt_to_mp:
                                # When using flow, p2_idx refers to flow_p2, not the independently extracted kpts/desc.
                                if not use_flow and desc is not None and kpt_idx < desc.shape[1]:
                                    desc_i = desc[:, kpt_idx].cpu().numpy() if isinstance(desc, torch.Tensor) else desc[:, kpt_idx]
                                else:
                                    desc_i = None
                                mp = self.map.create_map_point(pt3d, desc_i)
                                
                                # 맵 포인트의 태생 구조상 직전 프레임(p1)과 현재 프레임(p2)에서 모두 매칭된 것이므로
                                # 최소 관측 횟수 2회를 태생적으로 확보시켜 Culling 광탈을 막습니다. 
                                if self.last_keyframe_idx >= 0:
                                    mp.add_observation(self.last_keyframe_idx, p1_m_filtered[i])
                                mp.add_observation(frame_idx, p2_m_filtered[i])
                                
                                self.curr_kpt_to_mp[kpt_idx] = mp
                        
                        if len(local_pts) > 0:
                            # 로컬 3D 포인트를 현재 키프레임에 등록 (재투영용)
                            if len(self.keyframe_local_pts) > 0:
                                self.keyframe_local_pts[-1].append(local_pts.copy())

                            # 월드 좌표계 투영 (실시간 시각화용, 최적화 후 재구축됨)
                            world_pts = (self.cur_pose[:3, :3] @ local_pts.T).T + self.cur_pose[:3, 3]
                            
                            # 투영된 포인트 바로 저장
                            map_points_added = int(world_pts.shape[0])

                            cols = get_height_color(world_pts[:, 1])
                            
                            # 저장
                            self.all_map_points.append(world_pts)
                            self.all_map_colors.append(cols)
            triangulation_t1 = time.perf_counter()

            # 4. 추적 결과에 따른 포즈 확정 및 관성 주행 처리
            if not valid_step:
                # [Fix 1] Dead Reckoning 속도 감쇠 — 연속 실패 시 점진적 감속
                self.dead_reckoning_count += 1
                decay = self.cfg.stabilization.dead_reckoning_decay ** self.dead_reckoning_count
                T_rel = np.eye(4)
                T_rel[:3, 3] = self.last_t_vec * decay
                self.cur_pose = self.cur_pose @ T_rel

            # [Fix 5] 궤적 정체 감지 — 10프레임 동안 0.5m 미만 이동 시 강제 키프레임 재추출
            if len(self.traj_points) >= self.cfg.keyframe.stall_lookback:
                recent_dist = np.linalg.norm(
                    np.array(self.cur_pose[:3, 3]) - np.array(self.traj_points[-self.cfg.keyframe.stall_lookback])
                )
                if recent_dist < self.cfg.keyframe.stall_distance:
                    self.force_keyframe = True
                    self.last_t_vec *= 0.0  # 관성 리셋
                    self.dead_reckoning_count = 0
                    logger.warning("frame %d: [STALL DETECTED] Forcing keyframe re-extraction", frame_idx)

            # 궤적 저장
            curr_t = self.cur_pose[:3, 3]
            self.traj_points.append(curr_t)            
            # --- [Keyframe 관리 로직 강화] ---
            need_keyframe = False
            
            # 1. Parallax (시차) 기반
            if p1 is not None and p2 is not None and len(p1) > 0:
                # p1, p2는 현재 매칭된 특징점 (이전 프레임 - 현재 프레임)
                disp = np.linalg.norm(p2 - p1, axis=1)
                median_disp = np.median(disp) if len(disp) > 0 else 0
                self.accumulated_parallax += median_disp
                
                if self.accumulated_parallax >= self.cfg.keyframe.parallax_threshold:
                    need_keyframe = True
            
            # 2. Inlier 감소 기반 (적응형 임계값)
            if p1 is not None and p2 is not None:
                recent = self.inliers_list[-30:] if len(self.inliers_list) > 0 else [80]
                adaptive_thresh = max(int(np.median(recent) * self.cfg.keyframe.adaptive_inlier_ratio), self.cfg.keyframe.min_adaptive_inliers)
                if inliers < adaptive_thresh and (frame_idx - self.last_keyframe_idx) >= self.cfg.keyframe.min_gap:
                    need_keyframe = True
                    
            # 3. 일정 프레임 이상 경과 시 강제 추가 (Fallback)
            if (frame_idx - self.last_keyframe_idx) >= self.cfg.keyframe.max_gap:
                need_keyframe = True

            if need_keyframe:
                if desc is None:
                    # Descriptors가 없으면 다음 프레임에서 강제로 뽑도록 설정
                    self.force_keyframe = True
                else:
                    # Optical Flow를 사용했다면 curr_kpt_to_mp의 인덱스는 flow_p2 기준입니다.
                    # 이를 새로 추출된 kpts 인덱스에 매핑(Alignment)해야 KeyFrame 생성 시 에러가 나지 않습니다.
                    if use_flow and len(kpts) > 0:
                        aligned_kpt_to_mp = {}
                        for f_idx, mp in self.curr_kpt_to_mp.items():
                            if f_idx < len(flow_p2):
                                # 브로드캐스팅 거리 계산
                                dist = np.linalg.norm(kpts[:, :2] - flow_p2[f_idx], axis=1)
                                best_idx = int(np.argmin(dist))
                                if dist[best_idx] < self.cfg.keyframe.alignment_dist:
                                    aligned_kpt_to_mp[best_idx] = mp
                                    # 이 때 MapPoint의 descriptor 업데이트 (신규 추출된 것으로)
                                    mp.update_descriptor(desc[:, best_idx].cpu().numpy() if isinstance(desc, torch.Tensor) else desc[:, best_idx])
                        self.curr_kpt_to_mp = aligned_kpt_to_mp

                    self.add_keyframe(frame_idx, kpts, desc, curr_3d_pts_world, T_rel.copy())
                    self.last_keyframe_idx = frame_idx
                    self.accumulated_parallax = 0.0
                    
                    # [NEW] KeyFrame 생성 시, 현재 연결된 MapPoint들에게 관측 기록을 남겨 Culling에서 살아남게 함
                    if self.last_keyframe_idx >= 0:
                        for kpt_idx, mp in self.curr_kpt_to_mp.items():
                            if kpt_idx < len(kpts):
                                mp.add_observation(self.last_keyframe_idx, kpts[kpt_idx][:2])
                    
                    # 새 키프레임 생성 후 불량 맵 포인트 정리 (Culling)
                    # 키프레임 간 간격을 유지하면서 불필요한 관측 횟수 2 이하의 특징점 메모리 해제
                    if len(self.keyframes) % self.cfg.keyframe.culling_interval == 0:
                        self.map.cull_bad_map_points(min_observations=self.cfg.map_culling.min_observations)

            # 2D 뷰 표시 - 최적화: enable_viz=False인 경우 스킵
            vis_t0 = time.perf_counter()
            if self.enable_viz:
                img_vis = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
                for kp in kpts: cv2.circle(img_vis, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)
                cv2.putText(img_vis, f"Frame: {frame_idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow('Processing', img_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            vis_t1 = time.perf_counter()

            # 프레임 끝 시간
            frame_t1 = time.perf_counter()

            # ms 계산 + append
            sp_ms = (sp_t1 - sp_t0) * 1000.0
            match_ms = (match_t1 - match_t0) * 1000.0
            total_ms = (frame_t1 - frame_t0) * 1000.0
            clahe_ms = (clahe_t1 - clahe_t0) * 1000.0
            flow_ms = (flow_t1 - flow_t0) * 1000.0
            filter_ms = (filter_t1 - filter_t0) * 1000.0
            local_map_ms = (local_map_t1 - local_map_t0) * 1000.0
            pnp_ms = (pnp_t1 - pnp_t0) * 1000.0
            triangulation_ms = (triangulation_t1 - triangulation_t0) * 1000.0
            vis_ms = (vis_t1 - vis_t0) * 1000.0

            self.sp_ms_list.append(sp_ms)
            self.match_ms_list.append(match_ms)
            self.total_ms_list.append(total_ms)
            self.kpts_list.append(int(len(kpts)))
            self.matches_list.append(int(len(matches)))
            self.inliers_list.append(int(inliers))
            self.inlier_ratio_list.append(float(inlier_ratio))
            self.map_points_added_list.append(int(map_points_added))

            self.clahe_ms_list.append(clahe_ms)
            self.flow_ms_list.append(flow_ms)
            self.filter_ms_list.append(filter_ms)
            self.local_map_ms_list.append(local_map_ms)
            self.pnp_ms_list.append(pnp_ms)
            self.triangulation_ms_list.append(triangulation_ms)
            self.visualization_ms_list.append(vis_ms)

            # 프레임별 CSV 로깅
            if self._csv_writer is not None:
                self._csv_writer.writerow([
                    frame_idx, f"{sp_ms:.2f}", f"{match_ms:.2f}", f"{clahe_ms:.2f}",
                    f"{flow_ms:.2f}", f"{filter_ms:.2f}", f"{local_map_ms:.2f}",
                    f"{pnp_ms:.2f}", f"{triangulation_ms:.2f}", f"{vis_ms:.2f}",
                    f"{total_ms:.2f}", len(kpts), len(matches), inliers,
                    f"{inlier_ratio:.3f}", "PnP" if pnp_success else "DR",
                ])

            if run_infer:
                self.prev_desc = desc
                if len(kpts) > 0:
                    self.prev_kpts = kpts
                    self.prev_kpt_to_mp = self.curr_kpt_to_mp.copy()
                elif flow_p2 is not None and len(flow_p2) > 0:
                    self.prev_kpts = flow_p2
                    self.prev_kpt_to_mp = self.curr_kpt_to_mp.copy()
            else:
                if flow_p2 is not None and len(flow_p2) > 0:
                    self.prev_kpts = flow_p2
                    self.prev_kpt_to_mp = self.curr_kpt_to_mp.copy()
            self.prev_frame = img_gray
            self.prev_3d_pts_world = curr_3d_pts_world
            frame_idx += 1

        logger.info("Video Finished. Building Final Scene...")
        cap.release()
        cv2.destroyAllWindows()

        # CSV 로깅 종료
        if self._csv_file is not None:
            self._csv_file.close()
            logger.info("Per-frame CSV saved to: %s", os.path.join(self.save_dir, self.cfg.logging.csv_path))
        
        # Global Bundle Adjustment 실행 (최종 맵 최적화)
        self.run_global_ba()
        
        self.print_perf_summary()
        self.visualize_final_result()

    def add_keyframe(self, frame_idx, kpts, desc, pts_3d=None, T_rel=None):
        self.keyframes.append(self.cur_pose.copy())
        self.keyframe_indices.append(frame_idx)
        self.keyframe_local_pts.append([])  # 이 키프레임에 등록될 로컬 3D 포인트들
        self.keyframe_original_poses.append(self.cur_pose.copy())

        # GPU 텐서 → numpy 변환 (KeyFrame/LoopClosure 저장 경계)
        desc_np = desc.cpu().numpy() if isinstance(desc, torch.Tensor) else desc

        # [NEW] Create and store KeyFrame object in Map
        kf_obj = KeyFrame(frame_idx, self.cur_pose.copy(), kpts, desc_np)
        for kpt_idx, mp in self.curr_kpt_to_mp.items():
            kf_obj.add_map_point_association(kpt_idx, mp.id)
            mp.add_observation(frame_idx, kpts[kpt_idx][:2])
        self.map.add_keyframe(kf_obj)

        node_idx = len(self.keyframes) - 1

        if node_idx > 0 and T_rel is not None:
            self.g2o_edges.append({
                'type': 'odom',
                'from': node_idx - 1,
                'to': node_idx,
                'transform': T_rel.copy(),
                'scale': 1.0,
                'information_scale': 1.0,
            })

        # pts_3d는 이미 월드 좌표계 (curr_3d_pts_world에서 전달됨)
        self.loop_closure.add_keyframe(frame_idx, kpts, desc_np, pts_3d)
        loop = self.loop_closure.find_loop(frame_idx, kpts, desc_np)
        if loop is not None:
            # Sim3에서는 스케일 정보를 보존하여 전달 (정규화 없음!)
            scale = loop.scale
            logger.info("[Pose Graph] g2o Sim3 Edge: Scale=%.3f", scale)
            
            # 루프 클로저 엣지 저장
            self.g2o_edges.append({
                'type': 'loop',
                'from': loop.match_index,
                'to': node_idx,
                'transform': loop.transform,
                'scale': scale,
                'information_scale': 1.0,  # 루프 엣지는 본 체인에서 특수 처리
            })
            self.optimize_pose_graph()

        # LBA는 실시간 추적 시 cur_pose를 오염시키므로 비활성화
        # GBA만 후처리로 실행합니다 (process() 루프 종료 후)

    def _build_g2o_optimizer(self):
        """g2o Sim3 Pose Graph 최적화기 구축"""
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)
        return optimizer

    def _run_ba(self, kf_ids, num_iterations=10, fix_first=True):
        """BA thin wrapper → slam.bundle_adjustment 모듈 호출"""
        result = run_bundle_adjustment(
            slam_map=self.map,
            K=self.K,
            ba_config=self.cfg.ba,
            kf_ids=kf_ids,
            keyframes_list=self.keyframes,
            keyframe_indices=self.keyframe_indices,
            num_iterations=num_iterations,
            fix_first=fix_first,
        )
        if result is not None:
            self.ba_result = result
        return result

    def run_local_ba(self):
        """Local BA: 최근 N개 키프레임 최적화"""
        recent_kf_ids = sorted(self.map.keyframes.keys(), reverse=True)[:self.cfg.ba.local_keyframes]
        if len(recent_kf_ids) < 2:
            return
        self._run_ba(recent_kf_ids, num_iterations=self.cfg.ba.local_iterations, fix_first=True)

    def run_global_ba(self):
        """Global BA: 전체 키프레임 최적화"""
        all_kf_ids = sorted(self.map.keyframes.keys())
        if len(all_kf_ids) < 2:
            return
        logger.info("Running Global Bundle Adjustment...")
        self._run_ba(all_kf_ids, num_iterations=self.cfg.ba.global_iterations, fix_first=True)

        # GBA 후 전체 키프레임 Y축 높이 강제 클리핑 (평지 주행 가정)
        for i in range(len(self.keyframes)):
            self.keyframes[i][1, 3] = np.clip(self.keyframes[i][1, 3], *self.cfg.stabilization.y_clip)
        for kf_id, kf in self.map.keyframes.items():
            kf.pose[1, 3] = np.clip(kf.pose[1, 3], *self.cfg.stabilization.y_clip)

        logger.info("Global BA Complete.")

    def optimize_pose_graph(self):
        if len(self.keyframes) < 2:
            return

        optimizer = self._build_g2o_optimizer()

        # 1. 노드 추가 (SE3 vertex)
        for i, pose in enumerate(self.keyframes):
            v = g2o.VertexSE3()
            v.set_id(i)
            v.set_estimate(g2o.Isometry3d(pose))
            if i == 0:
                v.set_fixed(True)  # 첫 번째 키프레임 고정
            optimizer.add_vertex(v)

        # 2. 엣지 추가
        for edge_info in self.g2o_edges:
            e = g2o.EdgeSE3()
            e.set_vertex(0, optimizer.vertex(edge_info['from']))
            e.set_vertex(1, optimizer.vertex(edge_info['to']))
            
            transform = edge_info['transform'].copy()
            
            if edge_info['type'] == 'loop':
                # PnP 기반 루프 엣지: PnP가 돌려준 스케일(거리) 정보를 그대로 보존해야
                # g2o가 전체 Odometry 궤적의 스케일 드리프트를 펴줄 수 있습니다.
                
                # 비등방 Information Matrix:
                # - 회전(0-2): 높은 신뢰도 (방향 보정에 탁월)
                # - 병진(3-5): PnP inlier 품질 향상(15개 이상)에 따라 병진 신뢰도도 어느정도 반영
                info = np.eye(6)
                info[0, 0] = info[1, 1] = info[2, 2] = self.cfg.pose_graph.rotation_weight
                info[3, 3] = info[4, 4] = info[5, 5] = self.cfg.pose_graph.translation_weight
            else:
                # 오도메트리 엣지: 등방 Information Matrix
                info = np.eye(6) * edge_info['information_scale']
            
            e.set_measurement(g2o.Isometry3d(transform))
            e.set_information(info)
            optimizer.add_edge(e)

        # 3. 최적화 실행
        optimizer.initialize_optimization()
        optimizer.optimize(self.cfg.pose_graph.iterations)

        # 4. 결과 추출 → 키프레임 포즈 갱신
        for i in range(len(self.keyframes)):
            v = optimizer.vertex(i)
            if v is not None:
                self.keyframes[i] = v.estimate().matrix().copy()
        
        self.cur_pose = self.keyframes[-1].copy()

    def print_perf_summary(self):
        if not self.total_ms_list:
            return
        avg_total = float(np.mean(self.total_ms_list))
        avg_sp = float(np.mean(self.sp_ms_list))
        avg_match = float(np.mean(self.match_ms_list))
        avg_clahe = float(np.mean(self.clahe_ms_list))
        avg_flow = float(np.mean(self.flow_ms_list))
        avg_filter = float(np.mean(self.filter_ms_list))
        avg_local_map = float(np.mean(self.local_map_ms_list))
        avg_pnp = float(np.mean(self.pnp_ms_list))
        avg_triangulation = float(np.mean(self.triangulation_ms_list))
        avg_vis = float(np.mean(self.visualization_ms_list))
        fps = 1000.0 / max(avg_total, 1e-6)

        accounted = avg_sp + avg_match + avg_clahe + avg_flow + avg_filter + avg_local_map + avg_pnp + avg_triangulation + avg_vis
        other = avg_total - accounted

        logger.info("Performance Summary")
        logger.info("   Avg total:         %.2f ms  (FPS: %.2f)", avg_total, fps)
        logger.info("   Avg SuperPoint:    %.2f ms (%.1f%%)", avg_sp, avg_sp/avg_total*100)
        logger.info("   Avg Match:         %.2f ms (%.1f%%)", avg_match, avg_match/avg_total*100)
        logger.info("   Avg CLAHE:         %.2f ms (%.1f%%)", avg_clahe, avg_clahe/avg_total*100)
        logger.info("   Avg Optical Flow:  %.2f ms (%.1f%%)", avg_flow, avg_flow/avg_total*100)
        logger.info("   Avg Filter:        %.2f ms (%.1f%%)", avg_filter, avg_filter/avg_total*100)
        logger.info("   Avg Local Map:     %.2f ms (%.1f%%)", avg_local_map, avg_local_map/avg_total*100)
        logger.info("   Avg PnP:           %.2f ms (%.1f%%)", avg_pnp, avg_pnp/avg_total*100)
        logger.info("   Avg Triangulation: %.2f ms (%.1f%%)", avg_triangulation, avg_triangulation/avg_total*100)
        logger.info("   Avg Visualization: %.2f ms (%.1f%%)", avg_vis, avg_vis/avg_total*100)
        logger.info("   Other/Overhead:    %.2f ms (%.1f%%)", other, other/avg_total*100)

        if self.jetson_scale is not None:
            jetson_fps = fps * self.jetson_scale
            logger.info("Jetson Nano est. FPS: %.2f (scale=%s)", jetson_fps, self.jetson_scale)
        
        # ── 통계 JSON 저장 ──
        import json
        
        # 모델 메모리 측정
        model_params = sum(p.numel() for p in self.fe.net.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.fe.net.parameters()) / (1024 * 1024)
        
        # 맵 포인트 수
        total_map_pts = sum(len(pts) for pts in self.all_map_points) if self.all_map_points else 0
        
        # Inlier ratio 통계
        valid_ratios = [r for r in self.inlier_ratio_list if r > 0]
        
        stats = {
            "latency": {
                "avg_total_ms": round(avg_total, 2),
                "avg_sp_ms": round(avg_sp, 2),
                "avg_match_ms": round(avg_match, 2),
                "avg_clahe_ms": round(avg_clahe, 2),
                "avg_flow_ms": round(avg_flow, 2),
                "avg_filter_ms": round(avg_filter, 2),
                "avg_local_map_ms": round(avg_local_map, 2),
                "avg_pnp_ms": round(avg_pnp, 2),
                "avg_triangulation_ms": round(avg_triangulation, 2),
                "avg_visualization_ms": round(avg_vis, 2),
                "fps": round(fps, 2),
                "total_frames": len(self.total_ms_list),
            },
            "memory": {
                "model_params_M": round(model_params / 1e6, 2),
                "model_size_MB": round(model_size_mb, 2),
                "map_points": total_map_pts,
                "keyframes": len(self.keyframes),
            },
            "inlier_stats": {
                "mean_ratio": round(float(np.mean(valid_ratios)), 3) if valid_ratios else 0,
                "median_ratio": round(float(np.median(valid_ratios)), 3) if valid_ratios else 0,
                "min_ratio": round(float(np.min(valid_ratios)), 3) if valid_ratios else 0,
                "max_ratio": round(float(np.max(valid_ratios)), 3) if valid_ratios else 0,
            },
            "keypoint_stats": {
                "mean_kpts": round(float(np.mean(self.kpts_list)), 0) if self.kpts_list else 0,
                "min_kpts": int(np.min(self.kpts_list)) if self.kpts_list else 0,
                "max_kpts": int(np.max(self.kpts_list)) if self.kpts_list else 0,
            },
            "ba_stats": self.ba_result.to_dict() if self.ba_result else {},
        }
        
        stats_path = os.path.join(self.save_dir, "slam_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 3D 궤적 저장 (ATE 평가용)
        if self.traj_points:
            traj_3d = np.array(self.traj_points)
            np.savetxt(os.path.join(self.save_dir, "trajectory_xyz.txt"), traj_3d, fmt="%.4f")


    def visualize_final_result(self):
        # 1. 최적화된 Pose Graph 및 MapPoint로 포인트 클라우드 재구축
        # Global BA 실행 후 각 MapPoint.pos3d는 이미 최적화된 3D 좌표를 저장하고 있습니다.
        rebuilt_points = []
        rebuilt_colors = []

        active_mps = list(self.map.map_points.values())
        logger.info("Rendering %d active MapPoints from Map (BA optimized)...", len(active_mps))
        
        if len(active_mps) > 0:
            world_pts = np.array([mp.pos3d for mp in active_mps])
            cols = get_height_color(world_pts[:, 1])
            rebuilt_points.append(world_pts)
            rebuilt_colors.append(cols)
        
        if not rebuilt_points:
            # Fallback: 재투영 데이터가 없으면 원본 사용
            if not self.all_map_points:
                logger.warning("No points generated.")
                return
            points = np.vstack(self.all_map_points)
            colors = np.vstack(self.all_map_colors)
        else:
            points = np.vstack(rebuilt_points)
            colors = np.vstack(rebuilt_colors)
            logger.info("Rendering %d points from Map across %d keyframes.", len(points), len(self.keyframes))
        
        # 2. 포인트 클라우드 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # [중요] 점 크기 조절 안됨 -> Voxel Downsample로 밀도 조절
        # 너무 촘촘하면 보기 싫고, 너무 듬성하면 휑함. 적당히 0.1m 간격으로 정리
        pcd = pcd.voxel_down_sample(voxel_size=self.cfg.viz.voxel_size)

        # 3. 경로선 (Trajectory Line) - 최적화된 Keyframe들을 연결
        traj_pts = np.array([pose[:3, 3] for pose in self.keyframes])
        # Y축 평탄화 (GBA로 인한 Y진동 제거 → 깨끗한 3D 경로)
        traj_pts[:, 1] = 0.0
        if len(traj_pts) < 2:
            return

        lines = [[i, i+1] for i in range(len(traj_pts)-1)]
        traj_line = o3d.geometry.LineSet()
        traj_line.points = o3d.utility.Vector3dVector(traj_pts)
        traj_line.lines = o3d.utility.Vector2iVector(lines)
        traj_line.paint_uniform_color([0, 1, 0]) # 녹색 선 (영상 스타일)

        # 4. 키프레임 카메라 (Camera Frustums) - 영상의 파란 삼각형들
        vis_geoms = [pcd, traj_line]
        logger.info("Generating %d Keyframes...", len(self.keyframes))
        
        for pose in self.keyframes:
            # 피라미드 생성
            frustum = create_camera_frustum(scale=0.5, color=[0, 0, 1]) # 파란색
            # 카메라 포즈 적용
            frustum.transform(pose)
            vis_geoms.append(frustum)

        # 5. 최종 뷰어 실행
        logger.info("Visualization Ready!")
        vis = o3d.visualization.Visualizer()
        vis.create_window("SuperPoint SLAM Result", width=1280, height=720)
        
        # 배경색: 영상처럼 검은색(Dark)이 포인트가 제일 잘 보임
        vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.05])
        vis.get_render_option().point_size = self.cfg.viz.point_size
        
        for geom in vis_geoms:
            vis.add_geometry(geom)
            
        # 초기 시점: 위에서 비스듬히 (쿼터뷰)
        ctr = vis.get_view_control()
        ctr.set_lookat(traj_pts[len(traj_pts)//2]) # 경로 중간을 바라봄
        ctr.set_front([-0.5, -1.0, -0.5]) 
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.5)

        # 저장
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "final_slam_map.ply"), pcd)
        logger.info("Point Cloud saved to: %s", os.path.join(self.save_dir, "final_slam_map.ply"))

        # 2D 평면 지도(Top-Down Map) 생성 및 저장
        try:
            from .slam2d_common import render_topdown_map
            traj_2d = traj_pts[:, [0, 2]]  # 3D (X,Y,Z) -> 2D (X,Z) Top-Down
            
            # 3D 맵 포인트를 2D 평면 지도용으로 X, Z축만 추출
            map_3d = np.asarray(pcd.points)
            if len(map_3d) > 0:
                map_2d = map_3d[:, [0, 2]]
            else:
                map_2d = np.empty((0, 2))
                
            map_img = render_topdown_map(traj_2d, map_2d)
            cv2.imwrite(os.path.join(self.save_dir, "topdown_map.png"), map_img)
            np.savetxt(os.path.join(self.save_dir, "trajectory_xy.txt"), traj_2d, fmt="%.4f")
            logger.info("Topdown Map saved to: %s", os.path.join(self.save_dir, "topdown_map.png"))
        except Exception as e:
            logger.error("Could not save Top-down map: %s", e)

        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--config', type=str, default=None, help="YAML config path")
    parser.add_argument('--highway-mode', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable DEBUG logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = SLAMConfig.from_yaml(args.config) if args.config else SLAMConfig()
    slam = VisualSLAM3D(
        weights_path=args.weights,
        input_path=args.input,
        config=cfg,
        highway_mode=args.highway_mode,
    )
    slam.process()