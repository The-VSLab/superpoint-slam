import argparse
import logging
import numpy as np
import cv2
import os
import time

try:
    import g2o
    HAS_G2O = True
except ImportError:
    g2o = None
    HAS_G2O = False

logger = logging.getLogger(__name__)

# 기존 모듈
from frontend.superpoint_frontend import SuperPointFrontend
from matcher_module import BTMatcher
from slam.loop_closure import LoopClosureManager
from tracking.point_filter import PointFilter
from tracking.semantic_filter import SemanticFilter
from slam.map_elements import Map, KeyFrame
from slam.bundle_adjustment import run_bundle_adjustment
from slam.slam_utils import get_optimal_device, get_height_color, desc_to_numpy
from slam.slam_metrics import SLAMMetrics
from slam.slam_visualizer import SLAMVisualizer
from slam.pose_stabilizer import PoseStabilizer
from slam.pose_estimation import PoseEstimator
from slam.feature_extraction import FeatureExtractor
from slam.optical_flow import OpticalFlowTracker
from slam.triangulator import Triangulator
from slam.local_map_tracker import LocalMapTracker
from slam.keyframe_manager import KeyframeManager
from config.slam_config import SLAMConfig

class VisualSLAM3D:
    def __init__(
        self,
        weights_path,
        input_path,
        config: SLAMConfig = None,
        output_dir="path_final",
        resize=None,
        calib_path=None,
        highway_mode=False,
    ):
        # --- 설정 객체 ---
        self.cfg = config or SLAMConfig()
        c = self.cfg  # 축약 참조

        # 1. 장치 결정
        self.highway_mode = highway_mode
        self.device = get_optimal_device()
        self.use_cuda = (self.device == "cuda")

        self.enable_viz = c.viz.enabled
        self.use_clahe = c.clahe.enabled

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
        if calib_path is not None:
            self.K = self._load_kitti_calib(calib_path, orig_w, orig_h, self.W, self.H)
        else:
            focal = max(self.W, self.H) * c.camera.focal_multiplier
            self.K = np.array([[focal, 0, self.W / 2.0],
                                [0, focal, self.H / 2.0],
                                [0,     0,           1.0]])
        self.focal = self.K[0, 0]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

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

        # Semantic Filter (FP16 + sp_interval 주기 실행)
        semantic_filter = None
        if c.semantic.enabled:
            semantic_filter = SemanticFilter(conf_thresh=c.semantic.yolo_conf, half=c.semantic.half)
        self.matcher = BTMatcher(
            nn_thresh=c.superpoint.nn_thresh,
            use_cuda=self.use_cuda,
            mutual=True,  # 반드시 True로 강제 (또는 c.matcher.mutual 확인)
            ratio_thresh=0.85 # Lowe's Ratio test 강화
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
        self.last_keyframe_idx = 0

        # 포즈 안정화 + 추정 모듈
        self.stabilizer = PoseStabilizer(c.stabilization)
        self.pose_estimator = PoseEstimator(self.K, c.pnp, c.motion, c.epipolar)

        # 특징점 추출 + Optical Flow 모듈
        self.extractor = FeatureExtractor(
            fe=self.fe, point_filter=self.point_filter,
            semantic_filter=semantic_filter,
            sp_cfg=c.superpoint, point_filter_cfg=c.point_filter,
            use_cuda=self.use_cuda, use_semantic=c.semantic.enabled,
            filter_on_sp_only=c.performance.filter_on_sp_only, W=self.W, H=self.H,
        )
        self.flow_tracker = OpticalFlowTracker(c.optical_flow)
        self.triangulator = Triangulator(self.K, c.triangulation)
        self.local_map_tracker = LocalMapTracker(self.K, self.matcher, c.performance)
        self.kf_manager = KeyframeManager(c.keyframe)
        
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

        # 시각화 + 메트릭 모듈
        self.visualizer = SLAMVisualizer(self.cfg.viz, self.save_dir, self.W, self.H)
        self.enable_viz = self.visualizer.enabled
        self.metrics = SLAMMetrics(self.save_dir, self.cfg.logging)

    @staticmethod
    def _load_kitti_calib(calib_path: str, orig_w: int, orig_h: int, new_w: int, new_h: int) -> np.ndarray:
        """calib.txt에서 P0 내부 파라미터를 읽어 리사이즈 비율로 스케일링한 K를 반환."""
        with open(calib_path) as f:
            for line in f:
                if line.startswith("P0:"):
                    vals = list(map(float, line.strip().split()[1:]))
                    P = np.array(vals).reshape(3, 4)
                    K = P[:3, :3].copy()
                    K[0, 0] *= new_w / orig_w  # fx
                    K[1, 1] *= new_h / orig_h  # fy
                    K[0, 2] *= new_w / orig_w  # cx
                    K[1, 2] *= new_h / orig_h  # cy
                    logger.info(
                        "calib.txt loaded: fx=%.2f fy=%.2f cx=%.2f cy=%.2f (scaled %dx%d → %dx%d)",
                        K[0, 0], K[1, 1], K[0, 2], K[1, 2], orig_w, orig_h, new_w, new_h,
                    )
                    return K
        raise ValueError(f"P0 not found in {calib_path}")

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

            # 특징점 추출 (SuperPoint + Semantic/Point 필터)
            feat = self.extractor.extract(
                img_masked, img_curr, img_curr, run_infer,
            )
            kpts = feat.kpts
            desc = feat.desc
            sp_ms = feat.sp_ms
            filter_ms = feat.filter_ms

            # Optical Flow 트래킹
            flow_t0 = time.perf_counter()
            flow_result = self.flow_tracker.track(
                self.prev_frame, img_gray, self.prev_kpts,
            ) if self.prev_frame is not None else None
            flow_p1 = flow_result.flow_p1 if flow_result else None
            flow_p2 = flow_result.flow_p2 if flow_result else None
            status = flow_result.status if flow_result else None
            flow_t1 = time.perf_counter()

            if not run_infer and flow_p2 is not None:
                kpts = flow_p2
                # OF 프레임: 캐시된 dyn_mask로 동적 객체 제거
                if self.extractor._cached_dyn_mask is not None and len(kpts) > 0:
                    kpts_int = kpts.astype(int)
                    kx = np.clip(kpts_int[:, 0], 0, self.W - 1)
                    ky = np.clip(kpts_int[:, 1], 0, self.H - 1)
                    valid_mask = ~self.extractor._cached_dyn_mask[ky, kx]
                    kpts = kpts[valid_mask]

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
            p1 = None
            p2 = None
            p2_idx = None
            mask = None

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
                
            # 로컬 맵 트래킹 (PnP 강건성 확보)
            local_map_t0 = time.perf_counter()
            frame_shape = self.prev_frame.shape if self.prev_frame is not None else (self.H, self.W)
            lm_result = self.local_map_tracker.track(
                self.map, self.cur_pose, kpts, desc,
                self.curr_kpt_to_mp, frame_shape, frame_idx,
            )
            local_map_t1 = time.perf_counter()

            # --- 1. PnP 시도 (충분한 3D 점이 있을 때) ---
            pnp_t0 = time.perf_counter()
            obj_pts_list = []
            img_pts_list = []
            
            # 기존 P1-P2 추적분 추가
            if p2_idx is not None and len(p2_idx) > 0:
                valid_3d_mask = ~np.isnan(curr_3d_pts_world[p2_idx, 0])
                if np.any(valid_3d_mask):
                    obj_pts_list.append(curr_3d_pts_world[p2_idx][valid_3d_mask].astype(np.float32))
                    img_pts_list.append(p2[valid_3d_mask].astype(np.float32))
            
            # 신규 로컬 맵 매칭분 추가
            if lm_result.count > 0:
                lm_obj, lm_img = lm_result.as_arrays()
                obj_pts_list.append(lm_obj)
                img_pts_list.append(lm_img)

                # 매칭된 MapPoint 연결 끊어지지 않게 저장
                for i, kpt_idx in enumerate(lm_result.extra_kpt_idxs):
                    self.curr_kpt_to_mp[kpt_idx] = lm_result.extra_mp_ids[i]
                    curr_3d_pts_world[kpt_idx] = lm_result.extra_mp_ids[i].pos3d
            
            # 1. PnP 시도 (충분한 3D 점이 있을 때)
            pose_result = None
            if len(obj_pts_list) > 0:
                obj_pts = np.vstack(obj_pts_list)
                img_pts = np.vstack(img_pts_list)
                pose_result = self.pose_estimator.estimate_pnp(
                    obj_pts, img_pts, self.cur_pose, match_count, p1,
                    self.stabilizer.recent_speeds, self.stabilizer.last_t_vec, frame_idx,
                )

            # 2. PnP 실패 시 Essential Matrix 또는 Homography 폴백
            if (pose_result is None or not pose_result.success) and p1 is not None and p2 is not None and len(p1) >= 8:
                pose_result = self.pose_estimator.estimate_epipolar(
                    p1, p2, match_count,
                    self.stabilizer.recent_speeds, self.stabilizer.last_t_vec, frame_idx,
                )

            pnp_success = pose_result is not None and pose_result.success
            if pnp_success:
                R = pose_result.R
                t_vec = pose_result.t_vec
                inliers = pose_result.inliers
                inlier_ratio = pose_result.inlier_ratio
                mask = pose_result.mask
                method = pose_result.method
            else:
                mask = None
                inliers = 0
                inlier_ratio = 0.0
                method = "DR"
            pnp_t1 = time.perf_counter()

            # 3. 최적 포즈 적용 및 지도 업데이트
            triangulation_t0 = time.perf_counter()
            if pnp_success:
                # 포즈 안정화 (Y댐핑, Pitch/Roll 억제, 관성 혼합)
                R_damped, t_vec, R_orig, t_orig = self.stabilizer.stabilize(
                    R, t_vec, self.highway_mode
                )

                # Pose Update (Relative to absolute)
                T_rel = np.eye(4)
                T_rel[:3, :3] = R_damped
                T_rel[:3, 3] = t_vec
                self.cur_pose = self.cur_pose @ T_rel

                # 절대 Y축 높이 강제 고정 (Hard Clip)
                self.cur_pose[1, 3] = np.clip(self.cur_pose[1, 3], *self.cfg.stabilization.y_clip)

                valid_step = True

                # 삼각측량 + 필터링 + 월드 변환
                tri_result = self.triangulator.triangulate_and_filter(
                    R_orig, t_orig, p1, p2, p2_idx, mask,
                    self.cur_pose, curr_3d_pts_world,
                )

                if tri_result.count > 0:
                    # MapPoint 객체 등록
                    for i, pt3d in enumerate(tri_result.world_pts):
                        kpt_idx = tri_result.valid_indices[i]
                        if kpt_idx not in self.curr_kpt_to_mp:
                            if not use_flow and desc is not None and kpt_idx < desc.shape[1]:
                                desc_col = desc[:, kpt_idx]
                                desc_i = desc_to_numpy(desc_col)
                            else:
                                desc_i = None
                            mp = self.map.create_map_point(pt3d, desc_i)
                            if self.last_keyframe_idx >= 0:
                                mp.add_observation(self.last_keyframe_idx, tri_result.p1_filtered[i])
                            mp.add_observation(frame_idx, tri_result.p2_filtered[i])
                            self.curr_kpt_to_mp[kpt_idx] = mp

                    # 로컬 3D 포인트를 현재 키프레임에 등록
                    if len(self.keyframe_local_pts) > 0:
                        self.keyframe_local_pts[-1].append(tri_result.local_pts.copy())

                    # 시각화용 월드 좌표 저장
                    map_points_added = tri_result.count
                    cols = get_height_color(tri_result.world_pts[:, 1])
                    self.all_map_points.append(tri_result.world_pts)
                    self.all_map_colors.append(cols)
            triangulation_t1 = time.perf_counter()

            # 4. 추적 결과에 따른 포즈 확정 및 관성 주행 처리
            if not valid_step:
                # [Fix 1] Dead Reckoning 속도 감쇠 — 연속 실패 시 점진적 감속
                T_rel = self.stabilizer.dead_reckon()
                self.cur_pose = self.cur_pose @ T_rel

            # [Fix 5] 궤적 정체 감지 — 10프레임 동안 0.5m 미만 이동 시 강제 키프레임 재추출
            if len(self.traj_points) >= self.cfg.keyframe.stall_lookback:
                recent_dist = np.linalg.norm(
                    np.array(self.cur_pose[:3, 3]) - np.array(self.traj_points[-self.cfg.keyframe.stall_lookback])
                )
                if recent_dist < self.cfg.keyframe.stall_distance:
                    self.force_keyframe = True
                    self.stabilizer.reset_inertia()
                    logger.warning("frame %d: [STALL DETECTED] Forcing keyframe re-extraction", frame_idx)

            # 궤적 저장
            curr_t = self.cur_pose[:3, 3]
            self.traj_points.append(curr_t)            
            # --- [Keyframe 관리] ---
            need_keyframe = self.kf_manager.should_create_keyframe(
                p1, p2, inliers, self.metrics.inliers_list,
                frame_idx, self.last_keyframe_idx,
            )

            if need_keyframe:
                if desc is None:
                    # Descriptors가 없으면 다음 프레임에서 강제로 뽑도록 설정
                    self.force_keyframe = True
                else:
                    # Optical Flow를 사용했다면 curr_kpt_to_mp의 인덱스는 flow_p2 기준입니다.
                    # 이를 새로 추출된 kpts 인덱스에 매핑(Alignment)해야 KeyFrame 생성 시 에러가 나지 않습니다.
                    if use_flow and len(kpts) > 0:
                        from scipy.spatial import cKDTree
                        aligned_kpt_to_mp = {}
                        # O(N*log(N)) KDTree 기반 최근접 이웃 탐색 (기존 O(M*N) 루프 대체)
                        tree = cKDTree(kpts[:, :2])
                        valid_items = [(f_idx, mp) for f_idx, mp in self.curr_kpt_to_mp.items() if f_idx < len(flow_p2)]
                        if valid_items:
                            query_pts = np.array([flow_p2[f_idx] for f_idx, _ in valid_items])
                            dists, indices = tree.query(query_pts, k=1)
                            alignment_dist = self.cfg.keyframe.alignment_dist
                            desc_np = desc_to_numpy(desc)
                            for i, (f_idx, mp) in enumerate(valid_items):
                                if dists[i] < alignment_dist:
                                    best_idx = indices[i]
                                    aligned_kpt_to_mp[best_idx] = mp
                                    if desc_np is not None and best_idx < desc_np.shape[1]:
                                        mp.update_descriptor(desc_np[:, best_idx])
                        self.curr_kpt_to_mp = aligned_kpt_to_mp

                    self.add_keyframe(frame_idx, kpts, desc, curr_3d_pts_world, T_rel.copy())
                    self.last_keyframe_idx = frame_idx
                    self.kf_manager.reset_parallax()
                    
                    # [NEW] KeyFrame 생성 시, 현재 연결된 MapPoint들에게 관측 기록을 남겨 Culling에서 살아남게 함
                    if self.last_keyframe_idx >= 0:
                        for kpt_idx, mp in self.curr_kpt_to_mp.items():
                            if kpt_idx < len(kpts):
                                mp.add_observation(self.last_keyframe_idx, kpts[kpt_idx][:2])
                    
                    # 새 키프레임 생성 후 불량 맵 포인트 정리 (Culling)
                    # 키프레임 간 간격을 유지하면서 불필요한 관측 횟수 2 이하의 특징점 메모리 해제
                    if len(self.keyframes) % self.cfg.keyframe.culling_interval == 0:
                        self.map.cull_bad_map_points(min_observations=self.cfg.map_culling.min_observations)

            # 2D 뷰 표시 - SLAMVisualizer로 위임
            vis_ms = self.visualizer.show_live(img_curr, kpts, frame_idx)
            if vis_ms < 0:  # quit signal
                break

            # 프레임 끝 시간
            frame_t1 = time.perf_counter()

            # ms 계산 + append
            # sp_ms, filter_ms는 extractor에서 이미 계산됨
            match_ms = (match_t1 - match_t0) * 1000.0
            total_ms = (frame_t1 - frame_t0) * 1000.0
            clahe_ms = (clahe_t1 - clahe_t0) * 1000.0
            flow_ms = (flow_t1 - flow_t0) * 1000.0
            local_map_ms = (local_map_t1 - local_map_t0) * 1000.0
            pnp_ms = (pnp_t1 - pnp_t0) * 1000.0
            triangulation_ms = (triangulation_t1 - triangulation_t0) * 1000.0

            self.metrics.record_frame(
                frame_idx,
                timings={
                    "sp_ms": sp_ms, "match_ms": match_ms, "clahe_ms": clahe_ms,
                    "flow_ms": flow_ms, "filter_ms": filter_ms, "local_map_ms": local_map_ms,
                    "pnp_ms": pnp_ms, "triangulation_ms": triangulation_ms,
                    "vis_ms": vis_ms, "total_ms": total_ms,
                },
                counts={
                    "kpts": int(len(kpts)), "matches": int(len(matches)),
                    "inliers": int(inliers), "inlier_ratio": float(inlier_ratio),
                    "map_points_added": int(map_points_added),
                },
                method="PnP" if pnp_success else "DR",
            )

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
        self.visualizer.close()

        # CSV 로깅 종료
        self.metrics.close()

        # Pose Graph Optimization만 실행 (GBA는 단안 스케일 드리프트로 비활성화)
        self.optimize_pose_graph()

        # PGO 후 Y축 높이 클리핑 (평지 주행 가정)
        for i in range(len(self.keyframes)):
            self.keyframes[i][1, 3] = np.clip(self.keyframes[i][1, 3], *self.cfg.stabilization.y_clip)
        for kf_id, kf in self.map.keyframes.items():
            kf.pose[1, 3] = np.clip(kf.pose[1, 3], *self.cfg.stabilization.y_clip)

        self.metrics.print_summary(
            self.fe.net, self.all_map_points, self.keyframes,
            self.traj_points, self.jetson_scale,
        )
        self.visualizer.render_final(
            self.map, self.keyframes, self.all_map_points, self.all_map_colors,
        )

    def add_keyframe(self, frame_idx, kpts, desc, pts_3d=None, T_rel=None):
        self.keyframes.append(self.cur_pose.copy())
        self.keyframe_indices.append(frame_idx)
        self.keyframe_local_pts.append([])  # 이 키프레임에 등록될 로컬 3D 포인트들
        self.keyframe_original_poses.append(self.cur_pose.copy())

        # GPU 텐서 → numpy 변환 (KeyFrame/LoopClosure 저장 경계)
        desc_np = desc_to_numpy(desc)

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
        if not HAS_G2O:
            logger.warning("g2o not available - pose graph optimization skipped")
            return None
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)
        return optimizer

    def _run_ba(self, kf_ids, num_iterations=10, fix_first=True, two_pass=None):
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
            two_pass=two_pass,
        )
        if result is not None:
            self.metrics.ba_result = result
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
        self._run_ba(all_kf_ids, num_iterations=self.cfg.ba.global_iterations, fix_first=True, two_pass=False)

        # GBA 후 전체 키프레임 Y축 높이 강제 클리핑 (평지 주행 가정)
        for i in range(len(self.keyframes)):
            self.keyframes[i][1, 3] = np.clip(self.keyframes[i][1, 3], *self.cfg.stabilization.y_clip)
        for kf_id, kf in self.map.keyframes.items():
            kf.pose[1, 3] = np.clip(kf.pose[1, 3], *self.cfg.stabilization.y_clip)

        logger.info("Global BA Complete.")

    def optimize_pose_graph(self):
        if len(self.keyframes) < 2:
            return
        if not HAS_G2O:
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