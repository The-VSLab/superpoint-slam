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


# ---------------------------------------------------------------------------
# Sim3 Pose-Graph 변환 유틸 (규약: 앱 포즈 = T_wc(camera→world),
#   g2o VertexSim3Expmap = Scw(world→camera), 스케일 s는 Sim3에 내장)
#   합성 루프로 검증 완료: 스케일 드리프트 제거 + 좌표 왕복변환 일치.
# ---------------------------------------------------------------------------
def _twc_to_sim3(T_wc, scale=1.0):
    """camera→world 4x4 → g2o.Sim3 (world→camera, Scw)."""
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return g2o.Sim3(R_cw, t_cw, float(scale))


def _sim3_to_twc(S):
    """g2o.Sim3 (world→camera, scale s) → camera→world 4x4 (metric)."""
    R_cw = S.rotation().rotation_matrix()
    t_cw = np.asarray(S.translation())
    s = S.scale()
    R_wc = R_cw.T
    t_wc = -(1.0 / s) * R_wc @ t_cw
    T = np.eye(4)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc
    return T


def _rel_sim3_measurement(T_wc_from, T_rel, scale=1.0):
    """EdgeSim3 측정치 = S_to * S_from^{-1}  (T_to = T_from @ T_rel).
    scale=1일 때 앵커(T_wc_from)와 무관하게 inv(T_rel)과 동치임을 검증함."""
    T_wc_to = T_wc_from @ T_rel
    return _twc_to_sim3(T_wc_to, scale) * _twc_to_sim3(T_wc_from, 1.0).inverse()

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

        # 매 프레임 np.full() 할당 제거용 사전 할당 풀
        _max_pool = int(c.superpoint.max_kpts * 2 + 64)
        self._3d_pool = np.full((_max_pool, 3), np.nan)
        self._prev_3d_pool = np.full((_max_pool, 3), np.nan)

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
            else:
                mask = None
                inliers = 0
                inlier_ratio = 0.0
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
                    # flow 프레임에서도 디스크립터를 부여하기 위해, 새 MapPoint의 2D 위치와
                    # 가장 가까운 SP 키포인트(2px 이내)의 디스크립터를 샘플링한다.
                    # (기존: use_flow면 descriptor=None → LocalMapTracker 매칭 전멸)
                    _sp_tree = None
                    if use_flow and desc is not None and len(kpts) > 0:
                        from scipy.spatial import cKDTree as _cKDTree
                        _sp_tree = _cKDTree(kpts[:, :2])

                    # MapPoint 객체 등록
                    for i, pt3d in enumerate(tri_result.world_pts):
                        kpt_idx = tri_result.valid_indices[i]
                        if kpt_idx not in self.curr_kpt_to_mp:
                            desc_i = None
                            if not use_flow and desc is not None and kpt_idx < desc.shape[1]:
                                desc_i = desc_to_numpy(desc[:, kpt_idx])
                            elif _sp_tree is not None:
                                _d, _j = _sp_tree.query(tri_result.p2_filtered[i], k=1)
                                if _d < 2.0 and _j < desc.shape[1]:
                                    desc_i = desc_to_numpy(desc[:, _j])
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
                            for i, (_, mp) in enumerate(valid_items):
                                if dists[i] < alignment_dist:
                                    best_idx = indices[i]
                                    aligned_kpt_to_mp[best_idx] = mp
                                    if desc_np is not None and best_idx < desc_np.shape[1]:
                                        mp.update_descriptor(desc_np[:, best_idx])
                        self.curr_kpt_to_mp = aligned_kpt_to_mp

                    # 루프 클로저용 클린 kpt↔3D 쌍 구성.
                    # 소스: 트래킹 PnP가 실제 사용하는 (curr_3d_pts_world[p2_idx] ↔ p2) 쌍
                    #       — 자체 PnP 검증에서 정합 확인된 유일한 클린 소스.
                    # flow 위치 → SP 키포인트 연결은 인덱스 북키핑(dict, 오염됨)을 쓰지 않고
                    # 순수 기하 최근접(2px)으로 수행한다.
                    self._lc_pts3d_sp = None
                    if p2_idx is not None and p2 is not None and len(kpts) > 0:
                        _v = ~np.isnan(curr_3d_pts_world[p2_idx, 0])
                        if _v.sum() >= 4:
                            from scipy.spatial import cKDTree
                            _obj = curr_3d_pts_world[p2_idx][_v]
                            _pos2d = p2[_v]
                            _dists, _sp_idx = cKDTree(kpts[:, :2]).query(_pos2d, k=1)
                            _ok = _dists < 2.0
                            if np.count_nonzero(_ok) >= 4:
                                self._lc_pts3d_sp = np.full((len(kpts), 3), np.nan)
                                self._lc_pts3d_sp[_sp_idx[_ok]] = _obj[_ok]

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
        for _, kf in self.map.keyframes.items():
            kf.pose[1, 3] = np.clip(kf.pose[1, 3], *self.cfg.stabilization.y_clip)

        # 최종(post-PGO) 키프레임 3D 궤적 저장 — GT(ATE) 평가용 (frame_idx, x, y, z)
        if self.keyframes:
            kf_traj = np.array([
                [idx, pose[0, 3], pose[1, 3], pose[2, 3]]
                for idx, pose in zip(self.keyframe_indices, self.keyframes)
            ])
            np.savetxt(os.path.join(self.metrics.save_dir, "trajectory_kf.txt"), kf_traj, fmt="%.4f")

        loop_stats = {**self.loop_closure.stats, 'thresh': self.loop_closure.descriptor_similarity}
        self.metrics.print_summary(
            self.fe.net, self.all_map_points, self.keyframes,
            self.traj_points, self.jetson_scale, loop_stats=loop_stats,
        )
        self.visualizer.render_final(
            self.map, self.keyframes, self.all_map_points, self.all_map_colors,
        )

    def add_keyframe(self, frame_idx, kpts, desc, _pts_3d=None, T_rel=None):
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
            # 주의: 인자 T_rel은 '마지막 한 프레임'의 상대이동이라 키프레임 간격(~5프레임)보다
            # 훨씬 짧다. odom 엣지는 반드시 키프레임 포즈 간 합성 상대변환이어야
            # PGO가 궤적을 짧은 체인으로 수축시키지 않는다.
            T_rel_kf = np.linalg.inv(self.keyframes[node_idx - 1]) @ self.keyframes[node_idx]
            self.g2o_edges.append({
                'type': 'odom',
                'from': node_idx - 1,
                'to': node_idx,
                'transform': T_rel_kf,
                'scale': 1.0,
                'information_scale': 1.0,
            })

        # 루프 클로저용 kpt↔3D 쌍 (SP 키포인트 인덱스 기준).
        # 우선 소스: process()에서 기하 매칭으로 만든 클린 배열(_lc_pts3d_sp).
        # (curr_kpt_to_mp dict는 flow/SP 인덱스 세대 혼입으로 오염 — PnP 검증 불가)
        lc_arr = getattr(self, "_lc_pts3d_sp", None)
        if lc_arr is not None and len(lc_arr) == len(kpts):
            pts_3d_for_lc = lc_arr.copy()
        else:
            pts_3d_for_lc = np.full((len(kpts), 3), np.nan)
            for kpt_idx, mp in self.curr_kpt_to_mp.items():
                if kpt_idx < len(kpts):
                    pts_3d_for_lc[kpt_idx] = mp.pos3d
        logger.debug("[KF %d] loop-closure 3D density: %d/%d kpts have MapPoint",
                     frame_idx, int(np.count_nonzero(~np.isnan(pts_3d_for_lc[:, 0]))), len(kpts))
        # 자가진단: 저장한 kpt↔3D 매핑을 자기 pose로 재투영 — 매핑이 옳다면 오차가 수 px 이내
        if logger.isEnabledFor(logging.DEBUG):
            def _self_reproj(tag, pts3d_arr):
                _valid = ~np.isnan(pts3d_arr[:, 0])
                if _valid.sum() < 10:
                    return
                _Tcw = np.linalg.inv(self.cur_pose)
                _P = _Tcw[:3, :3] @ pts3d_arr[_valid].T + _Tcw[:3, 3:4]
                _ok = _P[2] > 0.1
                if _ok.sum() < 10:
                    return
                _uv = self.K @ _P[:, _ok]
                _uv = (_uv[:2] / _uv[2]).T
                _err = np.linalg.norm(_uv - kpts[_valid][_ok][:, :2], axis=1)
                logger.debug("[KF %d] %s self-reproj err: median=%.1fpx p90=%.1fpx (n=%d)",
                             frame_idx, tag, np.median(_err), np.percentile(_err, 90), int(_ok.sum()))
            _self_reproj("dict(curr_kpt_to_mp)", pts_3d_for_lc)
            if _pts_3d is not None:
                _self_reproj("array(curr_3d_pts_world)", np.asarray(_pts_3d)[: len(kpts)])
            # 판별 실험: 저장 쌍만으로 PnP — 쌍이 정상이면 inlier 다수 (cur_pose와 무관)
            _valid = ~np.isnan(pts_3d_for_lc[:, 0])
            if _valid.sum() >= 15:
                _ok2, _rv, _tv, _inl2 = cv2.solvePnPRansac(
                    pts_3d_for_lc[_valid].astype(np.float32), kpts[_valid][:, :2].astype(np.float32),
                    self.K, None, iterationsCount=300, reprojectionError=8.0,
                    confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
                logger.debug("[KF %d] same-frame PnP on stored pairs: success=%s inliers=%s/%d",
                             frame_idx, _ok2, None if _inl2 is None else len(_inl2), int(_valid.sum()))
        # 후보 relativize를 위해 현재 키프레임의 절대 포즈(T_wc)도 함께 저장
        self.loop_closure.add_keyframe(frame_idx, kpts, desc_np, pts_3d_for_lc, self.cur_pose.copy())
        loop = self.loop_closure.find_loop(frame_idx, kpts, desc_np)
        if loop is not None:
            logger.info("[Pose Graph] Sim3 Loop Edge: method=%s RelDist=%.3f", loop.method, loop.scale)

            # 루프 클로저 엣지 저장 (transform은 후보→현재 상대 포즈)
            self.g2o_edges.append({
                'type': 'loop',
                'from': loop.match_index,
                'to': node_idx,
                'transform': loop.transform,
                'scale': loop.scale,
                'method': loop.method,     # 'pnp'(metric 병진) / 'ess'(회전만)
                'information_scale': 1.0,
            })
            self.optimize_pose_graph()

        # LBA는 실시간 추적 시 cur_pose를 오염시키므로 비활성화
        # GBA만 후처리로 실행합니다 (process() 루프 종료 후)

    def _build_g2o_optimizer(self):
        """g2o Sim3 Pose Graph 최적화기 구축 (7-DOF: 회전+병진+스케일)"""
        if not HAS_G2O:
            logger.warning("g2o not available - pose graph optimization skipped")
            return None
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSim3(g2o.LinearSolverDenseSim3())
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
        for _, kf in self.map.keyframes.items():
            kf.pose[1, 3] = np.clip(kf.pose[1, 3], *self.cfg.stabilization.y_clip)

        logger.info("Global BA Complete.")

    def optimize_pose_graph(self):
        if len(self.keyframes) < 2:
            return
        if not HAS_G2O:
            return
        # 루프 엣지가 없으면 odom 체인만으로는 교정할 정보가 없음 (그래프가 이미 정확히 만족됨)
        if not any(e['type'] == 'loop' for e in self.g2o_edges):
            logger.info("[Pose Graph] no loop edges - skipping PGO")
            return

        optimizer = self._build_g2o_optimizer()
        if optimizer is None:
            return

        pg = self.cfg.pose_graph
        rot_w = pg.rotation_weight
        trans_w = pg.translation_weight
        scale_w = pg.scale_weight
        ess_trans_w = pg.essential_translation_weight

        # 1. 노드 추가 (Sim3 vertex, 초기 스케일=1)
        n_kf = len(self.keyframes)
        for i, pose in enumerate(self.keyframes):
            v = g2o.VertexSim3Expmap()
            v.set_id(i)
            v.set_estimate(_twc_to_sim3(pose, 1.0))
            v.set_fixed(i == 0)   # 첫 키프레임 고정 (게이지 + 스케일 앵커)
            optimizer.add_vertex(v)

        # 2. 엣지 추가 (측정치 = 후보/이전(from)→현재(to) 상대 Sim3)
        #    Sim3 log 탄젠트 순서: [0:3]=회전, [3:6]=병진, [6]=스케일
        edge_count = 0
        for edge_info in self.g2o_edges:
            i_from = edge_info['from']
            i_to = edge_info['to']
            if not (0 <= i_from < n_kf and 0 <= i_to < n_kf):
                continue
            T_rel = edge_info['transform']
            meas = _rel_sim3_measurement(self.keyframes[i_from], T_rel, 1.0)

            info = np.eye(7)
            if edge_info['type'] == 'loop':
                info[0, 0] = info[1, 1] = info[2, 2] = rot_w      # 회전: 강하게 신뢰
                if edge_info.get('method') == 'ess':
                    # Essential 루프: recoverPose 병진은 '단위' 스케일 → 병진/스케일 거의 무시
                    tw = ess_trans_w
                    info[6, 6] = ess_trans_w
                else:
                    # PnP 루프: 병진이 metric → 스케일 드리프트 교정에 사용
                    tw = trans_w
                    info[6, 6] = scale_w
                info[3, 3] = info[4, 4] = info[5, 5] = tw
            else:
                info *= edge_info['information_scale']            # 오도메트리: 등방

            e = g2o.EdgeSim3()
            e.set_vertex(0, optimizer.vertex(i_from))
            e.set_vertex(1, optimizer.vertex(i_to))
            e.set_measurement(meas)
            e.set_information(info)
            e.set_id(edge_count)
            optimizer.add_edge(e)
            edge_count += 1

        # 3. 최적화 실행
        optimizer.initialize_optimization()
        optimizer.optimize(pg.iterations)

        # 4. 결과 추출 → 키프레임 포즈 갱신 (Sim3 → T_wc, metric)
        for i in range(n_kf):
            v = optimizer.vertex(i)
            if v is not None:
                self.keyframes[i] = _sim3_to_twc(v.estimate())

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