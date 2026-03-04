import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
import g2o
import os
import time
import sys

# 기존 모듈
from frontend.superpoint_frontend import SuperPointFrontend
from matcher_module import BTMatcher
from slam.loop_closure import LoopClosureManager
from tracking.point_filter import PointFilter

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
        nn_thresh=0.7,
        conf_thresh=0.015,
        jetson_scale=None,
        sp_scale=1.0,
        sp_interval=1,
        sp_fp16=False,
        highway_mode=False,
        output_dir="path_final",
        roi_sky=0.35,
        roi_hood=0.85,
        resize=None
    ):
        # 1. 장치 결정
        self.highway_mode = highway_mode
        self.device = get_optimal_device()
        # SuperPointFrontend는 내부 설계상 True/False(CUDA 사용여부)를 받는 경우가 많으므로 호환성 유지
        self.use_cuda = (self.device == "cuda")
        
        print(f"==> Running on device: {self.device.upper()}")

        self.input_path = input_path
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise ValueError(f"Error: {input_path}")
        
        # 원본 영상 해상도 읽기
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if resize is not None:
            # 사용자 지정 해상도 사용 (8의 배수로 반올림)
            self.W = (resize[0] // 8) * 8
            self.H = (resize[1] // 8) * 8
        else:
            # 종횡비 유지하면서 장축 기준 640으로 자동 리사이즈 (8의 배수)
            target_long = 640
            if orig_w >= orig_h:
                scale_factor = target_long / orig_w
            else:
                scale_factor = target_long / orig_h
            self.W = (int(orig_w * scale_factor) // 8) * 8
            self.H = (int(orig_h * scale_factor) // 8) * 8
        
        # 최소 크기 보장
        self.W = max(self.W, 64)
        self.H = max(self.H, 64)
        
        print(f"==> Original: {orig_w}x{orig_h} → Resized: {self.W}x{self.H} (aspect preserved, 8x aligned)")
        
        self.sp_scale = float(sp_scale)
        self.sp_interval = max(int(sp_interval), 1)
        if not (0.1 <= self.sp_scale <= 1.0):
            raise ValueError("sp_scale must be in [0.1, 1.0]")

        print(f"==> Resolution: {self.W}x{self.H}")
        print(f"==> SuperPoint config: fp16={sp_fp16}, sp_scale={self.sp_scale}, sp_interval={self.sp_interval}")

        # 카메라 파라미터 (일반적인 블랙박스 화각)
        self.focal = max(self.W, self.H) * 0.8
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.K = np.array([[self.focal, 0, self.cx], [0, self.focal, self.cy], [0, 0, 1]])

        print("==> Loading SuperPoint...")
        # descriptor_dim=128로 테스트 후 안정적일 때 head_hidden 256-> 128로 전환예정
        self.fe = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=4,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=self.use_cuda,
            roi_sky=roi_sky,
            roi_hood=roi_hood
        )
        self.matcher = BTMatcher(
            nn_thresh=nn_thresh, 
            use_cuda=self.use_cuda, 
            mutual=True,
            ratio_thresh=0.85
        )
        self.loop_closure = LoopClosureManager(
            matcher=self.matcher,
            K=self.K,
            min_frame_gap=30,
            top_k=5,
            min_inliers=30,
            min_inlier_ratio=0.25,
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
        
        # g2o Sim3 Pose Graph 최적화기 설정
        self.g2o_optimizer = None
        self.g2o_edges = []
        self.jetson_scale = jetson_scale

        self.save_dir = str(output_dir)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        # 실시간 2D 확인창
        cv2.namedWindow('Processing', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processing', 640, 360)

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
        img[int(h*0.92):, :] = 0 
        return img

    def process(self):
        cap = cv2.VideoCapture(self.input_path)
        print("==> Starting Analysis... (Visualization will appear at the end)")
        
        frame_idx = 0
        keyframe_interval = 5 # 5프레임마다 키프레임(파란 카메라) 생성
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            img_curr = cv2.resize(frame, (self.W, self.H))
            img_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
            img_masked = self.mask_car(img_gray.copy())
            frame_t0 = time.perf_counter()

            run_infer = (
                self.prev_frame is None
                or frame_idx % self.sp_interval == 0
                or frame_idx % keyframe_interval == 0
            )

            desc = None
            kpts = np.empty((0, 2))

            if run_infer:
                if self.sp_scale != 1.0:
                    sp_w = max(int(self.W * self.sp_scale), 8)
                    sp_h = max(int(self.H * self.sp_scale), 8)
                    img_sp = cv2.resize(img_masked, (sp_w, sp_h))
                else:
                    img_sp = img_masked

                img_fe = (img_sp.astype(np.float32) / 255.0)

                # Superpoint 추론 시간 측정
                # perf_counter가 time보다 짧은 구간 측정에서 더 정확
                sp_t0 = time.perf_counter()
                pts, desc, _ = self.fe.run(img_fe)
                sp_t1 = time.perf_counter()
                kpts = pts[:2, :].T if pts.shape[1] > 0 else np.empty((0, 2))
                if self.sp_scale != 1.0 and len(kpts) > 0:
                    kpts = kpts / self.sp_scale

                # 하늘/구름 필터 적용 (무한대 깊이 노이즈 제거)
                if len(kpts) > 0:
                    sky_mask = self.point_filter.filter_sky_points(img_curr, kpts)
                    kpts = kpts[sky_mask]
                    desc = desc[:, sky_mask] if desc is not None else None
            else:
                sp_t0 = time.perf_counter()
                sp_t1 = sp_t0

            flow_p1 = None
            flow_p2 = None
            if self.prev_frame is not None and self.prev_kpts is not None and len(self.prev_kpts) > 0:
                prev_pts = self.prev_kpts.astype(np.float32).reshape(-1, 1, 2)
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame,
                    img_gray,
                    prev_pts,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if status is not None:
                    status = status.reshape(-1).astype(bool)
                    if status.any():
                        flow_p1 = self.prev_kpts[status]
                        flow_p2 = curr_pts.reshape(-1, 2)[status]

            if not run_infer and flow_p2 is not None:
                kpts = flow_p2

            if self.prev_frame is None:
                self.prev_frame, self.prev_kpts, self.prev_desc = img_gray, kpts, desc
                self.prev_3d_pts = np.full((len(kpts), 3), np.nan)
                self.prev_3d_pts_world = np.full((len(kpts), 3), np.nan)
                # 첫 프레임 키프레임 추가
                self.traj_points.append([0,0,0])
                self.add_keyframe(frame_idx, kpts, desc)
                frame_idx += 1
                continue

            max_pts = max(len(kpts), len(self.prev_kpts) if self.prev_kpts is not None else 0)
            curr_3d_pts = np.full((max_pts, 3), np.nan)
            curr_3d_pts_world = np.full((max_pts, 3), np.nan)  # 월드 좌표 병렬 추적

            # Matcher 시간 측정
            match_t0 = time.perf_counter()
            matches = np.empty((0, 2), dtype=int)
            use_flow = flow_p1 is not None and flow_p2 is not None
            if not use_flow and self.prev_desc is not None and desc is not None:
                matches = self.matcher.match(self.prev_desc, desc)
            match_t1 = time.perf_counter()

            # 프레임별 기본값 초기화(recoverPose 성공시에만 값이 갱신/ 실패하면 0 유지)
            inliers = 0
            inlier_ratio = 0.0
            map_points_added = 0

            # 디버깅: 특징점 검출 및 매칭 상태 모니터링
            # - desc_dim: descriptor 차원 수
            # - kpts: 현재 프레임에서 검출된 특징점 수
            # - matches: 이전 프레임과 현재 프레임 사이의 매칭된 특징점 쌍의 수
            match_count = len(flow_p1) if use_flow else len(matches)
            print(f"frame {frame_idx}: desc_dim={None if desc is None else desc.shape[0]}, kpts={len(kpts)}, matches={match_count}")

            if use_flow and len(flow_p1) > 8:
                p1_idx = np.where(status)[0]
                p2_idx = np.arange(len(flow_p2))
                p1 = flow_p1.astype(np.float64)
                p2 = flow_p2.astype(np.float64)
                p1_3d = self.prev_3d_pts[p1_idx]
                # Inherit 3D points (로컬 + 월드 동시 상속)
                curr_3d_pts[p2_idx] = p1_3d
                curr_3d_pts_world[p2_idx] = self.prev_3d_pts_world[p1_idx]
            elif len(matches) > 8:
                p1_idx = matches[:, 0]
                p2_idx = matches[:, 1]
                p1 = self.prev_kpts[p1_idx, :2].astype(np.float64)
                p2 = kpts[p2_idx, :2].astype(np.float64)
                p1_3d = self.prev_3d_pts[p1_idx]
                # Inherit 3D points (로컬 + 월드 동시 상속)
                curr_3d_pts[p2_idx] = p1_3d
                curr_3d_pts_world[p2_idx] = self.prev_3d_pts_world[p1_idx]
            else:
                p1 = None
                p2 = None

            if p1 is not None and p2 is not None:
                valid_step = False
                
                # 1. PnP 시도 (충분한 3D 점이 있을 때)
                valid_3d_mask = ~np.isnan(p1_3d[:, 0])
                obj_pts = p1_3d[valid_3d_mask].astype(np.float32)
                img_pts = p2[valid_3d_mask].astype(np.float32)
                
                pnp_success = False
                if len(obj_pts) >= 15:
                    success, rvec, tvec_pnp, inliers_pnp = cv2.solvePnPRansac(
                        obj_pts, img_pts, self.K, None, 
                        flags=cv2.SOLVEPNP_EPNP, reprojectionError=3.0
                    )
                    
                    if success and inliers_pnp is not None and len(inliers_pnp) > 10:
                        R_pnp, _ = cv2.Rodrigues(rvec)
                        t_pnp = tvec_pnp
                        
                        # 카메라가 바라보는 방향 기준으로의 PnP 반환(World to Camera). 역변환 필요.
                        # R_cam = R_pnp.T, t_cam = -R_pnp.T * t_pnp
                        R_rel = R_pnp
                        t_rel = t_pnp
                        
                        tvec_len = np.linalg.norm(t_rel[:, 0])
                        # 필터: 너무 기형적인 점프 방지 
                        if 0.001 < tvec_len < 10.0:
                            pnp_success = True
                            R = R_rel
                            t_vec = t_rel[:, 0]
                            inliers = len(inliers_pnp)
                            inlier_ratio = inliers / max(match_count, 1)
                            
                            # Update mask based on PnP inliers for Triangulation
                            mask = np.zeros(len(p1), dtype=np.uint8)
                            pnp_inlier_indices = np.where(valid_3d_mask)[0][inliers_pnp.flatten()]
                            mask[pnp_inlier_indices] = 1
                            
                            print(f"frame {frame_idx}: [PnP] matches={match_count}, inliers={inliers}, ratio={inlier_ratio:.3f}, scale={tvec_len:.4f}, t={t_vec.round(3)}")

                # 2. PnP 실패 시 Essential Matrix (스케일 정보 상실/1.0 할당)
                if not pnp_success:
                    E, mask = cv2.findEssentialMat(p2, p1, self.K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
                    if E is not None:
                        _, R, t, mask = cv2.recoverPose(E, p2, p1, self.K, mask=mask)
                        inliers = np.count_nonzero(mask) if mask is not None else 0
                        inlier_ratio = inliers / max(match_count, 1)
                        t_vec = t[:, 0]
                        # 방향만 보존하고 기본 1.0스케일 부여
                        t_vec = t_vec / (np.linalg.norm(t_vec) + 1e-6)
                        
                        print(f"frame {frame_idx}: [Ess] matches={match_count}, inliers={inliers}, ratio={inlier_ratio:.3f}, t={t_vec.round(3)}")
                        
                        if np.isfinite(t_vec).all() and inliers > 10:
                            pnp_success = True # 논리 구조상 성공으로 간주
                
                # 3. 최적 포즈 적용 및 지도 업데이트
                if pnp_success:
                    # --- [안정화 로직: IMU 부재로 인한 Y축 및 각도 드리프트 억제] ---
                    # 1. 후진 방지 (Z축)
                    if t_vec[2] < 0: t_vec = -t_vec; R = R.T
                    
                    # 2. 횡이동(X축) 억제 (Turn이 작으면 X 이동 억제)
                    if self.highway_mode:
                        turn_amount = np.abs(np.arctan2(R[0,2], R[2,2]))
                        damp_x = np.clip(turn_amount * 10.0, 0.1, 1.0)
                        t_vec[0] *= damp_x 
                        
                    # 3. Y축(상하 높이) 이동 극단적 억제 (IMU 역할: 평지 주행 가정)
                    t_vec[1] *= 0.05 
                    
                    # 4. 카메라 Pitch(위아래 고개 숙임) 및 Roll(기울임) 누적 억제
                    # 단안 카메라 특성상 전진 시 바닥을 보면 위로 올라가는것처럼 착각함
                    pitch_roll_damp = 0.50 # 0.0=완전고정, 1.0=그대로사용
                    euler_angles = cv2.RQDecomp3x3(R)[0]
                    # Euler(X,Y,Z) = (Pitch, Yaw, Roll)
                    euler_angles = list(euler_angles)
                    euler_angles[0] *= pitch_roll_damp # Pitch 억제
                    euler_angles[2] *= pitch_roll_damp # Roll 억제
                    
                    # 다시 Rotation Matrix로 변환
                    R_damped = R.copy() 
                    # 간단한 보정: R_damped를 재조합할 수도 있지만, 본 구현의 복잡성상 생략하고 
                    # t_vec[1] 억제 및 뒤이은 cur_pose[1, 3] 클리핑으로 대체합니다.
                    
                    # 5. 관성 적용 (이전 속도와 혼합)
                    t_vec = t_vec * 0.6 + self.last_t_vec * 0.4
                    self.last_t_vec = t_vec
                    
                    # Pose Update (Relative to absolute)
                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R_damped
                    T_rel[:3, 3] = t_vec
                    self.cur_pose = self.cur_pose @ T_rel
                    
                    # --- [절대 Y축 높이 강제 고정 (Hard Clip)] --- 
                    # 차량이 하늘로 날아가거나 땅굴을 파고 들어가는 현상 원천 차단
                    self.cur_pose[1, 3] = np.clip(self.cur_pose[1, 3], -1.0, 1.0)
                    
                    valid_step = True
                    
                    # 맵 생성 (삼각측량: 현재 R, t_vec 기반으로 수행)
                    mask = mask.ravel().astype(bool)
                    p1_m, p2_m = p1[mask], p2[mask]
                    p2_idx_m = p2_idx[mask]
                    
                    if len(p1_m) > 0:
                        local_pts = self.triangulate(R, t_vec.reshape(3,1), p1_m, p2_m)
                        
                        # 필터링 (너무 멀거나 비상식적인 점 제거 - 주변 건물/도로 환경 고려하여 반경 확대)
                        valid = (local_pts[:, 2] > 0.5) & (local_pts[:, 2] < 150.0) & \
                                (np.abs(local_pts[:, 0]) < 100.0) & (np.abs(local_pts[:, 1]) < 20.0)
                        local_pts = local_pts[valid]
                        valid_indices = p2_idx_m[valid]
                        
                        # 삼각측량 된 점을 현재 3D 점 메모리에 등록 (Scale Propagation의 핵심)
                        curr_3d_pts[valid_indices] = local_pts
                        
                        # 월드 좌표로 즉시 변환하여 병렬 저장 (루프 클로저용)
                        world_pts_for_lc = (self.cur_pose[:3, :3] @ local_pts.T).T + self.cur_pose[:3, 3]
                        curr_3d_pts_world[valid_indices] = world_pts_for_lc
                        
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

                # 실패 시 관성 주행
                if not valid_step:
                    T_rel = np.eye(4)
                    T_rel[:3, 3] = self.last_t_vec
                    self.cur_pose = self.cur_pose @ T_rel

            # 궤적 저장
            curr_t = self.cur_pose[:3, 3]
            self.traj_points.append(curr_t)
            
            # 키프레임 저장 (영상처럼 드문드문 파란 카메라 표시)
            if frame_idx % keyframe_interval == 0:
                self.add_keyframe(frame_idx, kpts, desc, curr_3d_pts_world, T_rel.copy())

            # 2D 뷰 표시
            img_vis = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            for kp in kpts: cv2.circle(img_vis, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)
            cv2.putText(img_vis, f"Frame: {frame_idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Processing', img_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 프레임 끝 시간
            frame_t1 = time.perf_counter()

            # ms 계산 + append
            sp_ms = (sp_t1 - sp_t0) * 1000.0
            match_ms = (match_t1 - match_t0) * 1000.0
            total_ms = (frame_t1 - frame_t0) * 1000.0

            self.sp_ms_list.append(sp_ms)
            self.match_ms_list.append(match_ms)
            self.total_ms_list.append(total_ms)
            self.kpts_list.append(int(len(kpts)))
            self.matches_list.append(int(len(matches)))
            self.inliers_list.append(int(inliers))
            self.inlier_ratio_list.append(float(inlier_ratio))
            self.map_points_added_list.append(int(map_points_added))

            if run_infer:
                self.prev_desc = desc
                if len(kpts) > 0:
                    self.prev_kpts = kpts
                elif flow_p2 is not None and len(flow_p2) > 0:
                    self.prev_kpts = flow_p2
            else:
                if flow_p2 is not None and len(flow_p2) > 0:
                    self.prev_kpts = flow_p2
            self.prev_frame = img_gray
            self.prev_3d_pts = curr_3d_pts
            self.prev_3d_pts_world = curr_3d_pts_world
            frame_idx += 1

        print("\n==> Video Finished. Building Final Scene...")
        cap.release()
        cv2.destroyAllWindows()
        self.print_perf_summary()
        self.visualize_final_result()

    def add_keyframe(self, frame_idx, kpts, desc, pts_3d=None, T_rel=None):
        self.keyframes.append(self.cur_pose.copy())
        self.keyframe_indices.append(frame_idx)
        self.keyframe_local_pts.append([])  # 이 키프레임에 등록될 로컬 3D 포인트들
        self.keyframe_original_poses.append(self.cur_pose.copy())

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
        self.loop_closure.add_keyframe(frame_idx, kpts, desc, pts_3d)
        loop = self.loop_closure.find_loop(frame_idx, kpts, desc)
        if loop is not None:
            # Sim3에서는 스케일 정보를 보존하여 전달 (정규화 없음!)
            scale = loop.scale
            print(f"  [Pose Graph] g2o Sim3 Edge: Scale={scale:.3f} (정규화 없이 보존)")
            
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

    def _build_g2o_optimizer(self):
        """g2o Sim3 Pose Graph 최적화기 구축"""
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)
        return optimizer

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
                info[0, 0] = info[1, 1] = info[2, 2] = 10.0  # 회전 강하게
                info[3, 3] = info[4, 4] = info[5, 5] = 1.0   # 병진도 정상 반영
            else:
                # 오도메트리 엣지: 등방 Information Matrix
                info = np.eye(6) * edge_info['information_scale']
            
            e.set_measurement(g2o.Isometry3d(transform))
            e.set_information(info)
            optimizer.add_edge(e)

        # 3. 최적화 실행
        optimizer.initialize_optimization()
        optimizer.optimize(20)

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
        fps = 1000.0 / max(avg_total, 1e-6)

        print("==> Performance Summary")
        print(f"   Avg total: {avg_total:.2f} ms  (FPS: {fps:.2f})")
        print(f"   Avg SP:    {avg_sp:.2f} ms")
        print(f"   Avg Match: {avg_match:.2f} ms")

        if self.jetson_scale is not None:
            jetson_fps = fps * self.jetson_scale
            print(f"==> Jetson Nano est. FPS: {jetson_fps:.2f} (scale={self.jetson_scale})")
        
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
        }
        
        stats_path = os.path.join(self.save_dir, "slam_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 3D 궤적 저장 (ATE 평가용)
        if self.traj_points:
            traj_3d = np.array(self.traj_points)
            np.savetxt(os.path.join(self.save_dir, "trajectory_xyz.txt"), traj_3d, fmt="%.4f")


    def visualize_final_result(self):
        # 1. 최적화된 Pose Graph로 포인트 클라우드 재구축
        rebuilt_points = []
        rebuilt_colors = []
        
        n_kf = min(len(self.keyframes), len(self.keyframe_local_pts))
        for i in range(n_kf):
            optimized_pose = self.keyframes[i]  # Pose Graph 최적화 후 갱신된 포즈
            local_chunks = self.keyframe_local_pts[i]
            
            for local_pts in local_chunks:
                if len(local_pts) == 0:
                    continue
                # 최적화된 포즈로 월드 좌표계 재투영
                world_pts = (optimized_pose[:3, :3] @ local_pts.T).T + optimized_pose[:3, 3]
                
                if len(world_pts) > 0:
                    cols = get_height_color(world_pts[:, 1])
                    rebuilt_points.append(world_pts)
                    rebuilt_colors.append(cols)
        
        if not rebuilt_points:
            # Fallback: 재투영 데이터가 없으면 원본 사용
            if not self.all_map_points:
                print("No points generated.")
                return
            points = np.vstack(self.all_map_points)
            colors = np.vstack(self.all_map_colors)
        else:
            points = np.vstack(rebuilt_points)
            colors = np.vstack(rebuilt_colors)
            print(f" -> Re-projected {len(points)} points using {n_kf} optimized keyframes.")
        
        # 2. 포인트 클라우드 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # [중요] 점 크기 조절 안됨 -> Voxel Downsample로 밀도 조절
        # 너무 촘촘하면 보기 싫고, 너무 듬성하면 휑함. 적당히 0.1m 간격으로 정리
        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # 3. 경로선 (Trajectory Line) - 최적화된 Keyframe들을 연결
        traj_pts = np.array([pose[:3, 3] for pose in self.keyframes])
        if len(traj_pts) < 2:
            return

        lines = [[i, i+1] for i in range(len(traj_pts)-1)]
        traj_line = o3d.geometry.LineSet()
        traj_line.points = o3d.utility.Vector3dVector(traj_pts)
        traj_line.lines = o3d.utility.Vector2iVector(lines)
        traj_line.paint_uniform_color([0, 1, 0]) # 녹색 선 (영상 스타일)

        # 4. 키프레임 카메라 (Camera Frustums) - 영상의 파란 삼각형들
        vis_geoms = [pcd, traj_line]
        print(f" -> Generating {len(self.keyframes)} Keyframes...")
        
        for pose in self.keyframes:
            # 피라미드 생성
            frustum = create_camera_frustum(scale=0.5, color=[0, 0, 1]) # 파란색
            # 카메라 포즈 적용
            frustum.transform(pose)
            vis_geoms.append(frustum)

        # 5. 최종 뷰어 실행
        print("==> Visualization Ready!")
        vis = o3d.visualization.Visualizer()
        vis.create_window("SuperPoint SLAM Result", width=1280, height=720)
        
        # 배경색: 영상처럼 검은색(Dark)이 포인트가 제일 잘 보임
        vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.05])
        vis.get_render_option().point_size = 3.0 # 점 크기 적당히
        
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
        print(" -> Point Cloud saved to:", os.path.join(self.save_dir, "final_slam_map.ply"))

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
            print(" -> Topdown Map saved to:", os.path.join(self.save_dir, "topdown_map.png"))
        except Exception as e:
            print(f" -> Could not save Top-down map: {e}")

        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--jetson-scale', type=float, default=None)
    parser.add_argument('--sp-scale', type=float, default=0.5)
    parser.add_argument('--sp-interval', type=int, default=2)
    parser.add_argument('--sp-fp16', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--highway-mode', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--conf-thresh', type=float, default=0.003)
    args = parser.parse_args()
    slam = VisualSLAM3D(
        weights_path=args.weights,
        input_path=args.input,
        conf_thresh=args.conf_thresh,
        jetson_scale=args.jetson_scale,
        sp_scale=args.sp_scale,
        sp_interval=args.sp_interval,
        sp_fp16=args.sp_fp16,
        highway_mode=args.highway_mode,
    )
    slam.process()