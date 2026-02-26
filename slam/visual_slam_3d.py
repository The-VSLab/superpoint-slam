import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
import os
import time
import sys

# 기존 모듈
from frontend.superpoint_frontend import SuperPointFrontend
from matcher_module import BTMatcher
from slam.loop_closure import LoopClosureManager

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
    """높이 기반 컬러링 (Turbo Style)"""
    y_vals = np.atleast_1d(y_vals)
    norm = np.clip((y_vals - y_min) / (y_max - y_min), 0, 1)
    colors = np.zeros((len(y_vals), 3))
    # Red(High) -> Green -> Blue(Low)
    colors[:, 0] = np.clip(1.5 - np.abs(2.0 * norm - 1.0) * 3.0, 0, 1) # R
    colors[:, 1] = np.clip(1.5 - np.abs(2.0 * norm - 0.5) * 3.0, 0, 1) # G
    colors[:, 2] = np.clip(1.5 - np.abs(2.0 * norm - 0.0) * 3.0, 0, 1) # B
    return colors

class VisualSLAM3D:
    def __init__(
        self,
        weights_path,
        input_path,
        nn_thresh=0.7,
        jetson_scale=None,
        sp_scale=1.0,
        sp_interval=1,
        sp_fp16=False,
    ):
        # 1. 장치 결정
        self.device = get_optimal_device()
        # SuperPointFrontend는 내부 설계상 True/False(CUDA 사용여부)를 받는 경우가 많으므로 호환성 유지
        self.use_cuda = (self.device == "cuda")
        
        print(f"==> Running on device: {self.device.upper()}")

        self.input_path = input_path
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise ValueError(f"Error: {input_path}")
        self.W, self.H = 640, 480
        cap.release()
        
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
            conf_thresh=0.003,
            nn_thresh=0.7,
            cuda=self.use_cuda,
        )
        self.matcher = BTMatcher(nn_thresh=nn_thresh, use_cuda=self.use_cuda, mutual=True)
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
        self.cur_pose = np.eye(4)
        
        # 데이터 저장소
        self.all_map_points = []
        self.all_map_colors = []
        self.keyframes = [] # keyframe poses for visualization
        self.keyframe_indices = []
        self.traj_points = []
        self.last_t_vec = np.array([0.0, 0.0, 1.0]) 
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.jetson_scale = jetson_scale

        self.save_dir = "path_final"
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
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))
        pts_4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        pts_3d = pts_4d[:3] / pts_4d[3]
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
                # 첫 프레임 키프레임 추가
                self.traj_points.append([0,0,0])
                self.add_keyframe(frame_idx, kpts, desc)
                frame_idx += 1
                continue

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
                p1 = flow_p1.astype(np.float64)
                p2 = flow_p2.astype(np.float64)
            elif len(matches) > 8:
                p1 = self.prev_kpts[matches[:, 0], :2].astype(np.float64)
                p2 = kpts[matches[:, 1], :2].astype(np.float64)
            else:
                p1 = None
                p2 = None

            if p1 is not None and p2 is not None:
                
                # RANSAC (엄격하게)
                E, mask = cv2.findEssentialMat(p2, p1, self.K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
                
                valid_step = False
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, p2, p1, self.K)
                    inliers = np.count_nonzero(mask) if mask is not None else 0
                    inlier_ratio = inliers / max(match_count, 1)
                    t_vec = t[:, 0]

                    # 디버깅: SLAM 추적 상태 모니터링
                    # - matches: 초기 특징점 매칭 쌍의 총 개수
                    # - inliers: RANSAC으로 선별된 기하학적으로 유효한 매칭의 수 (0이 아닌 mask 요소 개수)
                    # - t: 현재 프레임의 상대적 이동 벡터 [x,y,z], 소수점 3자리까지 표시하여 가독성 향상
                    print(f"frame {frame_idx}: matches={match_count}, inliers={inliers}, ratio={inlier_ratio:.3f}, t={t_vec.round(3)}")

                    if np.isfinite(t_vec).all():
                        # --- [안정화 로직: 고속도로 모드] ---
                        # 1. 후진 방지
                        if t_vec[2] < 0: t_vec = -t_vec; R = R.T
                        
                        # 2. 횡이동 억제 (Turn이 작으면 X 이동 억제)
                        turn_amount = np.abs(np.arctan2(R[0,2], R[2,2]))
                        damp_x = np.clip(turn_amount * 10.0, 0.1, 1.0)
                        t_vec[0] *= damp_x 
                        t_vec[1] *= 0.05 # Y(상하) 억제
                        
                        # 3. 관성 적용 (이전 속도와 혼합)
                        t_vec = t_vec * 0.6 + self.last_t_vec * 0.4
                        t_vec = t_vec / (np.linalg.norm(t_vec) + 1e-6) # 정규화 (속도 1.0 고정)
                        
                        self.last_t_vec = t_vec
                        
                        # Pose Update
                        T_rel = np.eye(4)
                        T_rel[:3, :3] = R
                        T_rel[:3, 3] = t_vec
                        self.cur_pose = self.cur_pose @ T_rel
                        valid_step = True
                        
                        # 맵 생성 (삼각측량)
                        mask = mask.ravel().astype(bool)
                        p1_m, p2_m = p1[mask], p2[mask]
                        if len(p1_m) > 0:
                            local_pts = self.triangulate(R, t_vec.reshape(3,1), p1_m, p2_m)
                            
                            # 필터링
                            valid = (local_pts[:, 2] > 1.0) & (local_pts[:, 2] < 200) & \
                                    (np.abs(local_pts[:, 0]) < 100) & (np.abs(local_pts[:, 1]) < 50)
                            local_pts = local_pts[valid]
                            
                            if len(local_pts) > 0:
                                world_pts = (self.cur_pose[:3, :3] @ local_pts.T).T + self.cur_pose[:3, 3]
                                
                                # [청소] 바닥 아래 지하 노이즈 제거
                                # OpenCV 좌표계: +Y가 아래. 바닥은 약 +1.6 ~ 1.7
                                # 1.8보다 큰 값(더 아래)은 노이즈
                                valid_ground = world_pts[:, 1] < 1.8
                                world_pts = world_pts[valid_ground]
                                # map_points_added 기록
                                map_points_added = int(world_pts.shape[0])

                                # 색상 계산
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
                self.add_keyframe(frame_idx, kpts, desc)

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
            frame_idx += 1

        print("\n==> Video Finished. Building Final Scene...")
        cap.release()
        cv2.destroyAllWindows()
        self.print_perf_summary()
        self.visualize_final_result()

    def add_keyframe(self, frame_idx, kpts, desc):
        self.keyframes.append(self.cur_pose.copy())
        self.keyframe_indices.append(frame_idx)

        node_idx = len(self.pose_graph.nodes)
        self.pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(self.cur_pose.copy())
        )

        if node_idx > 0:
            prev_pose = self.pose_graph.nodes[node_idx - 1].pose
            rel = np.linalg.inv(prev_pose) @ self.cur_pose
            information = np.eye(6)
            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    node_idx - 1,
                    node_idx,
                    rel,
                    information,
                    uncertain=False,
                )
            )

        self.loop_closure.add_keyframe(frame_idx, kpts, desc)
        loop = self.loop_closure.find_loop(frame_idx, kpts, desc)
        if loop is not None:
            information = np.eye(6)
            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    loop.match_index,
                    node_idx,
                    loop.transform,
                    information,
                    uncertain=True,
                )
            )
            self.optimize_pose_graph()

    def optimize_pose_graph(self):
        if len(self.pose_graph.nodes) < 2:
            return

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=1.0,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        for i, node in enumerate(self.pose_graph.nodes):
            self.keyframes[i] = node.pose.copy()
        self.cur_pose = self.pose_graph.nodes[-1].pose.copy()

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

    def visualize_final_result(self):
        # 1. 포인트 클라우드 병합
        if not self.all_map_points:
            print("No points generated.")
            return
            
        points = np.vstack(self.all_map_points)
        colors = np.vstack(self.all_map_colors)
        
        # 2. 포인트 클라우드 객체 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # [중요] 점 크기 조절 안됨 -> Voxel Downsample로 밀도 조절
        # 너무 촘촘하면 보기 싫고, 너무 듬성하면 휑함. 적당히 0.1m 간격으로 정리
        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # 3. 경로선 (Trajectory Line)
        traj_pts = np.array(self.traj_points)
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

        vis.run()
        vis.destroy_window()
        
        # 저장
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "final_slam_map.ply"), pcd)
        print(" -> Map saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--jetson-scale', type=float, default=None)
    parser.add_argument('--sp-scale', type=float, default=0.5)
    parser.add_argument('--sp-interval', type=int, default=2)
    parser.add_argument('--sp-fp16', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    slam = VisualSLAM3D(
        weights_path=args.weights,
        input_path=args.input,
        jetson_scale=args.jetson_scale,
        sp_scale=args.sp_scale,
        sp_interval=args.sp_interval,
        sp_fp16=args.sp_fp16,
    )
    slam.process()