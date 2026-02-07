import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
import os
import sys

# 기존 모듈 임포트
from frontend.superpoint_frontend import SuperPointFrontend
from matcher_module import BTMatcher

# --- 환경별 장치 자동 설정 함수 ---
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
        target_size=(640, 480),
        nn_thresh=0.7,
        mask_car=False,
        conf_thresh=0.003,
        nms_dist=4,
    ):
        # 1. 장치 결정
        self.device = get_optimal_device()
        self.use_cuda = (self.device == "cuda")
        
        print(f"==> Running on device: {self.device.upper()}")

        self.input_path = input_path
        
        # [수정 1] 해상도 고정 (Resolution Mismatch 해결)
        # 사용자가 입력한 target_size (예: 640x480)를 기준으로 모든 파라미터 고정
        self.W = target_size[0]
        self.H = target_size[1]
        
        print(f"==> Target resolution set to: {self.W}x{self.H}")

        # [수정 2] 카메라 파라미터 (변경된 W, H 기준)
        avg_dim = (self.W + self.H) / 2.0
        # Focal Length는 이미지 크기에 비례하여 설정 (0.85는 일반적인 FOV)
        self.focal = avg_dim * 0.85  
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.K = np.array([[self.focal, 0, self.cx], 
                          [0, self.focal, self.cy], 
                          [0, 0, 1]], dtype=np.float64)
        
        print(f"==> Camera Matrix (K):\n{self.K}")

        print("==> Loading SuperPoint...")
        self.fe = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=self.use_cuda,
        )
        self.matcher = BTMatcher(nn_thresh=nn_thresh, use_cuda=self.use_cuda, mutual=True)

        self.prev_frame = None
        self.prev_kpts = None
        self.prev_desc = None
        self.cur_pose = np.eye(4)
        
        # 데이터 저장소
        self.all_map_points = []
        self.all_map_colors = []
        self.keyframes = [] # (Pose, Frustum)
        self.traj_points = []
        self.last_t_vec = np.array([0.0, 0.0, 1.0]) 

        self.save_dir = "path_final"
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        
        self.mask_car_enabled = mask_car

        # 실시간 2D 확인창
        cv2.namedWindow('Processing', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processing', 640, 360)

    def triangulate(self, R, t, p1, p2):
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))
        pts_4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        pts_3d = pts_4d[:3] / pts_4d[3]
        return pts_3d.T

    def mask_car(self, img):
        # 대시보드(차체) 마스킹 (KITTI 기준 하단부 제거)
        h, w = img.shape[:2]
        img[int(h*0.82):, :] = 0 
        return img

    def process(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
             raise ValueError(f"Error: Cannot open video file {self.input_path}")

        print("==> Starting Analysis... (Visualization will appear at the end)")
        
        frame_idx = 0
        keyframe_interval = 5 
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            # [수정 3] 단일 리사이즈 (Double Resize 제거)
            # __init__에서 설정한 크기로 바로 변경하여 수학적 모델과 일치시킴
            img_curr = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
            
            img_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
            
            if self.mask_car_enabled:
                img_masked = self.mask_car(img_gray.copy())
            else:
                img_masked = img_gray
            
            # [수정 4] ★ 핵심 정규화 (Normalization) ★
            # 0~255 값을 0~1.0으로 변환하여 모델 입력 (모기장 현상 해결)
            img_fe = (img_masked.astype(np.float32) / 255.0)

            # 특징점 추출
            pts, desc, _ = self.fe.run(img_fe)
            kpts = pts[:2, :].T if pts.shape[1] > 0 else np.empty((0, 2))

            if self.prev_frame is None:
                self.prev_frame, self.prev_kpts, self.prev_desc = img_gray, kpts, desc
                # 첫 프레임 초기화
                self.traj_points.append([0,0,0])
                self.keyframes.append(np.eye(4))
                frame_idx += 1
                continue

            matches = self.matcher.match(self.prev_desc, desc)

            if len(matches) > 8:
                p1 = self.prev_kpts[matches[:, 0], :2].astype(np.float64)
                p2 = kpts[matches[:, 1], :2].astype(np.float64)
                
                # RANSAC
                E, mask = cv2.findEssentialMat(p2, p1, self.K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
                
                valid_step = False
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, p2, p1, self.K)
                    t_vec = t[:, 0]
                    
                    if np.isfinite(t_vec).all():
                        # --- [안정화 로직] ---
                        # 1. 후진 방지
                        if t_vec[2] < 0: t_vec = -t_vec; R = R.T
                        
                        # 2. 횡이동 억제
                        turn_amount = np.abs(np.arctan2(R[0,2], R[2,2]))
                        damp_x = np.clip(turn_amount * 10.0, 0.1, 1.0)
                        t_vec[0] *= damp_x 
                        t_vec[1] *= 0.05 
                        
                        # 3. 관성 적용
                        t_vec = t_vec * 0.6 + self.last_t_vec * 0.4
                        t_vec = t_vec / (np.linalg.norm(t_vec) + 1e-6) 
                        
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
                            
                            valid = (local_pts[:, 2] > 1.0) & (local_pts[:, 2] < 200) & \
                                    (np.abs(local_pts[:, 0]) < 100) & (np.abs(local_pts[:, 1]) < 50)
                            local_pts = local_pts[valid]
                            
                            if len(local_pts) > 0:
                                world_pts = (self.cur_pose[:3, :3] @ local_pts.T).T + self.cur_pose[:3, 3]
                                valid_ground = world_pts[:, 1] < 1.8
                                world_pts = world_pts[valid_ground]
                                cols = get_height_color(world_pts[:, 1])
                                self.all_map_points.append(world_pts)
                                self.all_map_colors.append(cols)

                if not valid_step:
                    T_rel = np.eye(4)
                    T_rel[:3, 3] = self.last_t_vec
                    self.cur_pose = self.cur_pose @ T_rel

            # 궤적 저장
            curr_t = self.cur_pose[:3, 3]
            self.traj_points.append(curr_t)
            
            if frame_idx % keyframe_interval == 0:
                self.keyframes.append(self.cur_pose.copy())

            # 2D 뷰 표시
            img_vis = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            for kp in kpts: cv2.circle(img_vis, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)
            cv2.putText(img_vis, f"Frame: {frame_idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Processing', img_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            self.prev_frame, self.prev_kpts, self.prev_desc = img_gray, kpts, desc
            frame_idx += 1

        print("\n==> Video Finished. Building Final Scene...")
        cap.release()
        cv2.destroyAllWindows()
        self.visualize_final_result()

    def visualize_final_result(self):
        if not self.all_map_points:
            print("No points generated.")
            return
            
        points = np.vstack(self.all_map_points)
        colors = np.vstack(self.all_map_colors)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        traj_pts = np.array(self.traj_points)
        lines = [[i, i+1] for i in range(len(traj_pts)-1)]
        traj_line = o3d.geometry.LineSet()
        traj_line.points = o3d.utility.Vector3dVector(traj_pts)
        traj_line.lines = o3d.utility.Vector2iVector(lines)
        traj_line.paint_uniform_color([0, 1, 0]) 

        vis_geoms = [pcd, traj_line]
        print(f" -> Generating {len(self.keyframes)} Keyframes...")
        
        for pose in self.keyframes:
            frustum = create_camera_frustum(scale=0.5, color=[0, 0, 1]) 
            frustum.transform(pose)
            vis_geoms.append(frustum)

        print("==> Visualization Ready!")
        vis = o3d.visualization.Visualizer()
        vis.create_window("SuperPoint SLAM Result", width=1280, height=720)
        vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.05])
        vis.get_render_option().point_size = 3.0 
        
        for geom in vis_geoms:
            vis.add_geometry(geom)
            
        ctr = vis.get_view_control()
        ctr.set_lookat(traj_pts[len(traj_pts)//2]) 
        ctr.set_front([-0.5, -1.0, -0.5]) 
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.5)

        vis.run()
        vis.destroy_window()
        
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "final_slam_map.ply"), pcd)
        print(" -> Map saved.")

if __name__ == "__main__":
    print(
        "이 모듈은 라이브러리로 사용됩니다. "
        "SLAM 실행은 `scripts/superpoint_app.py --mode slam ...`을 사용하세요."
    )
