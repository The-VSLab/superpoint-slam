import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
import os
import sys

# 기존 모듈
from scripts.py_superpoint import SuperPointFrontend
from matcher_module import BTMatcher

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
    def __init__(self, weights_path, input_path, nn_thresh=0.7):
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
        
        print(f"==> Resolution: {self.W}x{self.H}")

        # 카메라 파라미터 (일반적인 블랙박스 화각)
        self.focal = max(self.W, self.H) * 0.8
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.K = np.array([[self.focal, 0, self.cx], [0, self.focal, self.cy], [0, 0, 1]])

        print("==> Loading SuperPoint...")
        # descriptor_dim=128로 테스트 후 안정적일 때 head_hidden 256-> 128로 전환예정
        self.fe = SuperPointFrontend(weights_path=weights_path, nms_dist=4, conf_thresh=0.003, nn_thresh=0.7, cuda=self.use_cuda, head_hidden=256, descriptor_dim=128)
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
        # 대시보드(차체) 마스킹
        h, w = img.shape[:2]
        img[int(h*0.82):, :] = 0 
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
            img_fe = (img_masked.astype(np.float32) / 255.0)

            # 특징점 추출
            pts, desc, _ = self.fe.run(img_fe)
            kpts = pts[:2, :].T if pts.shape[1] > 0 else np.empty((0, 2))

            if self.prev_frame is None:
                self.prev_frame, self.prev_kpts, self.prev_desc = img_gray, kpts, desc
                # 첫 프레임 키프레임 추가
                self.traj_points.append([0,0,0])
                self.keyframes.append(np.eye(4))
                frame_idx += 1
                continue

            matches = self.matcher.match(self.prev_desc, desc)

            # 디버깅: 특징점 검출 및 매칭 상태 모니터링
            # - desc_dim: descriptor 차원 수
            # - kpts: 현재 프레임에서 검출된 특징점 수
            # - matches: 이전 프레임과 현재 프레임 사이의 매칭된 특징점 쌍의 수
            print(f"frame {frame_idx}: desc_dim={None if desc is None else desc.shape[0]}, kpts={len(kpts)}, matches={len(matches)}")

            if len(matches) > 8:
                p1 = self.prev_kpts[matches[:, 0], :2].astype(np.float64)
                p2 = kpts[matches[:, 1], :2].astype(np.float64)
                
                # RANSAC (엄격하게)
                E, mask = cv2.findEssentialMat(p2, p1, self.K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
                
                valid_step = False
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, p2, p1, self.K)
                    inliers = int(mask.ravel().sum()) if mask is not None else 0
                    t_vec = t[:, 0]

                    # 디버깅: SLAM 추적 상태 모니터링
                    # - matches: 초기 특징점 매칭 쌍의 총 개수
                    # - inliers: RANSAC으로 선별된 기하학적으로 유효한 매칭의 수 (mask.ravel().sum())
                    # - t: 현재 프레임의 상대적 이동 벡터 [x,y,z], 소수점 3자리까지 표시하여 가독성 향상
                    print(f"frame {frame_idx}: matches={len(matches)}, inliers={inliers}, t={t_vec.round(3)}")

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
    args = parser.parse_args()
    slam = VisualSLAM3D(weights_path=args.weights, input_path=args.input)
    slam.process()