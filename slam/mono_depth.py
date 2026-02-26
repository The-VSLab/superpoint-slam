import cv2
import numpy as np
import torch
import torch.nn.functional as F

class MonoDepthEstimator:
    """MiDaS와 실시간 기하 추정을 통합한 깊이 추정기"""

    def __init__(self, model_type="fast", device="cuda", use_fp16=False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.initialized = False

        if model_type == "fast":
            self.initialized = True # 별도 로드 없음
        else:
            try:
                self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
                self.model.to(self.device).eval()
                self.transform = torch.hub.load("intel-isl/MiDaS", f"transforms_{model_type.replace('.', '_')}")
                self.initialized = True
            except Exception as e:
                print(f"⚠️ MiDaS 로드 실패: {e}")

    def estimate(self, frame):
        """정규화된 깊이맵 반환"""
        if self.model_type == "fast":
            # Y 좌표 기반 실시간 깊이 맵 (하단=가까움=1.0)
            h, w = frame.shape[:2]
            y_grad = np.linspace(0, 1, h).reshape(h, 1).repeat(w, axis=1).astype(np.float32)
            return y_grad
        
        # MiDaS 추론 로직 (기존 유지)
        input_batch = self.transform(frame).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            output = F.interpolate(prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic").squeeze()
        depth = output.cpu().numpy()
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    def get_3d_points(self, kpts_2d, depth_map, camera_matrix=None):
        """특징점을 3D로 변환"""
        if len(kpts_2d) == 0: return np.array([]).reshape(0, 3)

        h, w = depth_map.shape
        x, y = kpts_2d[:, 0].astype(int), kpts_2d[:, 1].astype(int)
        
        # 미터(m) 단위 거리로 변환
        if self.model_type == "fast":
            # y_norm 기반 2m~30m 매핑
            actual_depths = 2.0 + (30.0 - 2.0) * (1.0 - (y / h))
        else:
            actual_depths = depth_map[y, x] * 20.0

        # 역프로젝션
        fx = camera_matrix[0, 0] if camera_matrix is not None else w / 2
        cx = camera_matrix[0, 2] if camera_matrix is not None else w / 2
        fy = camera_matrix[1, 1] if camera_matrix is not None else h / 2
        cy = camera_matrix[1, 2] if camera_matrix is not None else h / 2

        z = actual_depths
        x_3d = (kpts_2d[:, 0] - cx) / fx * z
        y_3d = (kpts_2d[:, 1] - cy) / fy * z
        return np.column_stack([x_3d, y_3d, z])