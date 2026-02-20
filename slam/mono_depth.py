"""
MiDaS를 이용한 모노 깊이 추정
벽, 천장, 주변 물체의 3D 구조 복원
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


class MonoDepthEstimator:
    """MiDaS 기반 모노 깊이 추정기"""

    def __init__(self, model_type="dpt_large", device="cuda", use_fp16=False):
        """
        Args:
            model_type: "dpt_large" | "dpt_hybrid" | "midas_v21_small"
            device: "cuda" | "cpu"
            use_fp16: FP16 추론 사용 여부
        """
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and device == "cuda"
        self.model_type = model_type

        # 모델 로드
        try:
            self.model = self._load_midas_model()
            self.transform = torch.hub.load("intel-isl/MiDaS", f"transforms_{model_type.replace('.', '_')}")
            self.initialized = True
        except Exception as e:
            print(f"⚠️ 깊이 추정 모델 로드 실패: {e}")
            print("   설치: pip install timm")
            self.initialized = False

    def _load_midas_model(self):
        """MiDaS 모델 로드"""
        try:
            model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            model = model.to(self.device)
            if self.use_fp16:
                model = model.half()
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"MiDaS '{self.model_type}' 로드 실패: {e}")

    def estimate(self, frame):
        """
        프레임에서 깊이맵 추정

        Args:
            frame: BGR 이미지 (H, W, 3)

        Returns:
            depth_map: 정규화된 깊이맵 (H, W), 0~1 범위
        """
        if not self.initialized:
            return None

        with torch.no_grad():
            # 전처리
            input_batch = self.transform(frame).to(self.device)
            if self.use_fp16:
                input_batch = input_batch.half()

            # 추론
            prediction = self.model(input_batch)

            # 원본 해상도로 리사이즈
            output = F.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # 정규화 (0-1)
            depth_map = output.cpu().numpy()
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        return depth_normalized

    def get_3d_points(self, kpts_2d, depth_map, camera_matrix=None):
        """
        2D 특징점을 깊이맵으로부터 3D 좌표로 복원

        Args:
            kpts_2d: 2D 특징점 (N, 2), (x, y) 형식
            depth_map: 깊이맵 (H, W)
            camera_matrix: 카메라 내부 파라미터 (3, 3), 없으면 단순화 버전 사용

        Returns:
            points_3d: 3D 점들 (N, 3), (x, y, z) 형식
        """
        if kpts_2d is None or len(kpts_2d) == 0:
            return np.array([]).reshape(0, 3)

        h, w = depth_map.shape
        kpts_2d = np.asarray(kpts_2d)

        # 범위 클리핑
        x = np.clip(kpts_2d[:, 0].astype(int), 0, w - 1)
        y = np.clip(kpts_2d[:, 1].astype(int), 0, h - 1)

        # 깊이값 추출
        depths = depth_map[y, x]

        if camera_matrix is None:
            # 단순화된 버전: 초점거리 = 1로 가정
            fx = fy = w / 2
            cx = w / 2
            cy = h / 2
        else:
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # 역프로젝션
        z = depths
        x_norm = (kpts_2d[:, 0] - cx) / fx * z
        y_norm = (kpts_2d[:, 1] - cy) / fy * z

        points_3d = np.column_stack([x_norm, y_norm, z])
        return points_3d


class SimpleDepthEstimator:
    """구조로부터의 모션(SfM) 기반 깊이 추정 (MiDaS 없을 때 대체용)"""

    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.initialized = False

    def estimate(self, frame, camera_motion=None):
        """
        엣지 감지 + 광류 + 특징 밀도로부터 깊이 추정

        Args:
            frame: 현재 프레임
            camera_motion: 카메라 이동 벡터 (optional)

        Returns:
            depth_map: 추론된 깊이맵 (0~1 범위)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        depth_map = np.ones((h, w), dtype=np.float32) * 0.5

        # 1. 엣지 감지 (벽, 경계 구조) - 매우 강조
        edges = cv2.Canny(gray, 30, 100)  # 더 낮은 임계값으로 더 많은 엣지
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        edge_depth = 1.0 - (edges.astype(np.float32) / 255.0) * 0.8  # 엣지: 깊음 (0.2), 평면: 얕음 (1.0)
        
        # 2. 광류로 모션 감지
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            # 큰 광류 = 가까운 물체 (깊이 큼 = 1.0)
            motion_depth = np.minimum(mag / 20.0, 1.0)
        else:
            motion_depth = np.ones((h, w), dtype=np.float32) * 0.5

        # 3. 밝기 변화율 (gradients) - 정면 벽 감지 매우 강조
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_depth = np.minimum(grad_mag / 50.0, 1.0)  # 높은 gradient = 깊음
        
        # 4. 멀티 스케일 텍스처 분석
        texture_depth = np.zeros((h, w), dtype=np.float32)
        for scale in [5, 11, 21]:
            blurred = cv2.GaussianBlur(gray, (scale, scale), 0)
            texture = cv2.absdiff(gray.astype(np.float32), blurred.astype(np.float32))
            texture_depth += np.minimum(texture / 80.0, 1.0)
        texture_depth /= 3.0
        
        # 5. 컨투어 기반 깊이 추정 (벽 경계)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_depth = np.zeros((h, w), dtype=np.float32)
        for contour in contours:
            cv2.drawContours(contour_depth, [contour], 0, 1.0, 2)
        contour_depth = cv2.GaussianBlur(contour_depth, (7, 7), 0)
        
        # 6. 모든 신호 결합 (가중 결합 - 벽 감지 매우 강조)
        depth_map = (
            0.35 * edge_depth +        # 엣지: 35%
            0.25 * motion_depth +      # 광류: 25%
            0.25 * grad_depth +        # 그래디언트: 25%
            0.10 * texture_depth +     # 텍스처: 10%
            0.05 * contour_depth       # 컨투어: 5%
        )
        
        # 정규화
        depth_map = np.clip(depth_map, 0.0, 1.0)
        # 히스토그램 평형화 (명암도 증가)
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        self.prev_gray = gray
        return depth_map

    def get_3d_points(self, kpts_2d, depth_map, camera_matrix=None):
        """MiDaS와 동일한 인터페이스"""
        if kpts_2d is None or len(kpts_2d) == 0:
            return np.array([]).reshape(0, 3)

        h, w = depth_map.shape
        kpts_2d = np.asarray(kpts_2d)
        x = np.clip(kpts_2d[:, 0].astype(int), 0, w - 1)
        y = np.clip(kpts_2d[:, 1].astype(int), 0, h - 1)
        depths = depth_map[y, x]

        if camera_matrix is None:
            fx = fy = w / 2
            cx = w / 2
            cy = h / 2
        else:
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        z = depths
        x_norm = (kpts_2d[:, 0] - cx) / fx * z
        y_norm = (kpts_2d[:, 1] - cy) / fy * z
        points_3d = np.column_stack([x_norm, y_norm, z])

        return points_3d
