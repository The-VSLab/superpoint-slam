"""SLAM 공통 유틸리티 (visual_slam_3d / visual_slam_drone 공유)."""
import logging

import numpy as np
import torch

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    o3d = None
    HAS_O3D = False

logger = logging.getLogger(__name__)


def desc_to_numpy(desc):
    """Descriptor를 numpy 배열로 변환 (torch.Tensor → numpy, 이미 numpy면 그대로 반환)."""
    if desc is None:
        return None
    if isinstance(desc, torch.Tensor):
        return desc.cpu().numpy()
    return desc


def desc_filter_by_mask(desc, mask):
    """Boolean mask로 descriptor 열을 필터링. Tensor/numpy 모두 지원."""
    if desc is None:
        return None
    if isinstance(desc, torch.Tensor):
        return desc[:, torch.from_numpy(mask).to(desc.device)]
    return desc[:, mask]


def recover_scale(t_vec, recent_speeds, last_t_vec):
    """단위 벡터 t_vec에 이전 속도 기반 스케일을 복원하여 반환."""
    vec_norm = np.linalg.norm(t_vec) + 1e-6
    if len(recent_speeds) > 0:
        prev_speed = np.median(recent_speeds[-10:])
    else:
        prev_speed = np.linalg.norm(last_t_vec)
    return (t_vec / vec_norm) * (prev_speed if prev_speed > 0 else 1.0)


def get_optimal_device() -> str:
    """NVIDIA GPU(CUDA), Apple Silicon(MPS), CPU 중 최적의 장치를 반환"""
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
    colors[:, 0] = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)  # R
    colors[:, 1] = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)  # G
    colors[:, 2] = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)  # B
    return colors
