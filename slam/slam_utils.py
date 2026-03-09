"""SLAM 공통 유틸리티 (visual_slam_3d / visual_slam_drone 공유)."""
import numpy as np
import open3d as o3d
import torch


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
