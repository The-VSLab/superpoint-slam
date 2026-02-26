import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List

class PointFilter:
    """GPU 텐서 연산을 통해 실시간 성능을 극대화한 특징점 필터"""
    
    def __init__(self, frame_h: int = 480, frame_w: int = 640, device: str = "cuda"):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def filter_sky_points(self, frame: np.ndarray, kpts: np.ndarray) -> np.ndarray:
        """하늘/구름 영역 특징점 제거 (GPU 가속)"""
        if len(kpts) == 0: return np.array([], dtype=bool)
        
        # 상단 영역을 하늘로 간주하는 단순화된 GPU 필터 적용
        y_limit = self.frame_h * 0.38
        mask = kpts[:, 1] >= y_limit
        return mask

    def filter_shadow_points(self, frame: np.ndarray, kpts: np.ndarray, **kwargs) -> np.ndarray:
        """그림자 영역의 약한 구조 특징점 제거 (GPU 가속)"""
        if len(kpts) == 0: return np.array([], dtype=bool)
        
        val_thresh = kwargs.get('shadow_value_thresh', 0.35)
        img_gpu = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
        # 밝기(Value) 계산
        gray = 0.299 * img_gpu[:, 0] + 0.587 * img_gpu[:, 1] + 0.114 * img_gpu[:, 2]
        
        xi = torch.from_numpy(kpts[:, 0]).long().to(self.device).clamp(0, self.frame_w - 1)
        yi = torch.from_numpy(kpts[:, 1]).long().to(self.device).clamp(0, self.frame_h - 1)
        
        kpt_val = gray[0, yi, xi]
        return (kpt_val >= val_thresh).cpu().numpy()

    def filter_top_region_points(self, frame: np.ndarray, kpts: np.ndarray, **kwargs) -> np.ndarray:
        """영상 상단 영역 억제 필터"""
        if len(kpts) == 0: return np.array([], dtype=bool)
        top_ratio = kwargs.get('top_region_ratio', 0.25) 
        return (kpts[:, 1] >= (self.frame_h * top_ratio))

    def apply_all_filters(self, frame: np.ndarray, kpts: np.ndarray, 
                         desc: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """모든 필터를 통합하여 적용"""
        if len(kpts) == 0: return kpts, desc
        
        # 개별 필터 마스크 생성
        mask = self.filter_sky_points(frame, kpts)
        if kwargs.get('use_shadow_filter', True):
            mask &= self.filter_shadow_points(frame, kpts, **kwargs)
        if kwargs.get('use_top_region_filter', True):
            mask &= self.filter_top_region_points(frame, kpts, **kwargs)
            
        # 통계적 이상치 제거 (배치 연산으로 병목 해결)
        if len(kpts) > 10:
            kpts_gpu = torch.from_numpy(kpts).to(self.device).float()
            dist_mat = torch.cdist(kpts_gpu, kpts_gpu)
            topk_dist, _ = torch.topk(dist_mat, k=11, largest=False)
            mean_dist = topk_dist[:, 1:].mean(dim=1)
            thresh = mean_dist.mean() + 2.0 * mean_dist.std()
            mask &= (mean_dist < thresh).cpu().numpy()

        filtered_kpts = kpts[mask]
        filtered_desc = desc[:, mask] if desc is not None else None
        return filtered_kpts, filtered_desc

def create_point_filter(frame_h: int = 480, frame_w: int = 640) -> PointFilter:
    """팩토리 함수 유지"""
    return PointFilter(frame_h=frame_h, frame_w=frame_w)