"""
포인트 필터링 모듈 - 노이즈 제거
- 구름/하늘 필터
- 전선/케이블 필터  
- 통계적 이상치 제거
- 기하학적 일관성 필터
"""
import cv2
import numpy as np
from typing import Tuple, List


class PointFilter:
    """SuperPoint 특징점 필터링"""
    
    def __init__(self, frame_h: int = 480, frame_w: int = 640):
        self.frame_h = frame_h
        self.frame_w = frame_w
    
    def filter_sky_points(self, frame: np.ndarray, kpts: np.ndarray, 
                         blue_thresh: float = 0.5, saturation_thresh: float = 0.3) -> np.ndarray:
        """하늘/구름 필터링 (HSV 색상 기반)
        
        Args:
            frame: BGR 원본 이미지
            kpts: 특징점 (N, 2)
            blue_thresh: 파란색 채널 비율 임계값
            saturation_thresh: 채도 임계값 (구름은 낮은 채도)
        
        Returns:
            필터링된 특징점 인덱스 (유효한 점들)
        """
        if len(kpts) == 0:
            return np.array([], dtype=bool)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # HSV 채널 분리
        h = hsv[:, :, 0] / 180.0
        s = hsv[:, :, 1] / 255.0
        v = hsv[:, :, 2] / 255.0
        
        # 파란색: Hue 100-130도 (0.55-0.72)
        is_blue = (h >= 0.5) & (h <= 0.75)
        
        # 구름: 낮은 채도 + 높은 명도
        is_cloud = (s < saturation_thresh) & (v > 0.6)
        
        # 하늘 영역: 파란색이거나 구름 같은 형태
        sky_mask = is_blue | is_cloud
        
        # 특징점 위치에서 하늘 여부 확인
        valid = np.ones(len(kpts), dtype=bool)
        for idx, (x, y) in enumerate(kpts):
            xi = int(np.clip(x, 0, self.frame_w - 1))
            yi = int(np.clip(y, 0, self.frame_h - 1))
            if sky_mask[yi, xi]:
                valid[idx] = False
        
        return valid
    
    def filter_lines(self, kpts: np.ndarray, desc: np.ndarray,
                    line_angle_tolerance: float = 10.0) -> np.ndarray:
        """직선엣지/전선 필터링 (특징점 방향성 기반)
        
        Args:
            kpts: 특징점 (N, 2)
            desc: 기술자 (특징점과 동일 개수)
            line_angle_tolerance: 각도 편차 임계값 (도)
        
        Returns:
            필터링된 특징점 인덱스 (valid mask)
        """
        if len(kpts) < 3:
            return np.ones(len(kpts), dtype=bool)
        
        valid = np.ones(len(kpts), dtype=bool)
        
        # 각 특징점 거리 정렬
        for idx in range(len(kpts)):
            # 같은 이웃 특징점들까지의 거리 계산
            distances = np.linalg.norm(kpts - kpts[idx], axis=1)
            distances[idx] = np.inf
            
            nearest_indices = np.argsort(distances)[:5]  # 가장 가까운 5개
            
            if len(nearest_indices) < 3:
                continue
            
            # 이웃 점들과의 방향성 확인
            neighbors = kpts[nearest_indices]
            directions = neighbors - kpts[idx]
            
            # 모든 방향의 각도 계산
            angles = np.arctan2(directions[:, 1], directions[:, 0]) * 180 / np.pi
            
            # 각도의 분산이 작으면 직선 위의 점
            angle_std = np.std(angles)
            
            # 분산이 매우 작으면 (직선상) 필터링
            if angle_std < line_angle_tolerance:
                valid[idx] = False
        
        return valid
    
    def filter_statistical_outliers(self, kpts: np.ndarray, 
                                   neighborhood: int = 10,
                                   std_ratio: float = 2.0) -> np.ndarray:
        """통계적 이상치 제거 (neighborhood distance 기반)
        
        Args:
            kpts: 특징점 (N, 2)
            neighborhood: 이웃 k개
            std_ratio: 표준편차 배수 (넘으면 이상치)
        
        Returns:
            유효한 포인트 마스크
        """
        if len(kpts) < neighborhood:
            return np.ones(len(kpts), dtype=bool)
        
        # 각 점의 이웃까지 거리 계산
        distances_all = np.linalg.norm(kpts[:, None] - kpts[None, :], axis=2)
        
        # 가장 가까운 k개까지의 평균 거리
        sorted_dist = np.sort(distances_all, axis=1)
        k_nearest_mean = np.mean(sorted_dist[:, 1:neighborhood+1], axis=1)
        
        # 통계 계산
        mean_dist = np.mean(k_nearest_mean)
        std_dist = np.std(k_nearest_mean)
        
        # 임계값 설정 (mean + std_ratio * std)
        threshold = mean_dist + std_ratio * std_dist
        
        # 유효한 점 (거리가 임계값 이하)
        valid = k_nearest_mean < threshold
        
        return valid
    
    def filter_low_confidence(self, kpts: np.ndarray, desc: np.ndarray,
                             confidence: np.ndarray = None,
                             min_conf: float = 0.1) -> np.ndarray:
        """낮은 신뢰도 포인트 제거
        
        Args:
            kpts: 특징점 (N, 2)
            desc: 기술자 (descriptor norm으로 신뢰도 사용 가능)
            confidence: 신뢰도 값 (제공되면 사용)
            min_conf: 최소 신뢰도 임계값
        
        Returns:
            유효한 포인트 마스크
        """
        if len(kpts) == 0:
            return np.array([], dtype=bool)
        
        if confidence is not None:
            return confidence >= min_conf
        
        # desc가 있으면 norm으로 신뢰도 추정
        if desc is not None and len(desc) > 0:
            desc_norm = np.linalg.norm(desc, axis=0 if desc.shape[0] < desc.shape[1] else 1)
            desc_norm = desc_norm / (np.max(desc_norm) + 1e-8)
            return desc_norm >= min_conf
        
        return np.ones(len(kpts), dtype=bool)
    
    def apply_all_filters(self, frame: np.ndarray, kpts: np.ndarray, 
                         desc: np.ndarray = None, confidence: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """모든 필터를 순서대로 적용
        
        Args:
            frame: BGR 원본 이미지
            kpts: 특징점 (N, 2)
            desc: 기술자 (선택사항)
            confidence: 신뢰도 (선택사항)
        
        Returns:
            (필터링된_특징점, 필터링된_기술자)
        """
        if len(kpts) == 0:
            return kpts, desc
        
        # 순차적으로 필터 적용
        mask_sky = self.filter_sky_points(frame, kpts)
        mask_lines = self.filter_lines(kpts, desc)
        mask_stats = self.filter_statistical_outliers(kpts)
        mask_conf = self.filter_low_confidence(kpts, desc, confidence)
        
        # 모든 필터 합치기 (AND 연산)
        final_mask = mask_sky & mask_lines & mask_stats & mask_conf
        
        # 필터링된 결과 반환
        filtered_kpts = kpts[final_mask]
        filtered_desc = desc[:, final_mask] if desc is not None else None
        
        return filtered_kpts, filtered_desc


def create_point_filter(frame_h: int = 480, frame_w: int = 640) -> PointFilter:
    """포인트 필터 인스턴스 생성"""
    return PointFilter(frame_h=frame_h, frame_w=frame_w)
