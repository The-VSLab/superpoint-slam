"""
GPU 기반 BT-Matcher (Batch Topk Matcher) 구현
논문: "BT-Matcher: Faster Feature Matching with Batch Topk and Point-to-line Metric"

GPU를 활용하여 대규모 특징점 매칭을 빠르고 효율적으로 수행합니다.
"""

import numpy as np
import torch
import torch.nn.functional as F


class BTMatcher:
    """
    GPU 기반 Batch Top-k Matcher
    
    두 이미지의 특징점 디스크립터를 입력받아 GPU에서 빠른 매칭을 수행합니다.
    양방향 매칭 검증을 통해 신뢰도 높은 매칭만 반환합니다.
    
    Parameters
    ----------
    nn_thresh : float
        매칭 거리 임계값 (기본값: 0.7)
    use_cuda : bool
        CUDA 사용 여부 (기본값: True)
    mutual : bool
        양방향 매칭 검증 수행 여부 (기본값: True)
    """
    
    def __init__(self, nn_thresh=0.7, use_cuda=True, mutual=True):
        self.nn_thresh = nn_thresh
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.mutual = mutual
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
    
    def match(self, desc1, desc2):
        """
        두 이미지의 디스크립터 간 매칭을 수행합니다.
        
        Parameters
        ----------
        desc1 : numpy.ndarray
            첫 번째 이미지의 디스크립터 (N x D)
            N: 특징점 개수, D: 디스크립터 차원
        desc2 : numpy.ndarray
            두 번째 이미지의 디스크립터 (M x D)
            
        Returns
        -------
        matches : numpy.ndarray
            매칭 결과 (L x 3)
            각 행: [idx1, idx2, distance]
            L: 매칭된 점의 개수
        """
        # 입력 검증
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        if desc1.shape[1] != desc2.shape[1]:
            raise ValueError("Descriptors must have the same dimension")
        
        # numpy -> torch 변환
        desc1_torch = torch.from_numpy(desc1).float().to(self.device)
        desc2_torch = torch.from_numpy(desc2).float().to(self.device)
        
        # L2 거리 계산
        # dmat[i, j] = ||desc1[i] - desc2[j]||^2
        dmat = self._compute_distance_matrix(desc1_torch, desc2_torch)
        
        # 단방향 매칭
        matches_1to2 = self._find_nn_matches(dmat)
        
        if self.mutual:
            # 양방향 매칭 검증
            dmat_reverse = dmat.t()
            matches_2to1 = self._find_nn_matches(dmat_reverse)
            
            # 양방향 일치 확인
            mutual_matches = self._filter_mutual_matches(
                matches_1to2, matches_2to1
            )
        else:
            mutual_matches = matches_1to2
        
        # torch -> numpy 변환
        matches = mutual_matches.cpu().numpy()
        
        return matches
    
    def _compute_distance_matrix(self, desc1, desc2):
        """
        두 디스크립터 세트 간의 L2 거리 행렬을 계산합니다.
        
        desc1: (N, D)
        desc2: (M, D)
        returns: (N, M) 거리 행렬
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        
        # L2 노름 계산
        norms1 = torch.norm(desc1, p=2, dim=1, keepdim=True)  # (N, 1)
        norms2 = torch.norm(desc2, p=2, dim=1, keepdim=True)  # (M, 1)
        
        # 코사인 유사도 계산 (정규화된 디스크립터의 경우)
        dot_product = torch.mm(desc1, desc2.t())  # (N, M)
        
        # 거리 = sqrt(2 - 2*cosine_similarity)
        # 하지만 L2 거리를 직접 계산하는 것이 더 정확함
        dmat = torch.cdist(desc1, desc2, p=2)  # (N, M)
        
        return dmat
    
    def _find_nn_matches(self, dmat):
        """
        거리 행렬에서 각 행의 최근접 이웃을 찾습니다.
        
        dmat: (N, M) 거리 행렬
        returns: (K, 3) 매칭 배열 [idx1, idx2, distance]
        """
        # 각 행에서 최소 거리 찾기
        min_dists, min_indices = torch.min(dmat, dim=1)
        
        # 임계값으로 필터링
        valid_mask = min_dists < self.nn_thresh
        
        # 매칭 결과 구성
        valid_indices1 = torch.where(valid_mask)[0]
        valid_indices2 = min_indices[valid_mask]
        valid_dists = min_dists[valid_mask]
        
        # (K, 3) 형태로 변환
        matches = torch.stack([
            valid_indices1.float(),
            valid_indices2.float(),
            valid_dists
        ], dim=1)
        
        return matches
    
    def _filter_mutual_matches(self, matches_1to2, matches_2to1):
        """
        양방향 매칭만 유지합니다.
        matches_1to2: (K1, 3) [idx1, idx2, dist]
        matches_2to1: (K2, 3) [idx2, idx1, dist]
        returns: (K, 3) 양방향 매칭
        """
        if matches_1to2.shape[0] == 0 or matches_2to1.shape[0] == 0:
            return torch.zeros((0, 3), device=self.device, dtype=matches_1to2.dtype)
        
        # dictionary 형성
        dict_1to2 = {}
        for i in range(matches_1to2.shape[0]):
            idx1 = int(matches_1to2[i, 0].item())
            idx2 = int(matches_1to2[i, 1].item())
            dist = matches_1to2[i, 2].item()
            dict_1to2[idx1] = (idx2, dist)
        
        # 양방향 확인
        mutual = []
        for i in range(matches_2to1.shape[0]):
            idx2 = int(matches_2to1[i, 0].item())
            idx1 = int(matches_2to1[i, 1].item())
            
            if idx1 in dict_1to2:
                matched_idx2, dist1to2 = dict_1to2[idx1]
                if matched_idx2 == idx2:
                    # 양방향 일치 확인
                    dist2to1 = matches_2to1[i, 2].item()
                    avg_dist = (dist1to2 + dist2to1) / 2.0
                    mutual.append([float(idx1), float(idx2), avg_dist])
        
        if len(mutual) == 0:
            return torch.zeros((0, 3), device=self.device, dtype=matches_1to2.dtype)
        
        mutual_tensor = torch.tensor(mutual, device=self.device, dtype=matches_1to2.dtype)
        return mutual_tensor


def match_features(desc1, desc2, nn_thresh=0.7, use_cuda=True, mutual=True):
    """
    두 이미지의 특징점 매칭 수행 (간단한 함수형 인터페이스)
    
    Parameters
    ----------
    desc1 : numpy.ndarray
        첫 번째 이미지의 디스크립터 (N x D)
    desc2 : numpy.ndarray
        두 번째 이미지의 디스크립터 (M x D)
    nn_thresh : float
        매칭 거리 임계값
    use_cuda : bool
        GPU 사용 여부
    mutual : bool
        양방향 매칭 검증 여부
        
    Returns
    -------
    matches : numpy.ndarray
        매칭 결과 (L x 3)
    """
    matcher = BTMatcher(nn_thresh=nn_thresh, use_cuda=use_cuda, mutual=mutual)
    return matcher.match(desc1, desc2)
