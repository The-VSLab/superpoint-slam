"""
GPU 기반 BT-Matcher (Batch Topk Matcher) 구현
논문: "BT-Matcher: Faster Feature Matching with Batch Topk and Point-to-line Metric"

GPU를 활용하여 대규모 특징점 매칭을 빠르고 효율적으로 수행합니다.
"""
import torch
import numpy as np
import cv2

class BTMatcher:
    def __init__(self, nn_thresh=0.7, use_cuda=True, mutual=False):
        self.nn_thresh = nn_thresh
        self.mutual = mutual
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    def match(self, desc1, desc2):
        """FP16 가속을 이용한 디스크립터 매칭"""
        if desc1.shape[1] == 0 or desc2.shape[1] == 0: return np.empty((0, 2), dtype=int)

        d1 = torch.from_numpy(desc1).to(self.device).t().half() # FP16 변환
        d2 = torch.from_numpy(desc2).to(self.device).t().half()

        with torch.no_grad():
            dist = torch.cdist(d1, d2, p=2) # [N1, N2] 거리 행렬
            min_dist, idxs = torch.min(dist, dim=1)
            
            mask = min_dist < self.nn_thresh
            if self.mutual:
                min_dist2, idxs2 = torch.min(dist, dim=0)
                mask &= (idxs2[idxs] == torch.arange(len(d1), device=self.device))
            
        idx1 = torch.arange(len(d1), device=self.device)[mask]
        idx2 = idxs[mask]
        return torch.stack([idx1, idx2], dim=1).cpu().numpy().astype(int)
# ==============================================================================
# 래퍼(Wrapper) 함수들
# ==============================================================================

def match_features(kpts1, desc1, kpts2, desc2, nn_thresh=0.7, use_cuda=True, mutual=False):
    """
    호환성을 위한 래퍼 함수. 
    4개의 인자를 받지만, 실제 매칭에는 desc1, desc2만 사용하여 클래스 메서드를 호출합니다.
    """
    matcher = BTMatcher(nn_thresh=nn_thresh, use_cuda=use_cuda, mutual=mutual)
    # kpts1, kpts2는 사용하지 않고 desc1, desc2만 전달
    return matcher.match(desc1, desc2)

def compute_fundamental_matrix(matches, kpts1, kpts2):
    if matches.shape[0] < 8:
        return None, []

    p1 = kpts1[:2, matches[:, 0]].T
    p2 = kpts2[:2, matches[:, 1]].T

    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return F, []
    return F, mask.ravel().astype(bool)

def draw_matches(img1, img2, kpts1, kpts2, matches, color=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    if len(img1.shape) == 2: img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
    out[:h1, :w1] = img1
    out[:h2, w1:w1+w2] = img2

    for m in matches:
        pt1 = (int(kpts1[0, m[0]]), int(kpts1[1, m[0]]))
        pt2 = (int(kpts2[0, m[1]] + w1), int(kpts2[1, m[1]]))
        c = color if color else np.random.randint(0, 255, 3).tolist()
        cv2.line(out, pt1, pt2, c, 1, cv2.LINE_AA)
        cv2.circle(out, pt1, 2, c, -1)
        cv2.circle(out, pt2, 2, c, -1)
        
    return out