"""
GPU 기반 BT-Matcher (Batch Topk Matcher) 구현
논문: "BT-Matcher: Faster Feature Matching with Batch Topk and Point-to-line Metric"

GPU를 활용하여 대규모 특징점 매칭을 빠르고 효율적으로 수행합니다.
"""
import torch
import numpy as np
import cv2

class BTMatcher:
    def __init__(self, nn_thresh=0.7, use_cuda=True, mutual=False, ratio_thresh=0.85):
        """
        :param nn_thresh: Descriptor 거리 임계값 (0.7 추천)
        :param use_cuda: CUDA 사용 여부
        :param mutual: 상호 매칭(Mutual Check) 사용 여부
        :param ratio_thresh: Lowe's Ratio Test 임계값 (0.85 추천, 1.0이면 비활성화)
        """
        self.nn_thresh = nn_thresh
        self.mutual = mutual
        self.ratio_thresh = ratio_thresh
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def match(self, desc1, desc2):
        """
        [수정됨] 불필요한 kpts 인자를 제거하고 desc1, desc2만 받습니다.
        Args:
            desc1: [256, N1] (Descriptor)
            desc2: [256, N2]
        Returns:
            matches: [M, 2] numpy array
        """
        # 1. 예외 처리: 디스크립터가 없으면 빈 배열 반환
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.empty((0, 2), dtype=int)

        # 2. Numpy -> Tensor 변환 및 Device 이동
        if isinstance(desc1, np.ndarray):
            d1 = torch.from_numpy(desc1).float().to(self.device)
        else:
            d1 = desc1.float().to(self.device)
            
        if isinstance(desc2, np.ndarray):
            d2 = torch.from_numpy(desc2).float().to(self.device)
        else:
            d2 = desc2.float().to(self.device)

        # ==========================================================
        # ★ 차원 전치 (Transpose) ★
        # SuperPoint 출력 [256, N] -> 거리 계산을 위해 [N, 256]으로 변경
        # ==========================================================
        d1 = d1.t()
        d2 = d2.t()

        # 3. 거리 행렬 계산 (Euclidean Distance)
        try:
            dist_mat = torch.cdist(d1, d2, p=2) # [N1, N2]
        except RuntimeError:
            # VRAM 부족 시 CPU로 폴백
            d1 = d1.cpu()
            d2 = d2.cpu()
            dist_mat = torch.cdist(d1, d2, p=2)

        # 4. Nearest Neighbor Search (Lowe's Ratio Test를 위해 2개 추출)
        if self.ratio_thresh < 1.0 and d2.shape[0] >= 2:
            top_dist, top_idxs = torch.topk(dist_mat, k=2, dim=1, largest=False)
            min_dist = top_dist[:, 0]
            idxs = top_idxs[:, 0]
            ratio_mask = (min_dist < self.ratio_thresh * top_dist[:, 1])
        else:
            min_dist, idxs = torch.min(dist_mat, dim=1)
            ratio_mask = torch.ones_like(min_dist, dtype=torch.bool)

        # 5. 매칭 필터링
        if self.mutual:
            # 상호 매칭 (Mutual Check)
            min_dist2, idxs2 = torch.min(dist_mat, dim=0)
            match_check = (idxs2[idxs] == torch.arange(d1.shape[0], device=d1.device))
            valid_mask = (min_dist < self.nn_thresh) & match_check & ratio_mask
        else:
            # 단순 거리 임계값
            valid_mask = (min_dist < self.nn_thresh) & ratio_mask

        # 6. 결과 인덱스 추출
        idx1 = torch.arange(d1.shape[0], device=d1.device)[valid_mask]
        idx2 = idxs[valid_mask]

        matches = torch.stack([idx1, idx2], dim=1)
        
        return matches.cpu().numpy().astype(int)

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