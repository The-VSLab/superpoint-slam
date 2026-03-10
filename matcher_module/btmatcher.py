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
        # CUDA만 GPU 매칭 활성화 (MPS는 소규모 행렬 연산 시 커널 디스패치 오버헤드로 CPU보다 느림)
        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def match(self, desc1, desc2):
        """
        GPU 텐서를 직접 받으면 CPU 전환 없이 GPU에서 매칭 수행.
        numpy 입력도 호환됩니다.
        Args:
            desc1: [256, N1] (Descriptor) — numpy array 또는 torch.Tensor
            desc2: [256, N2]
        Returns:
            matches: [M, 2] numpy array
        """
        # 1. 예외 처리: 디스크립터가 없으면 빈 배열 반환
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.empty((0, 2), dtype=int)

        # 2. Tensor 변환 (numpy면 변환, 이미 tensor면 dtype만 맞춤)
        if isinstance(desc1, np.ndarray):
            d1 = torch.from_numpy(desc1).float()
        else:
            d1 = desc1.float()

        if isinstance(desc2, np.ndarray):
            d2 = torch.from_numpy(desc2).float()
        else:
            d2 = desc2.float()

        # 3. 연산 디바이스 결정: CUDA 텐서가 있으면 GPU에서 직통 매칭
        #    (MPS는 소규모 행렬에서 커널 오버헤드가 커서 CPU가 빠름)
        if d1.device.type == 'cuda':
            device = d1.device
        elif d2.device.type == 'cuda':
            device = d2.device
        else:
            device = self.device

        d1 = d1.to(device)
        d2 = d2.to(device)

        # ★ 차원 전치: SuperPoint 출력 [256, N] → 거리 계산용 [N, 256]
        d1 = d1.t()
        d2 = d2.t()

        # 4. 코사인 유사도 → 유클리드 거리 제곱 변환
        sim_mat = torch.mm(d1, d2.t())
        dist_sq = torch.clamp(2.0 - 2.0 * sim_mat, min=0.0)

        # 5. Nearest Neighbor Search (Lowe's Ratio Test)
        if self.ratio_thresh < 1.0 and d2.shape[0] >= 2:
            top_dist_sq, top_idxs = torch.topk(dist_sq, k=2, dim=1, largest=False)
            min_dist_sq = top_dist_sq[:, 0]
            idxs = top_idxs[:, 0]
            ratio_sq = self.ratio_thresh ** 2
            ratio_mask = (min_dist_sq < ratio_sq * top_dist_sq[:, 1])
        else:
            min_dist_sq, idxs = torch.min(dist_sq, dim=1)
            ratio_mask = torch.ones_like(min_dist_sq, dtype=torch.bool)

        # 6. 매칭 필터링
        nn_thresh_sq = self.nn_thresh ** 2

        if self.mutual:
            min_dist_sq2, idxs2 = torch.min(dist_sq, dim=0)
            match_check = (idxs2[idxs] == torch.arange(d1.shape[0], device=device))
            valid_mask = (min_dist_sq < nn_thresh_sq) & match_check & ratio_mask
        else:
            valid_mask = (min_dist_sq < nn_thresh_sq) & ratio_mask

        # 7. 결과 인덱스 추출 → CPU numpy 반환 (OpenCV/SLAM 소비용)
        idx1 = torch.arange(d1.shape[0], device=device)[valid_mask]
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