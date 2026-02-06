"""
특징점 매칭 유틸리티 함수
매칭 결과 처리, 시각화 및 검증 관련 함수들
"""

import numpy as np
import cv2


def draw_matches(img1, pts1, img2, pts2, matches, status=None):
    """
    두 이미지 간의 매칭을 시각화합니다.
    
    Parameters
    ----------
    img1 : numpy.ndarray
        첫 번째 이미지 (H x W) 또는 (H x W x 3)
    pts1 : numpy.ndarray
        첫 번째 이미지의 특징점 (N x 2 또는 3 x N)
    img2 : numpy.ndarray
        두 번째 이미지 (H x W) 또는 (H x W x 3)
    pts2 : numpy.ndarray
        두 번째 이미지의 특징점 (M x 2 또는 3 x M)
    matches : numpy.ndarray
        매칭 결과 (L x 3) [idx1, idx2, distance]
    status : numpy.ndarray, optional
        매칭 상태 마스크 (L,) - True면 inlier
        
    Returns
    -------
    output : numpy.ndarray
        시각화된 이미지
    """
    try:
        # 이미지가 None이면 기본값 사용
        if img1 is None or img2 is None:
            return None
        
        # 이미지가 흑백이면 3채널로 변환
        if img1.ndim == 2:
            img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
        
        if img2.ndim == 2:
            img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            img2 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
        
        # 특징점 형태 변환 (3xN -> Nx2)
        if pts1.shape[0] == 2 or pts1.shape[0] == 3:
            pts1 = pts1[:2, :].T
        if pts2.shape[0] == 2 or pts2.shape[0] == 3:
            pts2 = pts2[:2, :].T
        
        # 특징점 개수 확인
        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=img1.dtype)
            output[:h1, :w1] = img1
            output[:h2, w1:w1+w2] = img2
            return output
        
        # 출력 이미지 생성
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=img1.dtype)
        output[:h1, :w1] = img1
        output[:h2, w1:w1+w2] = img2
        
        # 매칭 선 그리기
        for i, match in enumerate(matches):
            try:
                idx1 = int(match[0])
                idx2 = int(match[1])
                
                # 인덱스 범위 확인
                if idx1 >= pts1.shape[0] or idx2 >= pts2.shape[0]:
                    continue
                
                # 특징점 좌표
                pt1 = tuple(map(int, pts1[idx1]))
                pt2 = tuple(map(int, pts2[idx2]))
                
                # 색상 선택 (inlier/outlier)
                if status is not None and i < len(status):
                    color = (0, 255, 0) if status[i] else (0, 0, 255)
                else:
                    color = (200, 200, 0)  # Default cyan
                
                # 선 그리기
                pt2_adjusted = (pt2[0] + w1, pt2[1])
                cv2.line(output, pt1, pt2_adjusted, color, 1)
                
                # 점 표시
                cv2.circle(output, pt1, 3, color, -1)
                cv2.circle(output, pt2_adjusted, 3, color, -1)
            except:
                continue
        
        return output
    except Exception as e:
        print(f"draw_matches 오류: {e}")
        return None


def filter_matches_by_distance(matches, distance_threshold=None, percentile=75):
    """
    매칭을 거리로 필터링합니다.
    
    Parameters
    ----------
    matches : numpy.ndarray
        매칭 결과 (L x 3) [idx1, idx2, distance]
    distance_threshold : float, optional
        거리 임계값. None이면 백분위수 사용
    percentile : float
        사용할 백분위수 (0-100)
        
    Returns
    -------
    filtered_matches : numpy.ndarray
        필터링된 매칭
    """
    if matches.shape[0] == 0:
        return matches
    
    distances = matches[:, 2]
    
    if distance_threshold is None:
        distance_threshold = np.percentile(distances, percentile)
    
    valid_mask = distances <= distance_threshold
    return matches[valid_mask]


def compute_fundamental_matrix(pts1, pts2, matches, method=cv2.FM_RANSAC, threshold=1.0):
    """
    RANSAC을 사용하여 기본 행렬(Fundamental Matrix)을 계산합니다.
    
    Parameters
    ----------
    pts1 : numpy.ndarray
        첫 번째 이미지의 특징점 (N x 2 또는 3 x N)
    pts2 : numpy.ndarray
        두 번째 이미지의 특징점 (M x 2 또는 3 x M)
    matches : numpy.ndarray
        매칭 결과 (L x 3) [idx1, idx2, distance]
    method : int
        cv2.FM_RANSAC 또는 cv2.FM_LMEDS
    threshold : float
        RANSAC 임계값
        
    Returns
    -------
    F : numpy.ndarray
        기본 행렬 (3 x 3)
    mask : numpy.ndarray
        inlier 마스크
    """
    # 특징점 형태 변환
    if pts1.shape[0] == 2 or pts1.shape[0] == 3:
        pts1 = pts1[:2, :].T
    if pts2.shape[0] == 2 or pts2.shape[0] == 3:
        pts2 = pts2[:2, :].T
    
    # 매칭된 점 추출
    matched_pts1 = pts1[matches[:, 0].astype(int)]
    matched_pts2 = pts2[matches[:, 1].astype(int)]
    
    if len(matched_pts1) < 8:
        return None, None
    
    # 기본 행렬 계산
    F, mask = cv2.findFundamentalMat(
        matched_pts1.astype(np.float32),
        matched_pts2.astype(np.float32),
        method=method,
        threshold=threshold,
        confidence=0.99
    )
    
    return F, mask.ravel().astype(bool)


def compute_homography(pts1, pts2, matches, method=cv2.RANSAC, threshold=5.0):
    """
    RANSAC을 사용하여 호모그래피(Homography) 행렬을 계산합니다.
    
    Parameters
    ----------
    pts1 : numpy.ndarray
        첫 번째 이미지의 특징점 (N x 2 또는 3 x N)
    pts2 : numpy.ndarray
        두 번째 이미지의 특징점 (M x 2 또는 3 x M)
    matches : numpy.ndarray
        매칭 결과 (L x 3) [idx1, idx2, distance]
    method : int
        cv2.RANSAC 또는 cv2.LMEDS
    threshold : float
        RANSAC 임계값
        
    Returns
    -------
    H : numpy.ndarray
        호모그래피 행렬 (3 x 3)
    mask : numpy.ndarray
        inlier 마스크
    """
    # 특징점 형태 변환
    if pts1.shape[0] == 2 or pts1.shape[0] == 3:
        pts1 = pts1[:2, :].T
    if pts2.shape[0] == 2 or pts2.shape[0] == 3:
        pts2 = pts2[:2, :].T
    
    # 매칭된 점 추출
    matched_pts1 = pts1[matches[:, 0].astype(int)]
    matched_pts2 = pts2[matches[:, 1].astype(int)]
    
    if len(matched_pts1) < 4:
        return None, None
    
    # 호모그래피 계산
    H, mask = cv2.findHomography(
        matched_pts1.astype(np.float32),
        matched_pts2.astype(np.float32),
        method=method,
        ransacReprojThreshold=threshold
    )
    
    return H, mask.ravel().astype(bool)


def evaluate_matches(pts1, pts2, matches, F=None, H=None):
    """
    매칭 품질을 평가합니다.
    
    Parameters
    ----------
    pts1 : numpy.ndarray
        첫 번째 이미지의 특징점
    pts2 : numpy.ndarray
        두 번째 이미지의 특징점
    matches : numpy.ndarray
        매칭 결과 (L x 3)
    F : numpy.ndarray, optional
        기본 행렬
    H : numpy.ndarray, optional
        호모그래피 행렬
        
    Returns
    -------
    stats : dict
        매칭 통계
    """
    stats = {
        'num_matches': matches.shape[0],
        'avg_distance': np.mean(matches[:, 2]),
        'std_distance': np.std(matches[:, 2]),
        'min_distance': np.min(matches[:, 2]) if matches.shape[0] > 0 else 0,
        'max_distance': np.max(matches[:, 2]) if matches.shape[0] > 0 else 0,
    }
    
    return stats


def save_matches(filename, pts1, pts2, matches):
    """
    매칭 결과를 파일에 저장합니다.
    
    Parameters
    ----------
    filename : str
        저장할 파일 경로
    pts1 : numpy.ndarray
        첫 번째 이미지의 특징점
    pts2 : numpy.ndarray
        두 번째 이미지의 특징점
    matches : numpy.ndarray
        매칭 결과
    """
    # 특징점 형태 표준화
    if pts1.shape[0] == 3:
        pts1 = pts1[:2, :].T
    elif pts1.shape[0] > pts1.shape[1]:
        pts1 = pts1[:, :2]
    else:
        pts1 = pts1.T
        
    if pts2.shape[0] == 3:
        pts2 = pts2[:2, :].T
    elif pts2.shape[0] > pts2.shape[1]:
        pts2 = pts2[:, :2]
    else:
        pts2 = pts2.T
    
    # 데이터 저장
    data = {
        'pts1': pts1,
        'pts2': pts2,
        'matches': matches,
    }
    np.save(filename, data)
