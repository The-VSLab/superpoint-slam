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
