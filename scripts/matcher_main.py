#!/usr/bin/env python3
"""
SuperPoint Feature Matching Program with GPU-accelerated BT-Matcher

이 프로그램은 다음 기능을 수행합니다:
1. py_superpoint.py에서 추출한 특징점과 디스크립터 로드
2. GPU 기반 BT-Matcher를 사용하여 매칭 수행
3. RANSAC을 이용한 기하학적 검증
4. 매칭 결과 시각화 및 저장
"""

import argparse
import os
import glob
import numpy as np
import cv2
import time
from pathlib import Path
import sys

# 부모 디렉토리를 Python 경로에 추가 (matcher_module import 위함)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Matcher 모듈 import
from matcher_module import BTMatcher, draw_matches, compute_fundamental_matrix, compute_homography


class FeatureMatchingPipeline:
    """
    SuperPoint 특징점 매칭 파이프라인
    """
    
    def __init__(self, npy_dir, output_dir, nn_thresh=0.7, use_geometric_test=True):
        """
        파이프라인 초기화
        
        Parameters
        ----------
        npy_dir : str
            특징점과 디스크립터가 저장된 디렉토리
        output_dir : str
            결과를 저장할 디렉토리
        nn_thresh : float
            매칭 거리 임계값
        use_geometric_test : bool
            기하학적 검증 수행 여부
        """
        self.npy_dir = npy_dir
        self.output_dir = output_dir
        self.nn_thresh = nn_thresh
        self.use_geometric_test = use_geometric_test
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'matches_viz'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'matches_data'), exist_ok=True)
        
        # GPU 기반 Matcher 초기화
        self.matcher = BTMatcher(nn_thresh=nn_thresh, use_cuda=True, mutual=True)
        
        # 특징점 데이터 로드
        self.points_data = self._load_features()
    
    def _load_features(self):
        """
        NPY 파일들로부터 특징점과 디스크립터 로드
        
        Returns
        -------
        dict
            frame_id -> {'pts': (3xN), 'desc': (DxN), 'heatmap': (HxW)}
        """
        features = {}
        
        # .npy 파일 목록 정렬
        npy_files = sorted(glob.glob(os.path.join(self.npy_dir, '*_pts.npy')))
        
        print(f"Found {len(npy_files)} feature files")
        
        for pts_file in npy_files:
            # frame ID 추출 (예: frame_00001_pts.npy -> frame_00001)
            basename = os.path.basename(pts_file)
            frame_id = '_'.join(basename.split('_')[:-1])
            
            # 관련 파일들 경로
            desc_file = pts_file.replace('_pts.npy', '_desc.npy')
            heatmap_file = pts_file.replace('_pts.npy', '_heatmap.npy')
            
            # 파일 존재 확인 및 로드
            try:
                pts = np.load(pts_file)
                desc = np.load(desc_file) if os.path.exists(desc_file) else None
                heatmap = np.load(heatmap_file) if os.path.exists(heatmap_file) else None
                
                # 데이터 형식 검증
                if desc is not None and desc.shape[1] != pts.shape[1]:
                    # desc가 (DxN) 형식인지 확인, 아니면 전치
                    if desc.shape[0] == pts.shape[1]:
                        desc = desc.T
                
                features[frame_id] = {
                    'pts': pts,
                    'desc': desc,
                    'heatmap': heatmap,
                    'pts_file': pts_file
                }
            except Exception as e:
                print(f"Error loading {frame_id}: {e}")
                continue
        
        return features
    
    def match_consecutive_frames(self):
        """
        연속된 프레임 간의 매칭 수행
        """
        frame_ids = sorted(self.points_data.keys())
        
        if len(frame_ids) < 2:
            print("Not enough frames for matching")
            return
        
        print(f"\nMatching {len(frame_ids) - 1} consecutive frame pairs...")
        
        for i in range(len(frame_ids) - 1):
            frame1_id = frame_ids[i]
            frame2_id = frame_ids[i + 1]
            
            self._match_pair(frame1_id, frame2_id)
    
    def match_specific_pair(self, frame1_id, frame2_id):
        """
        특정 두 프레임 간의 매칭 수행
        """
        self._match_pair(frame1_id, frame2_id)
    
    def _match_pair(self, frame1_id, frame2_id):
        """
        두 프레임 간의 매칭 수행 (내부 함수)
        """
        print(f"\nMatching {frame1_id} -> {frame2_id}...", end=' ')
        
        start_time = time.time()
        
        # 특징점과 디스크립터 추출
        data1 = self.points_data.get(frame1_id)
        data2 = self.points_data.get(frame2_id)
        
        if data1 is None or data2 is None:
            print(f"ERROR: Missing data for {frame1_id} or {frame2_id}")
            return
        
        pts1, desc1 = data1['pts'], data1['desc']
        pts2, desc2 = data2['pts'], data2['desc']
        
        # 특징점 검증
        if pts1.shape[1] == 0 or pts2.shape[1] == 0:
            print(f"SKIP: No features detected")
            return
        
        if desc1 is None or desc2 is None:
            print(f"ERROR: Missing descriptors")
            return
        
        # 디스크립터 형식 표준화 (DxN -> NxD)
        if desc1.shape[0] > desc1.shape[1]:
            desc1 = desc1.T
        if desc2.shape[0] > desc2.shape[1]:
            desc2 = desc2.T
        
        # GPU 기반 매칭 수행
        try:
            matches = self.matcher.match(desc1, desc2)
        except Exception as e:
            print(f"ERROR during matching: {e}")
            return
        
        matching_time = time.time() - start_time
        
        print(f"{len(matches)} matches found ({matching_time:.3f}s)")
        
        if len(matches) == 0:
            return
        
        # 기하학적 검증
        inlier_mask = None
        if self.use_geometric_test:
            try:
                F, inlier_mask = compute_fundamental_matrix(pts1, pts2, matches)
                if F is not None:
                    num_inliers = np.sum(inlier_mask)
                    print(f"  Geometric test: {num_inliers}/{len(matches)} inliers")
            except:
                pass
        
        # 결과 저장
        self._save_results(frame1_id, frame2_id, pts1, pts2, matches, inlier_mask, data1, data2)
    
    def _save_results(self, frame1_id, frame2_id, pts1, pts2, matches, inlier_mask, data1, data2):
        """
        매칭 결과 저장
        """
        pair_name = f"{frame1_id}_{frame2_id}"
        
        # 1. 매칭 데이터 저장 (NPY)
        data_file = os.path.join(self.output_dir, 'matches_data', f'{pair_name}_matches.npy')
        np.save(data_file, {
            'matches': matches,
            'inlier_mask': inlier_mask,
            'frame1_id': frame1_id,
            'frame2_id': frame2_id,
        })
        
        # 2. 매칭 시각화
        self._visualize_matches(frame1_id, frame2_id, pts1, pts2, matches, inlier_mask, data1, data2)
    
    def _visualize_matches(self, frame1_id, frame2_id, pts1, pts2, matches, inlier_mask, data1, data2):
        """
        매칭을 시각화하고 저장
        """
        pair_name = f"{frame1_id}_{frame2_id}"
        
        # 이미지 로드
        img1_path = data1['pts_file'].replace('_pts.npy', '.png')
        img2_path = data2['pts_file'].replace('_pts.npy', '.png')
        
        # 이미지 찾기 (같은 디렉토리에 없으면 원본 이미지 디렉토리에서)
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            # 히트맵 또는 특징점으로 이미지 생성
            img1 = self._create_image_from_heatmap(data1)
            img2 = self._create_image_from_heatmap(data2)
        else:
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE) / 255.0
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # 매칭 시각화
        match_img = draw_matches(img1, pts1, img2, pts2, matches, inlier_mask)
        
        # 저장
        output_path = os.path.join(self.output_dir, 'matches_viz', f'{pair_name}_matches.png')
        cv2.imwrite(output_path, match_img)
    
    def _create_image_from_heatmap(self, data):
        """
        히트맵으로부터 이미지 생성
        """
        if data['heatmap'] is not None:
            heatmap = data['heatmap']
            # 정규화
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            return heatmap
        else:
            # 특징점으로부터 간단한 이미지 생성
            pts = data['pts']
            if pts.shape[1] > 0:
                H = int(pts[1].max()) + 10
                W = int(pts[0].max()) + 10
            else:
                H = W = 480
            img = np.zeros((H, W), dtype=np.float32)
            for pt in pts.T:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < W and 0 <= y < H:
                    img[y, x] = 1.0
            return img


def main():
    parser = argparse.ArgumentParser(
        description='SuperPoint Feature Matching with GPU-accelerated BT-Matcher'
    )
    
    parser.add_argument(
        '--npy_dir',
        type=str,
        default='results_npy',
        help='Directory containing feature NPY files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='matching_results',
        help='Directory to save matching results'
    )
    parser.add_argument(
        '--nn_thresh',
        type=float,
        default=0.7,
        help='Nearest neighbor distance threshold'
    )
    parser.add_argument(
        '--no_geometric_test',
        action='store_true',
        help='Skip geometric verification'
    )
    parser.add_argument(
        '--frame_pair',
        type=str,
        default=None,
        help='Match specific frame pair (format: frame_00001:frame_00002)'
    )
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = FeatureMatchingPipeline(
        npy_dir=args.npy_dir,
        output_dir=args.output_dir,
        nn_thresh=args.nn_thresh,
        use_geometric_test=not args.no_geometric_test
    )
    
    # 매칭 수행
    if args.frame_pair:
        frame_ids = args.frame_pair.split(':')
        if len(frame_ids) == 2:
            pipeline.match_specific_pair(frame_ids[0], frame_ids[1])
        else:
            print("Invalid frame pair format")
    else:
        pipeline.match_consecutive_frames()
    
    print(f"\nMatching complete. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
