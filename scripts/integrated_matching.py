#!/usr/bin/env python3
"""
Integrated SuperPoint Feature Extraction & GPU-Accelerated Matching

동영상 또는 이미지 폴더를 입력으로 받아:
1. 프레임별로 SuperPoint로 특징점 추출
2. 추출되자마자 이전 프레임과 GPU로 매칭
3. 매칭 결과를 실시간으로 시각화
4. 결과를 저장 (이미지 + NPY 데이터)

한 번에 모든 것을 처리하는 통합 파이프라인입니다!
"""

import argparse
import glob
import numpy as np
import os
import time
import sys
import cv2

# 부모 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SuperPoint 모듈
from scripts.py_superpoint import SuperPointFrontend

# Matcher 모듈
from matcher_module import BTMatcher, draw_matches, compute_fundamental_matrix


class VideoProcessor:
    """비디오 또는 이미지 폴더 처리"""
    
    def __init__(self, input_path):
        self.input_path = input_path
        self.frames = []
        self.frame_idx = 0
        
        self._load_input()
    
    def _load_input(self):
        """비디오 또는 이미지 폴더에서 프레임 로드"""
        if self.input_path.endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv', '.webm')):
            # 비디오 파일
            self._load_video()
        elif os.path.isdir(self.input_path):
            # 이미지 폴더
            self._load_images()
        else:
            raise ValueError(f"지원하지 않는 입력: {self.input_path}")
    
    def _load_video(self):
        """비디오에서 프레임 추출"""
        cap = cv2.VideoCapture(self.input_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없음: {self.input_path}")
        
        print(f"비디오 로드 중: {self.input_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 그레이스케일 변환 및 정규화
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            self.frames.append(gray)
        
        cap.release()
        print(f"로드된 프레임: {len(self.frames)}개")
    
    def _load_images(self):
        """이미지 폴더에서 프레임 로드"""
        image_files = sorted(glob.glob(os.path.join(self.input_path, '*.jpg')) +
                            glob.glob(os.path.join(self.input_path, '*.png')) +
                            glob.glob(os.path.join(self.input_path, '*.bmp')))
        
        if not image_files:
            raise ValueError(f"이미지를 찾을 수 없음: {self.input_path}")
        
        print(f"이미지 로드 중: {self.input_path}")
        
        for img_file in image_files:
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if img is not None:
                self.frames.append(img)
        
        print(f"로드된 이미지: {len(self.frames)}개")
    
    def get_frame(self):
        """다음 프레임 반환"""
        if self.frame_idx >= len(self.frames):
            return None, None
        
        frame = self.frames[self.frame_idx]
        frame_id = f"frame_{self.frame_idx:05d}"
        self.frame_idx += 1
        
        return frame, frame_id


class IntegratedMatchingPipeline:
    """
    통합 매칭 파이프라인
    특징점 추출과 매칭을 동시에 수행
    """
    
    def __init__(self, output_dir='matching_results_integrated', 
                 weights_path='superpoint_v1.pth', 
                 nn_thresh=0.7, cuda=True, display=True):
        """
        Parameters
        ----------
        output_dir : str
            결과 저장 디렉토리
        weights_path : str
            SuperPoint 가중치 파일 경로
        nn_thresh : float
            매칭 거리 임계값
        cuda : bool
            GPU 사용 여부
        display : bool
            실시간 시각화 여부
        """
        self.output_dir = output_dir
        self.nn_thresh = nn_thresh
        self.display = display
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'matches_viz'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'matches_data'), exist_ok=True)
        
        # SuperPoint 초기화
        print(f"SuperPoint 로드 중 (가중치: {weights_path})...")
        self.fe = SuperPointFrontend(weights_path=weights_path, cuda=cuda)
        
        # BT-Matcher 초기화
        print(f"BT-Matcher 초기화 중...")
        self.matcher = BTMatcher(nn_thresh=nn_thresh, use_cuda=cuda, mutual=True)
        
        # 상태 변수
        self.prev_pts = None
        self.prev_desc = None
        self.prev_frame_id = None
        self.prev_img = None
    
    def process_video(self, input_path):
        """
        비디오 또는 이미지 폴더 처리
        
        Parameters
        ----------
        input_path : str
            비디오 파일 또는 이미지 폴더 경로
        """
        # 입력 로드
        processor = VideoProcessor(input_path)
        total_frames = len(processor.frames)
        
        print(f"\n{'='*60}")
        print(f"처리 시작: {total_frames} 프레임")
        print(f"{'='*60}\n")
        
        frame_count = 0
        total_time = time.time()
        
        while True:
            # 다음 프레임 가져오기
            frame, frame_id = processor.get_frame()
            if frame is None:
                break
            
            frame_count += 1
            frame_start = time.time()
            
            # 1️⃣ 특징점 추출
            pts, desc, heatmap = self.fe.run(frame)
            
            # 초기화
            matches = []
            inlier_mask = None
            
            # 2️⃣ 이전 프레임과 매칭 (있는 경우)
            if self.prev_desc is not None and pts.shape[1] > 0 and self.prev_pts.shape[1] > 0:
                # 디스크립터 형식 확인 및 변환
                desc_norm = desc.T if desc.shape[0] > desc.shape[1] else desc
                prev_desc_norm = self.prev_desc.T if self.prev_desc.shape[0] > self.prev_desc.shape[1] else self.prev_desc
                
                # GPU 매칭
                try:
                    matches = self.matcher.match(prev_desc_norm, desc_norm)
                    matches = matches if isinstance(matches, list) else matches.tolist() if hasattr(matches, 'tolist') else []
                except Exception as e:
                    print(f"매칭 오류: {e}")
                    matches = []
                
                # 기하학적 검증
                if len(matches) > 4:
                    try:
                        F, inlier_mask = compute_fundamental_matrix(
                            self.prev_pts, pts, matches, threshold=2.0
                        )
                    except Exception as e:
                        print(f"기하학 검증 오류: {e}")
                        inlier_mask = None
                
                # 3️⃣ 결과 시각화 및 저장
                if len(matches) > 0:
                    self._save_matching_result(frame_id, self.prev_frame_id, 
                                              self.prev_pts, pts, 
                                              self.prev_img, frame, 
                                              matches, inlier_mask)
            
            # 4️⃣ 특징점 시각화 이미지 생성
            self._save_frame_visualization(frame, frame_id, pts, heatmap)
            
            # 상태 업데이트
            self.prev_pts = pts
            self.prev_desc = desc
            self.prev_frame_id = frame_id
            self.prev_img = frame
            
            # 특징점 저장 (추후 참조용)
            self._save_features(frame_id, pts, desc, heatmap)
            
            # 진행 상황 출력
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            
            inlier_count = np.sum(inlier_mask) if inlier_mask is not None else 0
            
            if frame_count % 5 == 0 or frame_count == 1:
                print(f"[{frame_count:3d}/{total_frames}] {frame_id}: "
                      f"특징점={pts.shape[1]:4d}개, "
                      f"매칭={len(matches):4d}개 (Inliers: {inlier_count:3d}), "
                      f"시간={frame_time:.2f}s, FPS={fps:.1f}")
            
            # 실시간 시각화 (선택사항)
            if self.display and self.prev_desc is not None and len(matches) > 0:
                self._display_result(frame_id, matches, inlier_mask)
        
        # 최종 통계
        total_time = time.time() - total_time
        print(f"\n{'='*60}")
        print(f"처리 완료!")
        print(f"총 시간: {total_time:.2f}s")
        print(f"평균 FPS: {total_frames/total_time:.2f}")
        print(f"결과 저장: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _save_matching_result(self, frame_id, prev_frame_id, pts1, pts2, img1, img2, matches, inlier_mask):
        """매칭 결과 저장"""
        if len(matches) == 0:
            return
        
        pair_name = f"{prev_frame_id}_{frame_id}"
        
        try:
            # 데이터 저장 (딕셔너리 대신 별도 파일로 저장)
            matches_data = {
                'matches': matches,
                'inlier_mask': inlier_mask if inlier_mask is not None else np.ones(len(matches), dtype=bool),
                'frame1_id': str(prev_frame_id),
                'frame2_id': str(frame_id),
            }
            data_file = os.path.join(self.output_dir, 'matches_data', f'{pair_name}_matches.npz')
            np.savez(data_file, **matches_data)
        except Exception as e:
            print(f"데이터 저장 오류 ({pair_name}): {e}")
        
        # 이미지 시각화 저장
        try:
            match_img = draw_matches(img1, pts1, img2, pts2, matches, inlier_mask)
            img_file = os.path.join(self.output_dir, 'matches_viz', f'{pair_name}_matches.png')
            cv2.imwrite(img_file, match_img)
        except Exception as e:
            print(f"시각화 저장 오류 ({pair_name}): {e}")
    
    def _save_frame_visualization(self, frame, frame_id, pts, heatmap):
        """프레임 시각화 저장"""
        try:
            # 히트맵을 RGB로 변환
            if heatmap is not None:
                heatmap_vis = np.clip(heatmap, 0, 1)
                heatmap_vis = (heatmap_vis * 255).astype(np.uint8)
                heatmap_rgb = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
            else:
                # 특징점으로 간단한 시각화
                frame_vis = (frame * 255).astype(np.uint8)
                heatmap_rgb = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)
                
                for pt in pts.T:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(heatmap_rgb, (x, y), 2, (0, 255, 0), -1)
            
            img_file = os.path.join(self.output_dir, 'frames', f'{frame_id}_heatmap.png')
            cv2.imwrite(img_file, heatmap_rgb)
        except:
            pass
    
    def _save_features(self, frame_id, pts, desc, heatmap):
        """특징점과 디스크립터 저장"""
        try:
            np.save(os.path.join(self.output_dir, 'features', f'{frame_id}_pts.npy'), pts)
            if desc is not None:
                np.save(os.path.join(self.output_dir, 'features', f'{frame_id}_desc.npy'), desc)
            if heatmap is not None:
                np.save(os.path.join(self.output_dir, 'features', f'{frame_id}_heatmap.npy'), heatmap)
        except:
            pass
    
    def _display_result(self, frame_id, matches, inlier_mask):
        """결과 실시간 표시 (선택사항)"""
        try:
            num_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
            print(f"  → {frame_id}: {len(matches)} 매칭 발견 ({num_inliers} inliers)")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='SuperPoint 특징점 추출 + GPU 매칭 통합 파이프라인'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력: 비디오 파일 (mp4, avi 등) 또는 이미지 폴더'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='matching_results_integrated',
        help='결과 저장 디렉토리 (기본값: matching_results_integrated)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='superpoint_v1.pth',
        help='SuperPoint 가중치 파일 경로'
    )
    parser.add_argument(
        '--nn_thresh',
        type=float,
        default=0.7,
        help='매칭 거리 임계값 (기본값: 0.7)'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='GPU 사용하지 않음 (CPU만 사용)'
    )
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='실시간 시각화 비활성화'
    )
    
    args = parser.parse_args()
    
    # 입력 경로 검증
    if not os.path.exists(args.input):
        print(f"❌ 입력 경로를 찾을 수 없음: {args.input}")
        return 1
    
    # 가중치 파일 검증
    if not os.path.exists(args.weights):
        print(f"❌ 가중치 파일을 찾을 수 없음: {args.weights}")
        return 1
    
    try:
        # 파이프라인 실행
        pipeline = IntegratedMatchingPipeline(
            output_dir=args.output,
            weights_path=args.weights,
            nn_thresh=args.nn_thresh,
            cuda=not args.no_cuda,
            display=not args.no_display
        )
        
        pipeline.process_video(args.input)
        
        return 0
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
