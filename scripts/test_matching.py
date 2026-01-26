#!/usr/bin/env python3
"""
Quick test script for SuperPoint Feature Matching

이 스크립트는 matcher_module이 제대로 설치되었는지 확인하고,
기본적인 매칭이 동작하는지 테스트합니다.
"""

import numpy as np
import sys
import os
import time

# 부모 디렉토리를 Python 경로에 추가 (matcher_module import 위함)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_matcher_import():
    """Matcher 모듈 import 테스트"""
    print("=" * 60)
    print("1. Testing Matcher Module Import")
    print("=" * 60)
    
    try:
        from matcher_module import BTMatcher, match_features
        print("✓ BTMatcher imported successfully")
        print("✓ match_features imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import matcher_module: {e}")
        return False


def test_gpu_availability():
    """GPU 가용성 테스트"""
    print("\n" + "=" * 60)
    print("2. Testing GPU Availability")
    print("=" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("⚠ CUDA is not available (will use CPU)")
        
        return cuda_available
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def test_basic_matching():
    """기본 매칭 테스트"""
    print("\n" + "=" * 60)
    print("3. Testing Basic Feature Matching")
    print("=" * 60)
    
    try:
        from matcher_module import BTMatcher
        import torch
        
        # 테스트용 무작위 디스크립터 생성
        print("Creating random descriptors...")
        desc1 = np.random.randn(100, 256).astype(np.float32)
        desc2 = np.random.randn(120, 256).astype(np.float32)
        
        # 정규화
        desc1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
        desc2 = desc2 / np.linalg.norm(desc2, axis=1, keepdims=True)
        
        # 매칭 수행
        print("Performing matching...")
        start = time.time()
        matcher = BTMatcher(nn_thresh=0.7, use_cuda=torch.cuda.is_available())
        matches = matcher.match(desc1, desc2)
        elapsed = time.time() - start
        
        print(f"✓ Matching completed in {elapsed:.3f}s")
        print(f"  Input: {desc1.shape[0]} x {desc2.shape[0]} descriptors")
        print(f"  Output: {len(matches)} matches found")
        
        if len(matches) > 0:
            print(f"  Match distance range: {matches[:, 2].min():.4f} ~ {matches[:, 2].max():.4f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Matching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_npy_file_loading():
    """NPY 파일 로드 테스트"""
    print("\n" + "=" * 60)
    print("4. Testing Feature File Loading")
    print("=" * 60)
    
    npy_dir = "results_npy"
    
    if not os.path.exists(npy_dir):
        print(f"⚠ Directory '{npy_dir}' not found")
        print("  Creating test files...")
        
        os.makedirs(npy_dir, exist_ok=True)
        
        # 테스트용 파일 생성
        pts = np.random.randint(0, 480, (3, 50)).astype(np.float32)
        desc = np.random.randn(256, 50).astype(np.float32)
        
        np.save(os.path.join(npy_dir, "frame_00001_pts.npy"), pts)
        np.save(os.path.join(npy_dir, "frame_00001_desc.npy"), desc)
        
        print(f"✓ Created sample files in '{npy_dir}'")
        return True
    
    else:
        # 기존 파일 확인
        import glob
        pts_files = glob.glob(os.path.join(npy_dir, "*_pts.npy"))
        desc_files = glob.glob(os.path.join(npy_dir, "*_desc.npy"))
        
        print(f"✓ Found {len(pts_files)} point files")
        print(f"✓ Found {len(desc_files)} descriptor files")
        
        if len(pts_files) > 0:
            # 첫 번째 파일 로드 테스트
            pts = np.load(pts_files[0])
            desc = np.load(desc_files[0])
            print(f"  Example: {os.path.basename(pts_files[0])}")
            print(f"    Points shape: {pts.shape}")
            print(f"    Descriptors shape: {desc.shape}")
            return True
        else:
            print("⚠ No feature files found in results_npy/")
            return False


def test_output_directory():
    """출력 디렉토리 테스트"""
    print("\n" + "=" * 60)
    print("5. Testing Output Directory Setup")
    print("=" * 60)
    
    output_dir = "matching_results"
    
    try:
        os.makedirs(os.path.join(output_dir, 'matches_viz'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'matches_data'), exist_ok=True)
        
        print(f"✓ Output directories ready in '{output_dir}'")
        print(f"  - {output_dir}/matches_viz/")
        print(f"  - {output_dir}/matches_data/")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create output directories: {e}")
        return False


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  SuperPoint Feature Matching - System Test              ║")
    print("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # 테스트 실행
    results['import'] = test_matcher_import()
    results['gpu'] = test_gpu_availability()
    results['matching'] = test_basic_matching() if results['import'] else False
    results['files'] = test_npy_file_loading()
    results['output'] = test_output_directory()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper():15} {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run SuperPoint to extract features:")
        print("   python py_superpoint.py --input <images> --save_npy")
        print("\n2. Run feature matching:")
        print("   python matcher_main.py --npy_dir results_npy --output_dir matching_results")
        print("\n3. Check results in matching_results/ directory")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("- Make sure PyTorch is installed: pip install torch torchvision")
        print("- Check CUDA installation for GPU support")
        print("- Verify file paths are correct")
        return 1


if __name__ == "__main__":
    sys.exit(main())
