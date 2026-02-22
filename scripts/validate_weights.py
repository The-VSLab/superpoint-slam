"""
SuperPoint 가중치 vs 모델 shape 검증 및 mismatched 레이어 복구
LightGlue 없이 현재 설정에서 최적 성능을 내기 위한 진단 스크립트
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.py_superpoint import SuperPointNetV2
from models.superpoint_mobilenet import SuperPointNetV2 as MobileNetV2


def diagnose_weights(weights_path: str):
    """모델과 가중치의 shape 불일치 진단."""
    print("==> SuperPoint 가중치 진단")
    print(f"가중치 경로: {weights_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    model = SuperPointNetV2().to(device)
    print(f"\n모델 파라미터 '{weights_path}'를 로드 중...")
    
    try:
        checkpoint = torch.load(weights_path, map_location=device)
    except Exception as e:
        print(f"ERROR: 가중치 로드 실패 - {e}")
        return
    
    model_state = model.state_dict()
    ckpt_state = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
    
    print(f"\n모델 파라미터: {len(model_state)}")
    print(f"체크포인트 파라미터: {len(ckpt_state)}")
    
    loaded = 0
    shape_mismatch = []
    missing_keys = []
    extra_keys = []
    
    for key in model_state:
        if key in ckpt_state:
            if model_state[key].shape == ckpt_state[key].shape:
                loaded += 1
            else:
                shape_mismatch.append((key, model_state[key].shape, ckpt_state[key].shape))
        else:
            missing_keys.append(key)
    
    for key in ckpt_state:
        if key not in model_state:
            extra_keys.append(key)
    
    print(f"\n로드된 파라미터: {loaded}/{len(model_state)} ({100*loaded/len(model_state):.1f}%)")
    
    if shape_mismatch:
        print(f"\n⚠️  Shape 불일치 ({len(shape_mismatch)}):")
        for key, m_shape, c_shape in shape_mismatch[:5]:
            print(f"  {key}: 모델={m_shape} vs 체크=={c_shape}")
    
    if missing_keys:
        print(f"\n❌ 누락된 파라미터 ({len(missing_keys)}):")
        for key in missing_keys[:5]:
            print(f"  {key}")
    
    if extra_keys:
        print(f"\n⚠️  체크포인트의 추가 파라미터 ({len(extra_keys)}):")
        for key in extra_keys[:5]:
            print(f"  {key}")
    
    if loaded == len(model_state):
        print("\n✅ 완전 매칭: 가중치 로드 성공")
        return model
    
    print("\n주의: Shape 불일치가 있습니다. 현재 설정상 호환 파라미터만 로드됩니다.")
    
    compatible_state = {}
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible_state[k] = v
    
    model.load_state_dict(compatible_state, strict=False)
    print(f"호환 파라미터 로드 완료: {len(compatible_state)}/{len(model_state)}")
    
    return model


def benchmark_inference(model, device, input_shape=(1, 1, 480, 640), num_iters=5):
    """추론 성능 벤치마크."""
    print(f"\n==> 벤치마크 ({input_shape}, {num_iters}회)")
    
    dummy = torch.randn(input_shape, device=device)
    times = []
    
    model.eval()
    with torch.no_grad():
        for i in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            
            if t0:
                t0.record()
            
            import time
            start = time.perf_counter()
            _ = model(dummy)
            end = time.perf_counter()
            
            if t1:
                t1.record()
                torch.cuda.synchronize()
                times.append(t1.elapsed_time(t0) / 1000.0)
            else:
                times.append(end - start)
    
    times = np.array(times[1:])
    print(f"평균 추론 시간: {times.mean()*1000:.2f}ms ± {times.std()*1000:.2f}ms")
    return times.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuperPoint 가중치 검증 및 진단")
    parser.add_argument("--weights", type=str, required=True, help="SuperPoint 가중치 경로")
    parser.add_argument("--benchmark", action="store_true", help="추론 성능 벤치마크 실행")
    opt = parser.parse_args()
    
    model = diagnose_weights(opt.weights)
    
    if opt.benchmark and model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        benchmark_inference(model, device)
