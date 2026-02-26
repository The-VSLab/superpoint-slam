import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 프로젝트 루트 경로 추가 (모델 임포트를 위해)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 기존에 사용하던 모델 클래스 임포트
from models.superpoint_mobilenet import SuperPointNetV2 

class SuperPointONNXWrapper(nn.Module):
    """TensorRT 최적화를 위해 후처리를 포함한 Wrapper 클래스"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 1. 모델 추론 (Backbone + Heads)
        semi, desc = self.model(x)
        
        # 2. TensorRT에서 지원하는 연산들로 히트맵 재구성
        B, C, Hc, Wc = semi.shape
        prob = F.softmax(semi, dim=1)[:, :-1]
        prob = prob.permute(0, 2, 3, 1).reshape(B, Hc, Wc, 8, 8)
        heatmap = prob.permute(0, 1, 3, 2, 4).reshape(B, Hc*8, Wc*8)
        
        # 3. 디스크립터 L2 정규화
        desc = F.normalize(desc, p=2, dim=1)
        
        return heatmap, desc

def main():
    # 설정
    weights_path = "checkpoints/v14_latest.pth" # 실제 가중치 경로
    save_path = "checkpoints/superpoint.onnx"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    model = SuperPointNetV2()
    checkpoint = torch.load(weights_path, map_location=device)
    
    # state_dict 키 대응 (필요시)
    state_dict = checkpoint["student"] if "student" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # 2. Wrapper 적용
    wrapper = SuperPointONNXWrapper(model)

    # 3. 더미 입력 생성 (SLAM에서 사용할 고정 해상도)
    # resize가 (640, 480)이라면 [1, 3, 480, 640]
    dummy_input = torch.randn(1, 3, 480, 640).to(device)

    # 4. ONNX 익스포트
    torch.onnx.export(
        wrapper, 
        dummy_input, 
        save_path,
        input_names=['image'],
        output_names=['heatmap', 'descriptors'],
        opset_version=12,
        do_constant_folding=True
    )
    print(f"✅ ONNX 변환 완료: {save_path}")

if __name__ == "__main__":
    main()