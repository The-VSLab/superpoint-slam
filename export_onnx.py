import torch
import torch.onnx
from py_superpoint import SuperPointNetV2  # 작성하신 파일명이 py_superpoint.py라고 가정

def export_superpoint_onnx(weights_path, output_path, device='cpu'):
    # 1. 모델 초기화 및 가중치 로드
    model = SuperPointNetV2()
    
    if weights_path:
        # device를 map_location에 적절히 변환
        if device == 'cpu':
            map_location = 'cpu'
        elif device.startswith('cuda'):
            map_location = device
        else:
            map_location = device
            
        checkpoint = torch.load(weights_path, map_location=map_location)
        
        # checkpoint가 딕셔너리이고 'state_dict' 키가 있는 경우 처리
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"가중치 로드 완료: {weights_path}")
    else:
        print("사전 학습된 가중치 없이 기본 MobileNetV2 가중치 상태로 내보냅니다.")

    # device를 torch.device 객체로 변환
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    model.eval()

    # 2. 더미 입력 데이터 생성 (1채널 흑백 이미지)
    # SLAM 환경에 맞춰 기본 해상도를 설정 (예: 640x480 또는 160x120)
    # 주의: SuperPoint는 8배 다운샘플링을 하므로 입력 크기는 8의 배수여야 합니다
    # 480x640은 8의 배수이므로 문제없습니다 (480%8=0, 640%8=0)
    dummy_input = torch.randn(1, 1, 480, 640).to(device)

    # 3. ONNX 내보내기
    input_names = ["input"]
    output_names = ["semi", "desc"]
    
    # 가변 크기 지원 (Batch size, Height, Width를 가변적으로 설정)
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "semi": {0: "batch_size", 2: "semi_height", 3: "semi_width"},
        "desc": {0: "batch_size", 2: "desc_height", 3: "desc_width"}
    }

    print(f"ONNX 변환 중: {output_path} ...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,  # TensorRT 호환성을 위해 12 이상 권장
        do_constant_folding=True,
        dynamic_axes=dynamic_axes
    )
    
    print("변환이 완료되었습니다.")

if __name__ == "__main__":
    # 파일 경로는 본인의 환경에 맞게 수정하세요.
    WEIGHTS = "superpoint_v2_mobilenet.pth" # 실제 가중치 파일이 있다면 지정
    OUTPUT = "superpoint_v2_mobilenet.onnx"
    
    export_superpoint_onnx(None, OUTPUT) # 현재는 가중치 파일이 없으므로 None 전달