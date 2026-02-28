import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class SuperPointNetV2(nn.Module):
    def __init__(self):
        super(SuperPointNetV2, self).__init__()

        # 1. MobileNetV2 로드
        # PyTorch 0.13+에서는 pretrained 대신 weights 파라미터 사용
        try:
            # 최신 PyTorch (0.13+)
            from torchvision.models import MobileNet_V2_Weights

            v2_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        except (ImportError, AttributeError):
            # 구버전 PyTorch 호환성
            v2_model = mobilenet_v2(pretrained=True).features

        # 2. 8배 다운샘플링 지점까지 자르기
        # Index 0~6까지가 딱 1/8 해상도 지점입니다. (입력 224 -> 출력 28)
        self.backbone = nn.Sequential(*list(v2_model.children())[:7])

        # MobileNetV2의 해당 지점 출력 채널 수는 32입니다.
        in_channels = 32

        # 3. 특징점 검출 헤드 (Detector Head) - Depthwise Separable Conv 적용
        self.convPa = nn.Sequential(
            # Depthwise: 채널 단위 공간 연산
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise: 채널 확장 연산
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

        # 4. 디스크립터 헤드 (Descriptor Head) - Depthwise Separable Conv 적용
        self.convDa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 1채널(흑백) 입력을 3채널로 확장
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # [핵심 버그 수정] ImageNet 정규화 적용 (Pre-trained MobileNetV2 백본을 위해 필수!)
        # 입력 x는 [0, 1] 범위의 텐서이므로, 백본이 학습했던 분포로 맞춰주어야 특징 추출이 가능합니다.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        x = self.backbone(x)  # [배치크기, 32, 높이/8, 너비/8]

        # 특징점 검출 헤드
        cPa = F.relu(self.convPa(x), inplace=True)
        semi = self.convPb(cPa)

        # 디스크립터 헤드
        cDa = F.relu(self.convDa(x), inplace=True)
        desc = self.convDb(cDa)
        
        # 모델 내부 L2 정규화 복구 (Cosine Embedding Loss의 안정적인 학습 및 수렴을 위해 필수적임)
        desc = F.normalize(desc, p=2, dim=1, eps=1e-8) 

        return semi, desc
