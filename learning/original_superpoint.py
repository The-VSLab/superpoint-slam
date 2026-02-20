import torch
import torch.nn as nn
from collections import OrderedDict
from types import SimpleNamespace

# 오리지널 유틸리티 함수 유지
def batched_nms(scores, nms_radius: int):
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(c_in, c_out, kernel_size, stride=1, padding=(kernel_size-1)//2)),
            ("activation", nn.ReLU(inplace=True) if relu else nn.Identity()),
            ("bn", nn.BatchNorm2d(c_out, eps=0.001))
        ]))

class SuperPoint(nn.Module):
    # [엄격한 모서리 선별 설정]
    default_conf = {
        "nms_radius": 12,              # [상향] 선 성분 억제 및 점 분산 유도
        "max_num_keypoints": 600,     # [고정] Top-K 600개 제한
        "detection_threshold": 0.025, # [상향] 확실한 코너만 인정
        "remove_borders": 12,         # [상향] 테두리 노이즈 제거
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.stride = 8
        
        # 가중치 파일 v6 구조와 100% 일치하는 backbone
        self.backbone = nn.Sequential(
            nn.Sequential(VGGBlock(1, 64, 3), VGGBlock(64, 64, 3), nn.MaxPool2d(2, 2)),
            nn.Sequential(VGGBlock(64, 64, 3), VGGBlock(64, 64, 3), nn.MaxPool2d(2, 2)),
            nn.Sequential(VGGBlock(64, 128, 3), VGGBlock(128, 128, 3), nn.MaxPool2d(2, 2)),
            nn.Sequential(VGGBlock(128, 128, 3), VGGBlock(128, 128, 3))
        )
        self.detector = nn.Sequential(VGGBlock(128, 256, 3), VGGBlock(256, 65, 1, relu=False))
        self.descriptor = nn.Sequential(VGGBlock(128, 256, 3), VGGBlock(256, 256, 1, relu=False))

    def forward(self, data):
        image = data["image"]
        features = self.backbone(image)
        
        logits = self.detector(features)
        desc = self.descriptor(features)
        desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        scores = torch.nn.functional.softmax(logits, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        
        # [핵심] NMS 및 테두리 제거
        scores = batched_nms(scores, self.conf.nms_radius)
        pad = self.conf.remove_borders
        scores[:, :pad] = 0; scores[:, -pad:] = 0
        scores[:, :, :pad] = 0; scores[:, :, -pad:] = 0
        
        res = {"keypoints": [], "desc": desc}
        for i in range(b):
            kp = torch.where(scores[i] > self.conf.detection_threshold)
            kp = torch.stack(kp, -1).flip(1).float()
            # Top-K 선별 (600개)
            if self.conf.max_num_keypoints:
                s = scores[i][torch.where(scores[i] > self.conf.detection_threshold)]
                if len(kp) > self.conf.max_num_keypoints:
                    _, indices = torch.topk(s, self.conf.max_num_keypoints)
                    kp = kp[indices]
            res["keypoints"].append(kp)
        return res