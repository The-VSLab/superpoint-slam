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
class SuperPoint(nn.Module):
    # [엄격한 모서리 선별 설정]
    default_conf = {
        "nms_radius": 12,              # [상향] 선 성분 억제 및 점 분산 유도
        "max_num_keypoints": 600,     # [고정] Top-K 600개 제한
        "detection_threshold": 0.025, # [상향] 확실한 코너만 인정
        "remove_borders": 12,         # [상향] 테두리 노이즈 제거
    }

    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.stride = 8
        self.return_desc = bool(getattr(self.conf, "return_desc", True))
        
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shared Encoder (Backbone)
        self.conv1a = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Detector Head
        self.convPa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor Head
        self.convDa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, data):
        image = data["image"]
        
        # Share Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector Head
        cPa = self.relu(self.convPa(x))
        logits = self.convPb(cPa)
        
        # Descriptor Head
        desc = None
        if self.return_desc:
            cDa = self.relu(self.convDa(x))
            desc = self.convDb(cDa)
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        scores = torch.nn.functional.softmax(logits, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        
        # [핵심] NMS 및 테두리 제거
        scores = batched_nms(scores, self.conf.nms_radius)
        pad = self.conf.remove_borders
        scores[:, :pad] = 0; scores[:, -pad:] = 0
        scores[:, :, :pad] = 0; scores[:, :, -pad:] = 0
        
        res = {"keypoints": []}
        if desc is not None:
            res["desc"] = desc
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