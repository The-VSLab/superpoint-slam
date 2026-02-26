import torch
import torch.nn.functional as F
import numpy as np
import os
from models.superpoint_mobilenet import SuperPointNetV2

class SuperPointFrontend(object):
    """GPU 연산 및 NMS가 최적화된 SuperPoint 프론트엔드"""

    def __init__(self, weights_path=None, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False, max_kpts=200):
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.max_kpts = int(max_kpts)

        self.net = SuperPointNetV2().to(self.device)
        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.net.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
        self.net.eval()

    def _gpu_nms(self, scores, radius):
        """MaxPooling을 이용한 GPU 기반 NMS"""
        if radius <= 0: return scores
        s4d = scores.unsqueeze(0).unsqueeze(0)
        m = F.max_pool2d(s4d, kernel_size=radius*2+1, stride=1, padding=radius)
        return (s4d * (s4d == m).float()).squeeze()

    @torch.inference_mode()
    def run(self, img):
        """이미지를 입력받아 특징점(pts), 디스크립터(desc), 히트맵(heatmap) 반환"""
        H, W = img.shape[:2]
        inp = torch.from_numpy(img).view(1, 1, H, W).to(self.device).float()

        semi, coarse_desc = self.net(inp)
        
        # 히트맵 재구성 (GPU)
        prob = F.softmax(semi, dim=1)[:, :-1]
        heatmap = prob.permute(0, 2, 3, 1).reshape(1, H//8, W//8, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(H, W)

        # NMS 및 필터링
        heatmap = self._gpu_nms(heatmap, self.nms_dist)
        indices = torch.where(heatmap >= self.conf_thresh)
        scores = heatmap[indices]

        if len(scores) > self.max_kpts:
            scores, topk_idx = torch.topk(scores, self.max_kpts)
            indices = (indices[0][topk_idx], indices[1][topk_idx])
            
        pts_gpu = torch.stack([indices[1], indices[0]], dim=1).float()
        
        # 디스크립터 보간 (grid_sample)
        grid = pts_gpu.view(1, 1, -1, 2).clone()
        grid[..., 0] = (grid[..., 0] / (W / 2.0)) - 1.0
        grid[..., 1] = (grid[..., 1] / (H / 2.0)) - 1.0
        desc = F.normalize(F.grid_sample(coarse_desc, grid, align_corners=True).squeeze(), p=2, dim=0)

        # 반환 포맷: pts(3, N), desc(D, N), heatmap(H, W)
        pts_final = torch.cat([pts_gpu.T, scores.unsqueeze(0)], dim=0)
        return pts_final.cpu().numpy(), desc.cpu().numpy(), heatmap.cpu().numpy()