import os

import numpy as np
import torch
import torch.nn.functional as F

from models.superpoint_mobilenet import SuperPointNetV2


class SuperPointFrontend(object):
    """PyTorch 네트워크를 감싸서 이미지 전처리 및 후처리를 도와주는 클래스"""

    def __init__(
        self, weights_path=None, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, max_keypoints=1000, cuda=False
    ):
        self.name = "SuperPointV2"
        self.max_keypoints = max_keypoints
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # 좋은 매칭을 위한 L2 디스크립터 거리 임계값
        self.cell = 8  # 각 출력 셀의 크기. 고정값입니다.
        self.border_remove = 4  # 경계에서 이 거리만큼 가까운 점들을 제거

        # 추론 모드로 네트워크 로드
        self.net = SuperPointNetV2()
        if weights_path is not None and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu")

            # checkpoint가 'state_dict' 키를 포함하는 딕셔너리 형태인지 확인
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict_to_load = checkpoint["state_dict"]
                print("[SuperPointFrontend] Loading 'state_dict' from wrapped checkpoint.")
            elif isinstance(checkpoint, dict) and "student" in checkpoint:
                state_dict_to_load = checkpoint["student"]
                print("[SuperPointFrontend] Loading 'student' state_dict from wrapped checkpoint.")
            else:
                state_dict_to_load = checkpoint
                print("[SuperPointFrontend] Loading raw checkpoint (not wrapped).")

            # Shape mismatch 무시하고 호환되는 파라미터만 로드
            model_state = self.net.state_dict()
            compatible_state = {}
            for k, v in state_dict_to_load.items():
                # 'module.' 접두사 제거 (DataParallel로 저장된 모델 호환성)
                if k.startswith("module."):
                    k = k[7:]
                if k in model_state and model_state[k].shape == v.shape:
                    compatible_state[k] = v

            self.net.load_state_dict(compatible_state, strict=False)

            # 로드된 파라미터 비율 로깅 (디버깅 및 검증용)
            total_params = len(model_state)
            loaded_params = len(compatible_state)
            print(
                f"[SuperPointFrontend] Loaded {loaded_params}/{total_params} parameters "
                f"from '{weights_path}' (shape-matched only)."
            )

        # 디바이스 선택 (weights 로드 여부와 무관하게 일관되게 적용)
        if self.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("   -> Using NVIDIA GPU (CUDA)")
            # 고정 해상도 입력에서는 커널 튜닝으로 약간의 속도 향상 가능
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("   -> Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("   -> Using CPU")

        self.net = self.net.to(self.device)
        if self.device.type == "cuda":
            self.net = self.net.to(memory_format=torch.channels_last)
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        numpy 코너 배열에 대해 빠른 근사 비최대 억제(Non-Max-Suppression) 수행
        입력 형태: 3xN [x_i, y_i, conf_i]^T

        알고리즘 요약: HxW 크기의 그리드를 생성합니다. 각 코너 위치에 1을 할당하고,
        나머지는 0으로 설정합니다. 모든 1들을 순회하면서 -1 또는 0으로 변환합니다.
        주변 값을 0으로 설정하여 점들을 억제합니다.

        그리드 값 의미:
        -1 : 유지됨
         0 : 비어있거나 억제됨
         1 : 처리 대기 중 (유지 또는 억제로 변환될 예정)

        참고: NMS는 먼저 점들을 정수로 반올림하므로, NMS 거리가 정확히 dist_thresh와
        같지 않을 수 있습니다. 또한 점들이 이미지 경계 내에 있다고 가정합니다.

        입력
          in_corners - 코너들의 3xN numpy 배열 [x_i, y_i, confidence_i]^T
          H - 이미지 높이
          W - 이미지 너비
          dist_thresh - 억제할 거리 (무한 노름 거리로 측정)
        반환
          nmsed_corners - 살아남은 코너들의 3xN numpy 행렬
          nmsed_inds - 살아남은 코너 인덱스들의 N 길이 numpy 벡터
        """
        grid = np.zeros((H, W)).astype(int)  # NMS 데이터 추적
        inds = np.zeros((H, W)).astype(int)  # 점들의 인덱스 저장
        # 신뢰도로 정렬하고 가장 가까운 정수로 반올림
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # 반올림된 코너들
        # 0개 또는 1개 코너인 경계 케이스 확인
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # 그리드 초기화
        for i, _ in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # 경계 근처의 점들도 NMS할 수 있도록 그리드 경계에 패딩 추가
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")
        # 점들을 순회하며, 신뢰도가 높은 것부터 낮은 것 순으로 주변 억제
        count = 0
        for _, rc in enumerate(rcorners.T):
            # 상단 및 왼쪽 패딩 고려
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # 아직 억제되지 않은 경우
                grid[pt[1] - pad : pt[1] + pad + 1, pt[0] - pad : pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # 살아남은 모든 -1들을 가져와서 정렬된 코너 배열 반환
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img, return_desc_tensor=False, use_fp16=False):
        """numpy 이미지를 처리하여 특징점과 디스크립터를 추출합니다.
        입력
          img - [0,1] 범위의 HxW numpy float32 입력 이미지
        출력
          corners - 코너들의 3xN numpy 배열 [x_i, y_i, confidence_i]^T
          desc - 해당하는 단위 정규화된 디스크립터들의 256xN numpy 배열
          heatmap - 점 신뢰도의 [0,1] 범위 HxW numpy 히트맵
        """
        assert img.ndim == 2, "이미지는 흑백이어야 합니다."
        assert img.dtype == np.float32, "이미지는 float32여야 합니다."
        H, W = img.shape[0], img.shape[1]

        # H, W가 self.cell (8)의 배수인지 확인
        assert H % self.cell == 0 and W % self.cell == 0, (
            f"입력 이미지 크기 (H={H}, W={W})는 self.cell ({self.cell})의 배수여야 합니다."
        )
        inp = np.ascontiguousarray(img)
        inp = torch.from_numpy(inp).view(1, 1, H, W)
        # 모델 파라미터와 동일 디바이스로 이동 (CUDA/MPS/CPU 모두 안전)
        inp = inp.to(self.device)
        if self.device.type == "cuda":
            inp = inp.contiguous(memory_format=torch.channels_last)
        # 네트워크 순전파
        with torch.inference_mode():
            if self.device.type == "cuda" and use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    semi, coarse_desc = self.net(inp)
                semi = semi.float()
                coarse_desc = coarse_desc.float()
            else:
                semi, coarse_desc = self.net(inp)
        # --- 특징점 처리 (GPU에서 softmax 후 CPU로 이동)
        dense = F.softmax(semi, dim=1)[0, :-1, :, :]  # dustbin 제거
        nodust = dense.detach().cpu().numpy()
        # 전체 해상도 히트맵을 얻기 위해 재구성
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = nodust.reshape(Hc, Wc, self.cell, self.cell)
        heatmap = heatmap.transpose(0, 2, 1, 3).reshape(Hc * self.cell, Wc * self.cell)
        xs, ys = np.where(heatmap >= self.conf_thresh)  # 신뢰도 임계값
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.empty((3, len(xs)), dtype=np.float32)  # 3xN 크기의 점 데이터 채우기
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # NMS 적용
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # 신뢰도로 정렬
        # 경계선을 따라 있는 점들 제거
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # 점 개수 제한
        if self.max_keypoints > 0 and pts.shape[1] > self.max_keypoints:
            pts = pts[:, :self.max_keypoints]
        # --- 디스크립터 처리
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            if return_desc_tensor:
                desc = torch.empty((D, 0), device=self.device, dtype=torch.float32)
            else:
                desc = np.zeros((D, 0), dtype=np.float32)
        else:
            # 2D 점 위치를 사용하여 디스크립터 맵에 보간
            # align_corners=True일 때 올바른 매핑 식: (coord / (size - 1)) * 2 - 1
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / float(W - 1)) * 2.0 - 1.0
            samp_pts[1, :] = (samp_pts[1, :] / float(H - 1)) * 2.0 - 1.0
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(self.device)

            # coarse_desc가 올바른 디바이스에 있는지 확인 (안전성 체크)
            if coarse_desc.device != self.device:
                coarse_desc = coarse_desc.to(self.device)
            coarse_desc = coarse_desc.float()

            desc_t = F.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc_t = desc_t.reshape(D, -1)
            desc_t = F.normalize(desc_t.float(), p=2, dim=0, eps=1e-8)

            if return_desc_tensor:
                desc = desc_t.detach()
            else:
                desc = desc_t.detach().cpu().numpy()
        return pts, desc, heatmap
