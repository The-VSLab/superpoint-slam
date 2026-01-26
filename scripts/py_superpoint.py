import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# 시각화를 위한 Jet 컬러맵
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

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

        # 3. 특징점 검출 헤드 (Detector Head)
        self.convPa = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

        # 4. 디스크립터 헤드 (Descriptor Head)
        self.convDa = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 1채널(흑백) 입력을 3채널로 확장
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)  # [배치크기, 32, 높이/8, 너비/8]

        # 특징점 검출 헤드
        cPa = nn.functional.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # 디스크립터 헤드
        cDa = nn.functional.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc.div(dn)

        return semi, desc


class SuperPointFrontend(object):
    """ PyTorch 네트워크를 감싸서 이미지 전처리 및 후처리를 도와주는 클래스 """
    def __init__(self, weights_path=None, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7,
                 cuda=False):
        self.name = 'SuperPointV2'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # 좋은 매칭을 위한 L2 디스크립터 거리 임계값
        self.cell = 8  # 각 출력 셀의 크기. 고정값입니다.
        self.border_remove = 4  # 경계에서 이 거리만큼 가까운 점들을 제거

        # 추론 모드로 네트워크 로드
        self.net = SuperPointNetV2()
        if weights_path is not None and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Shape mismatch 무시하고 호환되는 파라미터만 로드
            model_state = self.net.state_dict()
            compatible_state = {}
            for k, v in checkpoint.items():
                if k in model_state and model_state[k].shape == v.shape:
                    compatible_state[k] = v
            
            self.net.load_state_dict(compatible_state, strict=False)
            
            if cuda:
                self.net = self.net.cuda()
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
        inds1 = np.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().astype(int)  # 반올림된 코너들
        # 0개 또는 1개 코너인 경계 케이스 확인
        if rcorners.shape[1] == 0:
            return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, np.zeros((1)).astype(int)
        # 그리드 초기화
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        # 경계 근처의 점들도 NMS할 수 있도록 그리드 경계에 패딩 추가
        pad = dist_thresh
        grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
        # 점들을 순회하며, 신뢰도가 높은 것부터 낮은 것 순으로 주변 억제
        count = 0
        for i, rc in enumerate(rcorners.T):
            # 상단 및 왼쪽 패딩 고려
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1:  # 아직 억제되지 않은 경우
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # 살아남은 모든 -1들을 가져와서 정렬된 코너 배열 반환
        keepy, keepx = np.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ numpy 이미지를 처리하여 특징점과 디스크립터를 추출합니다.
        입력
          img - [0,1] 범위의 HxW numpy float32 입력 이미지
        출력
          corners - 코너들의 3xN numpy 배열 [x_i, y_i, confidence_i]^T
          desc - 해당하는 단위 정규화된 디스크립터들의 256xN numpy 배열
          heatmap - 점 신뢰도의 [0,1] 범위 HxW numpy 히트맵
        """
        assert img.ndim == 2, '이미지는 흑백이어야 합니다.'
        assert img.dtype == np.float32, '이미지는 float32여야 합니다.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = inp.view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # 네트워크 순전파
        with torch.no_grad():
            outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # PyTorch -> numpy 변환
        semi = semi.data.cpu().numpy().squeeze()
        # --- 특징점 처리
        dense = np.exp(semi)  # Softmax
        dense = dense / (np.sum(dense, axis=0)+.00001)  # 합이 1이 되도록 정규화
        # 더스트빈 제거
        nodust = dense[:-1, :, :]
        # 전체 해상도 히트맵을 얻기 위해 재구성
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # 신뢰도 임계값
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # 3xN 크기의 점 데이터 채우기
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # NMS 적용
        inds = np.argsort(pts[2,:])
        pts = pts[:,inds[::-1]]  # 신뢰도로 정렬
        # 경계선을 따라 있는 점들 제거
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- 디스크립터 처리
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # 2D 점 위치를 사용하여 디스크립터 맵에 보간
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


class PointTracker(object):
    """ 고정 메모리의 점들과 디스크립터를 관리하여 희소 광학 흐름 점 추적을 가능하게 하는 클래스

    내부적으로, 추적기는 M x (2+L) 크기의 'tracks' 행렬을 저장하며, 최대 길이 L을 가진
    M개의 트랙으로 구성됩니다. 각 행은 다음에 해당합니다:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m]
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl+2))
        self.track_count = 0
        self.max_score = 9999

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        두 디스크립터 집합에 대해 양방향 최근접 이웃 매칭을 수행합니다.
        디스크립터 A->B의 NN 매칭이 B->A의 NN 매칭과 같아야 합니다.

        입력:
          desc1 - N개의 M차원 디스크립터들의 NxM numpy 행렬
          desc2 - N개의 M차원 디스크립터들의 NxM numpy 행렬
          nn_thresh - 좋은 매칭으로 간주할 디스크립터 거리 임계값

        반환:
          matches - L개의 매칭을 담은 3xL numpy 배열 (L <= N), 각 열 i는
                    이미지 1의 디스크립터 d_i와 이미지 2의 디스크립터 d_j'의 매칭:
                    [d_i 인덱스, d_j' 인덱스, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\'는 음수가 아니어야 합니다')
        # L2 거리 계산. 벡터가 단위 정규화되어 있어 쉽습니다.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        # NN 인덱스와 점수 얻기
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # NN 매칭에 임계값 적용
        keep = scores < nn_thresh
        # 최근접 이웃이 양방향으로 일치하는지 확인하고 유지
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # 살아남은 점 인덱스 얻기
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # 최종 3xN 매칭 데이터 구조 채우기
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        """ 점들의 리스트를 순회하며 오프셋 값을 누적합니다. 전역 점 ID를
        점들의 리스트로 인덱싱하는 데 사용됩니다.

        반환
          offsets - 정수 오프셋 위치를 담은 N 길이 배열
        """
        # ID 오프셋 계산
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts)-1):  # 마지막 카메라 크기는 건너뜀 (필요 없음)
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """ 추적기에 새로운 점과 디스크립터 관측값 세트를 추가합니다.

        입력
          pts - 2D 점 관측값들의 3xN numpy 배열
          desc - 해당하는 D차원 디스크립터들의 DxN numpy 배열
        """
        if pts is None or desc is None:
            print('PointTracker: 경고, 추적기에 점이 추가되지 않았습니다.')
            return
        assert pts.shape[1] == desc.shape[1]
        # last_desc 초기화
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # 가장 오래된 점들 제거, 나중에 ID를 업데이트하기 위해 크기 저장
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # 트랙에서 가장 오래된 점 제거
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # 트랙 오프셋 업데이트
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # 새로운 -1 열 추가
        self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
        # 기존 트랙에 추가 시도
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            # 매칭된 트랙에 새로운 점 추가
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # 트랙 점수 초기화
                    self.tracks[row, 1] = match[2]
                else:
                    # 이동 평균으로 트랙 점수 업데이트
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
        # 매칭되지 않은 트랙 추가
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # 트랙 카운트 업데이트
        # 빈 트랙 제거
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # 마지막 디스크립터 저장
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """ 주어진 최소 길이를 가진 점 트랙들을 검색합니다.
        입력
          min_length - 최소 트랙 길이를 나타내는 >= 1인 정수
        출력
          returned_tracks - 트랙 인덱스를 저장하는 M x (2+L) 크기 행렬,
            여기서 M은 트랙의 수이고 L은 최대 트랙 길이입니다.
        """
        if min_length < 1:
            raise ValueError('\'min_length\'가 너무 작습니다.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # 가장 최근 프레임에서 관측값이 없는 트랙 제거
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """ 단일 이미지 위에 모든 트랙을 오버레이하여 시각화합니다.
        입력
          out - 트랙이 오버레이될 HxWx3 크기의 numpy uint8 이미지
          tracks - 트랙 정보를 저장하는 M x (2+L) 크기 행렬
        """
        # 카메라당 점의 수 저장
        pts_mem = self.all_pts
        N = len(pts_mem)  # 카메라/이미지의 수
        # pts_mem을 참조하는 데 필요한 오프셋 ID 얻기
        offsets = self.get_offsets()
        # 그려질 트랙과 점 원의 두께
        stroke = 1
        # 각 트랙을 순회하며 그리기
        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
            for i in range(N-1):
                if track[i+2] == -1 or track[i+3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i+1]
                idx1 = int(track[i+2]-offset1)
                idx2 = int(track[i+3]-offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i+1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # 각 트랙의 끝점 그리기
                if i == N-2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


class VideoStreamer(object):
    """ 이미지 스트림 처리를 도와주는 클래스. 세 가지 유형의 입력 가능:
      1.) USB 웹캠
      2.) 이미지 디렉토리 ('img_glob'과 일치하는 디렉토리 내 파일들)
      3.) 비디오 파일 (예: .mp4 또는 .avi 파일)
    """
    def __init__(self, basedir, camid, height, width, skip, img_glob):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000
        # "basedir" 문자열이 "camera"라는 단어이면 웹캠 사용
        if basedir == "camera/" or basedir == "camera":
            print('==> 웹캠 입력 처리 중.')
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            # 비디오로 열기 시도
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('비디오 파일을 열 수 없습니다')
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
                print('==> 비디오 입력 처리 중.')
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)
                self.listing = self.listing[::self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print('==> 이미지 디렉토리 입력 처리 중.')
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                self.listing.sort()
                self.listing = self.listing[::self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError('이미지를 찾을 수 없습니다 (잘못된 \'--img_glob\' 파라미터일 수 있음)')

    def read_image(self, impath, img_size):
        """ 이미지를 흑백으로 읽고 img_size로 크기 조정합니다.
        입력
          impath: 입력 이미지 경로
          img_size: 크기 조정 크기를 지정하는 (W, H) 튜플
        반환
          grayim: [0, 1] 범위의 값을 가진 H x W 크기의 float32 numpy 배열
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('이미지 읽기 오류 %s' % impath)
        # OpenCV를 통해 이미지 크기 조정
        interp = cv2.INTER_AREA
        grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)
        return grayim

    def next_frame(self):
        """ 다음 프레임을 반환하고 내부 카운터를 증가시킵니다.
        반환
           image: 다음 H x W 이미지
           status: 이미지가 로드되었는지 여부에 따른 True 또는 False
        """
        if self.i == self.maxlen:
            return (None, False)
        if self.camera:
            ret, input_image = self.cap.read()
            if ret is False:
                print('VideoStreamer: 카메라에서 이미지를 가져올 수 없습니다 (잘못된 --camid일 수 있음)')
                return (None, False)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                   interpolation=cv2.INTER_AREA)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float')/255.0
        else:
            image_file = self.listing[self.i]
            input_image = self.read_image(image_file, self.sizer)
        # 내부 카운터 증가
        self.i = self.i + 1
        input_image = input_image.astype('float32')
        return (input_image, True)


if __name__ == '__main__':

    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint V2 데모 (MobileNet 백본).')
    parser.add_argument('input', type=str, default='',
        help='이미지 디렉토리 또는 비디오 파일 또는 "camera" (웹캠용).')
    parser.add_argument('--weights_path', type=str, default=None,
        help='사전 학습된 가중치 파일 경로 (선택사항, None이면 사전 학습된 MobileNet 사용).')
    parser.add_argument('--img_glob', type=str, default='*.png',
        help='이미지 디렉토리가 지정된 경우 glob 패턴 (기본값: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
        help='입력이 비디오 또는 디렉토리인 경우 건너뛸 이미지 수 (기본값: 1).')
    parser.add_argument('--show_extra', action='store_true',
        help='추가 디버그 출력 표시 (기본값: False).')
    parser.add_argument('--H', type=int, default=120,
        help='입력 이미지 높이 (기본값: 120).')
    parser.add_argument('--W', type=int, default=160,
        help='입력 이미지 너비 (기본값: 160).')
    parser.add_argument('--display_scale', type=int, default=2,
        help='출력 시각화를 확대할 배율 (기본값: 2).')
    parser.add_argument('--min_length', type=int, default=2,
        help='점 트랙의 최소 길이 (기본값: 2).')
    parser.add_argument('--max_length', type=int, default=5,
        help='점 트랙의 최대 길이 (기본값: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
        help='비최대 억제(NMS) 거리 (기본값: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
        help='검출기 신뢰도 임계값 (기본값: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
        help='디스크립터 매칭 임계값 (기본값: 0.7).')
    parser.add_argument('--camid', type=int, default=0,
        help='OpenCV 웹캠 비디오 캡처 ID, 보통 0 또는 1 (기본값: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
        help='OpenCV waitkey 시간(ms) (기본값: 1).')
    parser.add_argument('--cuda', action='store_true',
        help='네트워크 처리 속도를 높이기 위해 CUDA GPU 사용 (기본값: False)')
    parser.add_argument('--no_display', action='store_true',
        help='화면에 이미지를 표시하지 않음. 원격 실행 시 유용 (기본값: False).')
    parser.add_argument('--write', action='store_true',
        help='출력 프레임을 디렉토리에 저장 (기본값: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
        help='출력 프레임을 저장할 디렉토리 (기본값: tracker_outputs/).')
    parser.add_argument('--save_npy', action='store_true',
        help='각 프레임의 특징점/디스크립터/히트맵을 .npy로 저장 (기본값: False).')
    parser.add_argument('--save_npy_dir', type=str, default='npy_outputs/',
        help='.npy 결과를 저장할 디렉토리 (기본값: npy_outputs/).')
    opt = parser.parse_args()
    print(opt)

    # 다양한 소스에서 입력 이미지를 로드하는 데 도움이 되는 클래스
    vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

    print('==> 사전 학습된 네트워크 로딩 중.')
    # SuperPoint 네트워크를 실행하고 출력을 처리하는 클래스
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,
                            conf_thresh=opt.conf_thresh,
                            nn_thresh=opt.nn_thresh,
                            cuda=opt.cuda)
    print('==> 사전 학습된 네트워크 로딩 완료.')

    # 연속된 점 매칭을 트랙으로 병합하는 데 도움이 되는 클래스
    tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

    # 데모를 표시할 창 생성
    if not opt.no_display:
        win = 'SuperPoint V2 Tracker (MobileNet)'
        cv2.namedWindow(win)
    else:
        print('시각화 건너뛰기, GUI를 표시하지 않습니다.')

    # 시각화를 위한 폰트 파라미터
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    # 원하는 경우 출력 디렉토리 생성
    if opt.write:
        print('==> 출력을 %s에 저장합니다' % opt.write_dir)
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)
    if opt.save_npy:
        if not os.path.exists(opt.save_npy_dir):
            os.makedirs(opt.save_npy_dir)

    print('==> 데모 실행 중.')
    while True:

        start = time.time()

        # 새로운 이미지 가져오기
        img, status = vs.next_frame()
        if status is False:
            break

        # 특징점과 디스크립터 가져오기
        start1 = time.time()
        pts, desc, heatmap = fe.run(img)
        end1 = time.time()

        # 추적기에 특징점과 디스크립터 추가
        tracker.update(pts, desc)

        # 모든 프레임에서 성공적으로 매칭된 점들의 트랙 가져오기
        tracks = tracker.get_tracks(opt.min_length)

        # 주요 출력 - 입력 이미지 위에 점 트랙 오버레이 표시
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        if tracks.shape[0] > 0:
            tracks[:, 1] /= float(fe.nn_thresh)  # 트랙 점수를 [0,1]로 정규화
            tracker.draw_tracks(out1, tracks)
        if opt.show_extra:
            cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

        # 추가 출력 -- 현재 점 검출 표시
        out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

        # 추가 출력 -- 점 신뢰도 히트맵 표시
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
            out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
            out3 = (out3*255).astype('uint8')
        else:
            out3 = np.zeros_like(out2)
        cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

        # 최종 출력 크기 조정
        if opt.show_extra:
            out = np.hstack((out1, out2, out3))
            out = cv2.resize(out, (3*opt.display_scale*opt.W, opt.display_scale*opt.H))
        else:
            out = cv2.resize(out1, (opt.display_scale*opt.W, opt.display_scale*opt.H))

        # 화면에 시각화 이미지 표시
        if not opt.no_display:
            cv2.imshow(win, out)
            key = cv2.waitKey(opt.waitkey) & 0xFF
            if key == ord('q'):
                print('종료, \'q\' 키가 눌렸습니다.')
                break

        # 선택적으로 이미지를 디스크에 저장
        if opt.write:
            out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
            print('이미지를 %s에 저장 중' % out_file)
            cv2.imwrite(out_file, out)
        # 선택적으로 npy 결과 저장
        if opt.save_npy:
            base = os.path.join(opt.save_npy_dir, 'frame_%05d' % vs.i)
            np.save(base + '_pts.npy', pts)
            if desc is not None:
                np.save(base + '_desc.npy', desc)
            if heatmap is not None:
                np.save(base + '_heatmap.npy', heatmap)

        end = time.time()
        net_t = (1./ float(end1 - start))
        total_t = (1./ float(end - start))
        if opt.show_extra:
            print('이미지 %d 처리 완료 (네트워크+후처리: %.2f FPS, 전체: %.2f FPS).'\
                  % (vs.i, net_t, total_t))

    # 남아있는 모든 창 닫기
    cv2.destroyAllWindows()

    print('==> 데모 완료.')