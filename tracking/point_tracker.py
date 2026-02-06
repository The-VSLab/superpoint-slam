import numpy as np
import cv2

# 시각화를 위한 Jet 컬러맵
myjet = np.array(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.99910873],
        [0.0, 0.37843137, 1.0],
        [0.0, 0.83333333, 1.0],
        [0.30044276, 1.0, 0.66729918],
        [0.66729918, 1.0, 0.30044276],
        [1.0, 0.90123457, 0.0],
        [1.0, 0.48002905, 0.0],
        [0.99910873, 0.07334786, 0.0],
        [0.5, 0.0, 0.0],
    ]
)


class PointTracker(object):
    """고정 메모리의 점들과 디스크립터를 관리하여 희소 광학 흐름 점 추적을 가능하게 하는 클래스

    내부적으로, 추적기는 M x (2+L) 크기의 'tracks' 행렬을 저장하며, 최대 길이 L을 가진
    M개의 트랙으로 구성됩니다. 각 행은 다음에 해당합니다:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m]
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError("max_length must be greater than or equal to 2.")
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
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
            raise ValueError("'nn_thresh'는 음수가 아니어야 합니다")
        # L2 거리 계산. 벡터가 단위 정규화되어 있어 쉽습니다.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
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
        """점들의 리스트를 순회하며 오프셋 값을 누적합니다. 전역 점 ID를
        점들의 리스트로 인덱싱하는 데 사용됩니다.

        반환
          offsets - 정수 오프셋 위치를 담은 N 길이 배열
        """
        # ID 오프셋 계산
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # 마지막 카메라 크기는 건너뜀 (필요 없음)
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """추적기에 새로운 점과 디스크립터 관측값 세트를 추가합니다.

        입력
          pts - 2D 점 관측값들의 3xN numpy 배열
          desc - 해당하는 D차원 디스크립터들의 DxN numpy 배열
        """
        if pts is None or desc is None:
            print("PointTracker: 경고, 추적기에 점이 추가되지 않았습니다.")
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
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
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
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.0
                    frac = 1.0 / float(track_len)
                    self.tracks[row, 1] = (1.0 - frac) * self.tracks[row, 1] + frac * match[2]
        # 매칭되지 않은 트랙 추가
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # 트랙 카운트 업데이트
        # 빈 트랙 제거
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # 마지막 디스크립터 저장
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """주어진 최소 길이를 가진 점 트랙들을 검색합니다.
        입력
          min_length - 최소 트랙 길이를 나타내는 >= 1인 정수
        출력
          returned_tracks - 트랙 인덱스를 저장하는 M x (2+L) 크기 행렬,
            여기서 M은 트랙의 수이고 L은 최대 트랙 길이입니다.
        """
        if min_length < 1:
            raise ValueError("'min_length'가 너무 작습니다.")
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # 가장 최근 프레임에서 관측값이 없는 트랙 제거
        not_headless = self.tracks[:, -1] != -1
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """단일 이미지 위에 모든 트랙을 오버레이하여 시각화합니다.
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
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # 각 트랙의 끝점 그리기
                if i == N - 2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)
