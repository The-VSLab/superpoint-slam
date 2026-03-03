import numpy as np
import cv2
import faiss  # 최적화 핵심 라이브러리
from dataclasses import dataclass

@dataclass
class LoopClosureResult:
    match_index: int
    transform: np.ndarray
    inliers: int
    inlier_ratio: float
    matches: int
    scale: float

class LoopClosureManager:
    def __init__(
        self,
        matcher,
        K,
        min_frame_gap=15,
        top_k=5,
        min_inliers=30,
        min_inlier_ratio=0.25,
        n_words=128,          # 어휘집 크기 (필요 시 더 키워도 Faiss는 빠릅니다)
        similarity_thresh=0.35,
        vocab_max_desc=20000,
        vocab_rebuild_interval=10,
    ):
        self.matcher = matcher
        self.K = K
        self.min_frame_gap = min_frame_gap
        self.top_k = top_k
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio

        self.n_words = n_words
        self.similarity_thresh = similarity_thresh
        self.vocab_max_desc = vocab_max_desc
        self.vocab_rebuild_interval = vocab_rebuild_interval

        self.keyframes = []
        self._desc_pool = []
        
        # Faiss 관련 속성
        self._vocab_centroids = None
        self._vocab_index = None  # 특징점을 단어로 변환하기 위한 인덱스
        self._history_index = None # 키프레임 히스토그램들을 검색하기 위한 인덱스
        self._vocab_dirty = False

    def _to_numpy_float32(self, desc):
        if desc is None or desc.size == 0:
            return None
        # SuperPoint의 (256, N) -> (N, 256) 변환 및 float32 강제
        return np.ascontiguousarray(desc.T.astype(np.float32))

    def _add_desc_to_pool(self, desc):
        desc_vec = self._to_numpy_float32(desc)
        if desc_vec is None: return

        # 풀이 너무 커지지 않게 샘플링
        if len(desc_vec) > 500:
            idx = np.random.choice(len(desc_vec), 500, replace=False)
            desc_vec = desc_vec[idx]
            
        self._desc_pool.append(desc_vec)
        self._vocab_dirty = True

    def _maybe_rebuild_vocab(self):
        if not self._vocab_dirty or len(self.keyframes) < 2:
            return
        if len(self.keyframes) > 200:
            self._vocab_dirty = False
            return

        all_desc = np.vstack(self._desc_pool)
        if len(all_desc) < self.n_words:
            return

        if len(all_desc) > self.vocab_max_desc:
            idx = np.random.choice(len(all_desc), self.vocab_max_desc, replace=False)
            all_desc = all_desc[idx]

        # 1. Faiss KMeans로 어휘집 학습 (sklearn보다 훨씬 빠름)
        d = all_desc.shape[1]
        kmeans = faiss.Kmeans(d, self.n_words, niter=20, verbose=False, seed=42)
        kmeans.train(all_desc)
        self._vocab_centroids = kmeans.centroids

        # 2. 특징점 -> 단어 변환용 L2 인덱스 구축
        self._vocab_index = faiss.IndexFlatL2(d)
        self._vocab_index.add(self._vocab_centroids)

        # 3. 기존 모든 키프레임의 히스토그램 재계산 및 검색 인덱스 갱신
        self._history_index = faiss.IndexFlatIP(self.n_words) # Cosine Similarity용
        for kf in self.keyframes:
            kf["bovw_hist"] = self._compute_bovw_hist(kf.get("desc"))
            if kf["bovw_hist"] is not None:
                self._history_index.add(kf["bovw_hist"].reshape(1, -1))

        self._vocab_dirty = False
        print(f"  [Faiss-BoVW] Vocab Updated: {self.n_words} words.")

    def _compute_bovw_hist(self, desc):
        if self._vocab_index is None: return None
        desc_vec = self._to_numpy_float32(desc)
        if desc_vec is None: return None

        # 1. 각 디스크립터가 어느 단어(Centroid)에 속하는지 검색
        _, labels = self._vocab_index.search(desc_vec, 1)
        
        # 2. 히스토그램 생성 및 정규화
        hist = np.bincount(labels.ravel(), minlength=self.n_words).astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm < 1e-6: return None
        return hist / norm

    def add_keyframe(self, frame_idx, kpts, desc, pts_3d=None):
        self._add_desc_to_pool(desc)
        self._maybe_rebuild_vocab()
        
        bovw_hist = self._compute_bovw_hist(desc)
        
        # 검색 인덱스에 새 히스토그램 추가
        if bovw_hist is not None and self._history_index is not None:
            self._history_index.add(bovw_hist.reshape(1, -1))

        self.keyframes.append({
            "frame_idx": int(frame_idx),
            "kpts": kpts,
            "desc": desc,
            "pts_3d": pts_3d,
            "bovw_hist": bovw_hist,
        })

    def find_loop(self, frame_idx, kpts, desc):
        # 1. 검사 주기 제한: 모든 프레임에서 수행하지 않고 5프레임마다 한 번만 수행
        if frame_idx % 10 != 0: 
            return None

        query_hist = self._compute_bovw_hist(desc)
        if query_hist is None: return None

        # 2. Faiss 검색 (후보를 넉넉히 뽑되, 나중에 군집화함)
        search_k = min(len(self.keyframes), 20) 
        sims, indices = self._history_index.search(query_hist.reshape(1, -1), search_k)
        
        sims = sims.ravel()
        indices = indices.ravel()

        # 3. 군집화(Clustering) 및 필터링
        tested_clusters = set()
        candidates = []

        max_verify = 1
        count = 0
        
        for sim, cand_idx in candidates:
            if count >= max_verify: break
            result = self._verify_candidate(cand_idx, kpts, desc)
            count += 1
            if result is not None: return result
            if sim < self.similarity_thresh: continue
            
            kf_idx = self.keyframes[cand_idx]["frame_idx"]
            
            # 시간적 거리(Gap) 확인
            if frame_idx - kf_idx < self.min_frame_gap: continue
            
            # 군집화: 이미 검사한 프레임의 근처(+/- 20프레임)라면 건너뜀
            cluster_id = kf_idx // 20 
            if cluster_id in tested_clusters: continue
            
            candidates.append((sim, cand_idx))
            tested_clusters.add(cluster_id)
            
            # 너무 많은 후보를 검사하지 않도록 제한 (최대 3개 군집만)
            if len(candidates) >= 3: break

        # 4. 선별된 대표 후보만 검증
        for sim, cand_idx in candidates:
            # 검증 로직 진입 전 로그 출력 (디버깅용)
            # print(f"  [Loop Search] Checking Representative Frame {self.keyframes[cand_idx]['frame_idx']}")
            
            result = self._verify_candidate(cand_idx, kpts, desc)
            if result is not None:
                # [중요] 루프를 하나라도 찾으면 즉시 종료하여 FPS 확보
                return result

        return None

    def _verify_candidate(self, cand_idx, kpts, desc):
        # ... (기존 _verify_candidate 코드와 동일하게 유지)
        # PnP 검증 로직은 수학적 계산이므로 그대로 사용하시면 됩니다.
        cand = self.keyframes[cand_idx]
        if cand["desc"] is None or desc is None: return None

        matches = self.matcher.match(cand["desc"], desc)
        if matches.shape[0] < 8: return None

        if cand["pts_3d"] is not None:
            p1_3d = cand["pts_3d"][matches[:, 0]]
            p2_2d = kpts[matches[:, 1], :2].astype(np.float64)
            
            valid_mask = ~np.isnan(p1_3d[:, 0])
            obj_pts = p1_3d[valid_mask].astype(np.float32)
            img_pts = p2_2d[valid_mask].astype(np.float32)
            
            if len(obj_pts) >= 15:
                obj_pts_c = np.ascontiguousarray(obj_pts).reshape(-1, 1, 3)
                img_pts_c = np.ascontiguousarray(img_pts).reshape(-1, 1, 2)
                
                success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                    obj_pts_c, img_pts_c, self.K, np.zeros(4, dtype=np.float32), 
                    iterationsCount=1000, reprojectionError=15.0, confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success and inliers_pnp is not None:
                    inliers = len(inliers_pnp)
                    inlier_ratio = inliers / max(len(obj_pts), 1)
                    
                    if inliers >= 12 and inlier_ratio >= 0.1: # 약간 더 엄격하게 조정 가능
                        R, _ = cv2.Rodrigues(rvec)
                        transform = np.eye(4)
                        transform[:3, :3] = R.T
                        transform[:3, 3] = -R.T @ tvec[:, 0]
                        scale = np.linalg.norm(transform[:3, 3])
                        
                        return LoopClosureResult(
                            match_index=cand_idx, transform=transform,
                            inliers=inliers, inlier_ratio=inlier_ratio,
                            matches=int(matches.shape[0]), scale=scale
                        )
        return None