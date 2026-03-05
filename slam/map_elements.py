import numpy as np

class MapPoint:
    """
    3D 공간 상의 한 점을 나타내는 클래스.
    여러 키프레임에서 어떻게 관측되었는지(Covisibility)를 추적합니다.
    """
    def __init__(self, pt_id, pos3d, descriptor=None):
        self.id = pt_id                  # 포인트 고유 ID
        self.pos3d = np.array(pos3d)     # [x, y, z] 3차원 전역 좌표
        self.descriptor = descriptor     # 대표 디스크립터 (가장 마지막에 관측된 것으로 업데이트 가능)
        
        # 이 점을 관측한 키프레임들 기록
        # {keyframe_id: (u, v) 픽셀 좌표}
        self.observations = {}           
        self.is_bad = False              # Culling 여부 플래그

    def add_observation(self, keyframe_id, pixel_uv):
        if not self.is_bad:
            self.observations[keyframe_id] = pixel_uv

    def get_observation_count(self):
        return len(self.observations)

    def update_descriptor(self, new_descriptor):
        self.descriptor = new_descriptor
        
    def set_bad(self):
        self.is_bad = True
        self.observations.clear()


class KeyFrame:
    """
    SLAM 지도에 등록되는 핵심 프레임.
    자신의 포즈(위치/회전)와 추출된 특징점들을 가집니다.
    """
    def __init__(self, kf_id, pose, kpts, desc):
        self.id = kf_id                  # 키프레임 고유 ID (일반적으로 프레임 인덱스)
        self.pose = pose                 # 4x4 변환 행렬 (World to Camera)
        self.kpts = kpts                 # [N, 3] 특징점 [x, y, confidence]
        self.desc = desc                 # [256, N] 디스크립터
        
        # 특징점 인덱스와 맵포인트 ID의 매핑
        # 특정 특징점 인덱스 i가 어떤 MapPoint에 연결되어 있는지 추적
        self.map_points = {}             

    def add_map_point_association(self, kpt_idx, map_point_id):
        self.map_points[kpt_idx] = map_point_id


class Map:
    """
    키프레임들과 맵포인트들을 전체적으로 관리하는 시스템.
    추후 Bundle Adjustment에 직접적으로 입력되는 컨테이너입니다.
    """
    def __init__(self):
        self.keyframes = {}              # {keyframe_id: KeyFrame object}
        self.map_points = {}             # {map_point_id: MapPoint object}
        self.next_mp_id = 0

    def add_keyframe(self, keyframe):
        self.keyframes[keyframe.id] = keyframe

    def create_map_point(self, pos3d, descriptor=None):
        mp = MapPoint(self.next_mp_id, pos3d, descriptor)
        self.map_points[self.next_mp_id] = mp
        self.next_mp_id += 1
        return mp
        
    def cull_bad_map_points(self):
        """
        관측 횟수가 너무 적은 불량 MapPoint들을 메모리에서 삭제합니다.
        스케일 드리프트나 오매칭으로 생성된 노이즈 포인트를 정리합니다.
        """
        bad_mp_ids = []
        for mp_id, mp in self.map_points.items():
            # 2번 이하로 관측된 맵포인트는 제거 
            # (키프레임 생성 시 보통 최소 2개의 관측이 기본으로 생김: 생성 프레임 + 직전 프레임)
            if mp.get_observation_count() < 3:
                mp.set_bad()
                bad_mp_ids.append(mp_id)
                
        # 맵에서 삭제
        for mp_id in bad_mp_ids:
            del self.map_points[mp_id]
            
        print(f"[Map Culling] Removed {len(bad_mp_ids)} bad MapPoints. Remaining active: {len(self.map_points)}")
        return len(bad_mp_ids)

    def get_map_point(self, mp_id):
        return self.map_points.get(mp_id, None)

    def get_recent_map_points(self, num_recent_keyframes=5):
        """
        가장 최근에 추가된 키프레임들에서 관측된 맵포인트들만 모아서 반환.
        현재 프레임을 매칭할 타겟(Local Map)으로 사용.
        """
        recent_kfs = sorted(self.keyframes.values(), key=lambda k: k.id, reverse=True)[:num_recent_keyframes]
        
        local_map_points = {}
        for kf in recent_kfs:
            for kpt_idx, mp_id in kf.map_points.items():
                if mp_id not in local_map_points:
                    # 삭제된(Culling된) 맵 포인트는 건너뛰도록 체크 추가
                    if mp_id in self.map_points:
                        local_map_points[mp_id] = self.map_points[mp_id]
                    
        return local_map_points
