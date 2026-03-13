"""포즈 안정화 모듈 — IMU 부재 시 Y축/Pitch/Roll 드리프트 억제 및 Dead Reckoning."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PoseStabilizer:
    """단안 카메라 SLAM에서 IMU 없이 포즈를 안정화합니다.

    주요 기능:
      - Y축(높이) 이동 댐핑
      - 횡이동(X축) 억제 (highway 모드)
      - Pitch/Roll 억제 (RQDecomp3x3 기반)
      - 적응적 관성 혼합
      - Dead Reckoning (추적 실패 시 감쇠 전진)
    """

    def __init__(self, stab_cfg):
        self.cfg = stab_cfg
        self.last_t_vec = np.array([0.0, 0.0, 1.0])
        self.dead_reckoning_count = 0
        self.recent_speeds: list[float] = []

    def stabilize(self, R: np.ndarray, t_vec: np.ndarray, highway_mode: bool):
        """R, t_vec를 안정화하여 (R_damped, t_vec_stabilized, R_orig, t_orig)를 반환합니다.

        Args:
            R: 3x3 회전 행렬 (PnP/Essential/Homography 결과)
            t_vec: 3-벡터 이동 벡터
            highway_mode: True면 횡이동(X축)을 추가 억제

        Returns:
            R_damped: 안정화된 회전 행렬
            t_vec: 안정화된 이동 벡터
            R_orig: 원본 R (삼각측량용)
            t_orig: 원본 t_vec (삼각측량용)
        """
        cfg = self.cfg

        # 원본 보존 (삼각측량에 필요)
        t_speed = np.linalg.norm(t_vec)
        R_orig = R.copy()
        t_orig = t_vec.copy()

        # 1. 횡이동(X축) 억제 (Turn이 작으면 X 이동 억제)
        if highway_mode:
            turn_amount = np.abs(np.arctan2(R[0, 2], R[2, 2]))
            damp_x = np.clip(turn_amount * 10.0, 0.1, 1.0)
            t_vec[0] *= damp_x

        # 2. Y축(상하 높이) 이동 극단적 억제 (평지 주행 가정)
        t_vec[1] *= cfg.y_damping

        # 3. 댐핑 후 스케일 복원 (속도 감소 방지)
        t_len_after = np.linalg.norm(t_vec)
        if t_len_after > 1e-6 and t_speed > 0:
            t_vec = (t_vec / t_len_after) * t_speed

        # 4. Pitch/Roll 억제 (카메라가 바닥을 보면 위로 올라가는 착각 방지)
        pitch_roll_damp = cfg.pitch_roll_damping
        euler_ret = cv2.RQDecomp3x3(R)
        euler_angles = np.array(euler_ret[0])  # (Pitch, Yaw, Roll) in degrees
        # Qx, Qy, Qz = euler_ret[1], euler_ret[2], euler_ret[3]

        pitch_rad = np.deg2rad(euler_angles[0] * pitch_roll_damp)
        yaw_rad = np.deg2rad(euler_angles[1])  # Yaw는 그대로 유지
        roll_rad = np.deg2rad(euler_angles[2] * pitch_roll_damp)

        # 억제된 Euler 각도로 Rotation Matrix 재구성 (Rz * Ry * Rx)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                       [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
        Ry = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                       [0, 1, 0],
                       [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])
        Rz = np.array([[np.cos(roll_rad), -np.sin(roll_rad), 0],
                       [np.sin(roll_rad), np.cos(roll_rad), 0],
                       [0, 0, 1]])
        R_damped = Rz @ Ry @ Rx

        # 5. 적응적 관성 혼합 (방향 변화에 따라 가중치 조절)
        last_t_norm = np.linalg.norm(self.last_t_vec)
        t_norm = np.linalg.norm(t_vec)
        if last_t_norm > 1e-6 and t_norm > 1e-6:
            cos_sim = np.dot(t_vec, self.last_t_vec) / (t_norm * last_t_norm)
            inertia_w = np.clip(cos_sim * cfg.inertia_weight, 0.0, cfg.inertia_weight)
        else:
            inertia_w = 0.0
        mixed_t = t_vec * (1.0 - inertia_w) + self.last_t_vec * inertia_w
        mixed_len = np.linalg.norm(mixed_t)
        if mixed_len > 1e-6 and t_speed > 0:
            t_vec = (mixed_t / mixed_len) * t_speed
        else:
            t_vec = mixed_t

        # 상태 업데이트
        self.last_t_vec = t_vec.copy()
        self.dead_reckoning_count = 0
        self.recent_speeds.append(float(np.linalg.norm(t_vec)))
        if len(self.recent_speeds) > cfg.speed_history_len:
            self.recent_speeds = self.recent_speeds[-cfg.speed_history_len:]

        return R_damped, t_vec, R_orig, t_orig

    def dead_reckon(self) -> np.ndarray:
        """추적 실패 시 관성 기반 Dead Reckoning 상대 변환을 반환합니다.

        Returns:
            T_rel: 4x4 상대 변환 행렬
        """
        self.dead_reckoning_count += 1
        decay = self.cfg.dead_reckoning_decay ** self.dead_reckoning_count
        T_rel = np.eye(4)
        T_rel[:3, 3] = self.last_t_vec * decay
        return T_rel

    def reset_inertia(self):
        """궤적 정체 감지 시 관성을 리셋합니다."""
        self.last_t_vec *= 0.0
        self.dead_reckoning_count = 0
