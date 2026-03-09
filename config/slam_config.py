"""
SLAM 시스템 설정 모듈
모든 하이퍼파라미터를 계층적 dataclass로 관리하고 YAML 파일에서 로드/저장합니다.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from pathlib import Path
import yaml


# ============================================================================
# 개별 설정 섹션
# ============================================================================

@dataclass
class CameraConfig:
    focal_multiplier: float = 0.8
    min_dimension: int = 64
    resize_target: int = 640
    dashboard_mask_ratio: float = 0.92


@dataclass
class SuperPointConfig:
    conf_thresh: float = 0.015
    nn_thresh: float = 0.7
    max_kpts: int = 1500
    min_kpts: int = 600
    roi_sky: float = 0.35
    roi_hood: float = 0.85
    sp_scale: float = 1.0
    sp_interval: int = 1
    sp_fp16: bool = False


@dataclass
class CLAHEConfig:
    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: List[int] = field(default_factory=lambda: [8, 8])


@dataclass
class MatcherConfig:
    mutual: bool = True
    ratio_thresh: float = 0.85


@dataclass
class SemanticConfig:
    enabled: bool = False
    yolo_conf: float = 0.3
    dynamic_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])


@dataclass
class PointFilterConfig:
    use_shadow_filter: bool = True
    use_top_region_filter: bool = True
    shadow_value_thresh: float = 0.46
    shadow_saturation_thresh: float = 0.30
    min_shadow_grad: float = 15.0
    min_shadow_local_std: float = 8.0
    shadow_rel_dark_thresh: float = 0.82
    top_region_ratio: float = 0.30
    top_region_min_grad: float = 25.0
    top_region_min_std: float = 10.0


@dataclass
class FlowConfig:
    win_size: int = 21
    max_level: int = 3
    fb_thresh: float = 1.0
    termination_epsilon: float = 0.01
    termination_max_iter: int = 30


@dataclass
class PnPConfig:
    min_points: int = 15
    min_inliers: int = 10
    reprojection_error: float = 5.0
    min_translation: float = 0.001
    max_translation: float = 10.0
    ransac_iterations: int = 1000
    confidence: float = 0.99


@dataclass
class MotionConfig:
    speed_jump_multiplier: float = 5.0
    min_speed_thresh: float = 0.5
    direction_reversal_cos: float = -0.5


@dataclass
class StabilizationConfig:
    y_damping: float = 0.05
    pitch_roll_damping: float = 0.15
    inertia_weight: float = 0.4
    y_clip: List[float] = field(default_factory=lambda: [-0.5, 0.5])
    dead_reckoning_decay: float = 0.8
    speed_history_len: int = 30


@dataclass
class TriangulationConfig:
    min_depth: float = 0.5
    max_depth: float = 150.0
    max_x_extent: float = 100.0
    max_y_extent: float = 20.0


@dataclass
class KeyframeConfig:
    parallax_threshold: float = 30.0
    min_gap: int = 5
    max_gap: int = 30
    adaptive_inlier_ratio: float = 0.4
    min_adaptive_inliers: int = 30
    stall_distance: float = 0.5
    stall_lookback: int = 10
    alignment_dist: float = 3.0
    culling_interval: int = 5


@dataclass
class EpipolarConfig:
    essential_threshold: float = 0.5
    homography_threshold: float = 3.0
    homography_selection_ratio: float = 0.85
    homography_min_inliers: int = 15


@dataclass
class BAConfig:
    local_keyframes: int = 5
    local_iterations: int = 5
    global_iterations: int = 10
    min_observations: int = 2
    min_map_points: int = 10
    reprojection_weight: float = 0.5
    huber_delta: float = 5.991
    pose_drift_threshold: float = 3.0


@dataclass
class LoopClosureConfig:
    min_frame_gap: int = 30
    top_k: int = 5
    min_inliers: int = 30
    min_inlier_ratio: float = 0.25
    descriptor_similarity: float = 0.97
    # PnP 수락 기준 (loop_closure.py 내부)
    verify_min_inliers: int = 25
    verify_min_inlier_ratio: float = 0.15
    verify_min_3d_points: int = 15
    verify_reprojection_error: float = 5.0


@dataclass
class PoseGraphConfig:
    rotation_weight: float = 10.0
    translation_weight: float = 1.0
    iterations: int = 20


@dataclass
class MapCullingConfig:
    min_observations: int = 3


@dataclass
class VizConfig:
    enabled: bool = True
    window_size: List[int] = field(default_factory=lambda: [640, 360])
    keypoint_radius: int = 2
    voxel_size: float = 0.1
    point_size: float = 3.0


@dataclass
class PerformanceConfig:
    filter_on_sp_only: bool = False
    local_map_limit: int = 0
    local_map_keyframes: int = 5
    local_map_projection_dist: float = 10.0


# ============================================================================
# 최상위 설정
# ============================================================================

@dataclass
class SLAMConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    superpoint: SuperPointConfig = field(default_factory=SuperPointConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    point_filter: PointFilterConfig = field(default_factory=PointFilterConfig)
    optical_flow: FlowConfig = field(default_factory=FlowConfig)
    pnp: PnPConfig = field(default_factory=PnPConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    epipolar: EpipolarConfig = field(default_factory=EpipolarConfig)
    ba: BAConfig = field(default_factory=BAConfig)
    loop_closure: LoopClosureConfig = field(default_factory=LoopClosureConfig)
    pose_graph: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    map_culling: MapCullingConfig = field(default_factory=MapCullingConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SLAMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SLAMConfig":
        """중첩 딕셔너리에서 재귀적으로 dataclass 인스턴스 생성"""
        kwargs = {}
        for fld in cls.__dataclass_fields__.values():
            if fld.name in data:
                val = data[fld.name]
                fld_type = fld.type
                # 문자열로 된 타입 어노테이션을 실제 클래스로 해석
                if isinstance(fld_type, str):
                    fld_type = eval(fld_type)
                if hasattr(fld_type, "__dataclass_fields__") and isinstance(val, dict):
                    kwargs[fld.name] = _build_dataclass(fld_type, val)
                else:
                    kwargs[fld.name] = val
        return cls(**kwargs)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def merge_cli(self, overrides: dict):
        """CLI 오버라이드를 dot-notation 키로 적용. 예: {'pnp.min_points': 20}"""
        for key, value in overrides.items():
            if value is None:
                continue
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)


def _build_dataclass(dc_cls, data: dict):
    """범용 dataclass 빌더"""
    kwargs = {}
    for fld in dc_cls.__dataclass_fields__.values():
        if fld.name in data:
            val = data[fld.name]
            fld_type = fld.type
            if isinstance(fld_type, str):
                fld_type = eval(fld_type)
            if hasattr(fld_type, "__dataclass_fields__") and isinstance(val, dict):
                kwargs[fld.name] = _build_dataclass(fld_type, val)
            else:
                kwargs[fld.name] = val
    return dc_cls(**kwargs)
