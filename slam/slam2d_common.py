from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Slam2DStats:
    name: str
    frames: int
    avg_extract_ms: float
    avg_match_ms: float
    avg_total_ms: float
    avg_kpts: float
    avg_matches: float
    avg_inliers: float
    avg_inlier_ratio: float
    trajectory_length: float
    output_dir: str

    def to_dict(self):
        return asdict(self)


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values))


def compose_pose2d(pose_xy_yaw: np.ndarray, delta_local: np.ndarray, delta_yaw: float) -> np.ndarray:
    x, y, yaw = pose_xy_yaw
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx_local, dy_local = delta_local
    dx_world = c * dx_local - s * dy_local
    dy_world = s * dx_local + c * dy_local
    return np.array([x + dx_world, y + dy_world, yaw + delta_yaw], dtype=np.float64)


def path_length(traj_xy: np.ndarray) -> float:
    if len(traj_xy) < 2:
        return 0.0
    diffs = np.diff(traj_xy, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def filter_sparse_points(map_xy: np.ndarray, cell_size: float = 0.25, min_count: int = 2) -> np.ndarray:
    if map_xy is None or len(map_xy) == 0:
        return map_xy
    if min_count <= 1:
        return map_xy

    cells = np.floor(map_xy / max(cell_size, 1e-6)).astype(np.int32)
    _, inv, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inv] >= min_count
    return map_xy[keep]


def render_topdown_map(
    trajectory_xy: np.ndarray,
    map_xy: np.ndarray,
    is_floor_array: np.ndarray | None = None,
    canvas_size: tuple[int, int] = (800, 800),
    margin: int = 40,
    occupancy_grid: np.ndarray | None = None,
) -> np.ndarray:
    """탑다운 맵 렌더링 (경로 + 특징점 단순 표시)
    
    Args:
        trajectory_xy: 카메라 경로 좌표 (N, 2)
        map_xy: 맵 특징점 좌표 (M, 2)
        is_floor_array: 바닥 여부 배열 (M,), True면 바닥(빨간색), False면 사물(파란색)
        canvas_size: 출력 이미지 크기 (H, W)
        margin: 여백
        occupancy_grid: 사용하지 않음 (호환성 유지)
    
    Legend:
        - 파란색 점: 사물(구조물) 위치
        - 빨간색 점: 바닥 위치
        - 녹색 선: 카메라 이동 경로
        - 파란점: 시작점
        - 빨간점: 끝점
    """
    h, w = canvas_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)  # 흰 배경

    if len(trajectory_xy) == 0 and len(map_xy) == 0:
        return canvas

    all_pts = []
    if len(trajectory_xy) > 0:
        all_pts.append(trajectory_xy)
    if len(map_xy) > 0:
        all_pts.append(map_xy)
    all_pts = np.vstack(all_pts)

    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    scale_x = (w - 2 * margin) / span[0]
    scale_y = (h - 2 * margin) / span[1]
    scale = min(scale_x, scale_y)

    def project(points: np.ndarray) -> np.ndarray:
        p = (points - min_xy) * scale
        p[:, 0] += margin
        p[:, 1] += margin
        p[:, 1] = h - p[:, 1]
        return p.astype(np.int32)

    # 1. 특징점 표시 (바닥/사물 구분)
    if len(map_xy) > 0:
        filtered_map_xy = filter_sparse_points(map_xy, cell_size=0.5, min_count=1)
        map_px = project(filtered_map_xy)
        
        # 바닥/사물 정보가 있으면 필터링된 인덱스 추적
        if is_floor_array is not None and len(is_floor_array) == len(map_xy):
            # 필터링된 점들 중 어떤 것이 바닥인지 알아야 함
            # filter_sparse_points가 인덱스를 반환하지 않으므로, simple approach 사용
            for i, pt in enumerate(map_px):
                # is_floor 확률 계산: 필터링된 점 주변의 원본 점들 중 바닥 비율
                if is_floor_array[i if i < len(is_floor_array) else -1]:
                    cv2.circle(canvas, tuple(pt), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)  # 빨간색 (바닥)
                else:
                    cv2.circle(canvas, tuple(pt), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)  # 파란색 (사물)
        else:
            # 기본值: 모두 파란색
            for pt in map_px:
                cv2.circle(canvas, tuple(pt), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    # 2. 경로 표시 - 진한 녹색 선 (두께 증가)
    if len(trajectory_xy) > 1:
        traj_px = project(trajectory_xy)
        cv2.polylines(canvas, [traj_px], isClosed=False, color=(0, 150, 0), thickness=3, lineType=cv2.LINE_AA)
        # 시작점 - 주황색, 끝점 - 자주색
        cv2.circle(canvas, tuple(traj_px[0]), 7, (0, 165, 255), -1, lineType=cv2.LINE_AA)    # 시작: 주황
        cv2.circle(canvas, tuple(traj_px[-1]), 7, (255, 0, 255), -1, lineType=cv2.LINE_AA)   # 끝: 자주

    # 3. 범례 추가
    feature_count = len(filtered_map_xy) if len(map_xy) > 0 else 0
    floor_count = int(np.sum(is_floor_array)) if is_floor_array is not None else 0
    object_count = feature_count - floor_count
    
    cv2.putText(canvas, f"Features: {feature_count} (Floor: {floor_count}, Objects: {object_count})", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Blue=Objects | Red=Floor", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, f"Path: {len(trajectory_xy)}", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 1)
    
    return canvas


def render_topdown_map_with_confidence(
    trajectory_xy: np.ndarray,
    map_xy_inlier: np.ndarray,
    map_xy_outlier: np.ndarray,
    canvas_size: tuple[int, int] = (800, 800),
    margin: int = 40,
) -> np.ndarray:
    """신뢰도 기반 점 색상으로 구분된 특징점 맵 렌더링 (2D 맨아래 보기)
    
    - 녹색 (inlier): 높은 신뢰도 매칭점
    - 회색 (outlier): 낮은 신뢰도 점
    - 주황색 선: 카메라 궤적
    - 3D 포인트의 경우 처음 두 열만 사용 (z축 무시)
    """
    h, w = canvas_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    if len(trajectory_xy) == 0 and len(map_xy_inlier) == 0 and len(map_xy_outlier) == 0:
        return canvas

    all_pts = []
    if len(trajectory_xy) > 0:
        all_pts.append(trajectory_xy)
    if len(map_xy_inlier) > 0:
        # 3D 포인트의 경우 처음 두 열만 사용
        inlier_2d = map_xy_inlier[:, :2] if map_xy_inlier.shape[1] > 2 else map_xy_inlier
        all_pts.append(inlier_2d)
    if len(map_xy_outlier) > 0:
        # 3D 포인트의 경우 처음 두 열만 사용
        outlier_2d = map_xy_outlier[:, :2] if map_xy_outlier.shape[1] > 2 else map_xy_outlier
        all_pts.append(outlier_2d)
    
    if not all_pts:
        return canvas
    
    all_pts = np.vstack(all_pts)

    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    scale_x = (w - 2 * margin) / span[0]
    scale_y = (h - 2 * margin) / span[1]
    scale = min(scale_x, scale_y)

    def project(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        # 2D 포인트만 사용
        p = points[:, :2] if points.shape[1] > 2 else points
        p = (p - min_xy) * scale
        p[:, 0] += margin
        p[:, 1] += margin
        p[:, 1] = h - p[:, 1]
        return p.astype(np.int32)

    # 1. Outlier 점들 (회색, 작음)
    if len(map_xy_outlier) > 0:
        outlier_px = project(map_xy_outlier)
        for pt in outlier_px:
            cv2.circle(canvas, tuple(pt), 1, (150, 150, 150), -1, lineType=cv2.LINE_AA)

    # 2. Inlier 점들 (녹색, 더 큼)
    if len(map_xy_inlier) > 0:
        inlier_px = project(map_xy_inlier)
        for pt in inlier_px:
            cv2.circle(canvas, tuple(pt), 2, (0, 255, 100), -1, lineType=cv2.LINE_AA)

    # 3. 궤적 (주황색 선)
    if len(trajectory_xy) > 1:
        traj_px = project(trajectory_xy)
        cv2.polylines(canvas, [traj_px], isClosed=False, color=(255, 180, 0), thickness=3, lineType=cv2.LINE_AA)
        # 시작점 (파란색), 끝점 (빨간색)
        cv2.circle(canvas, tuple(traj_px[0]), 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(traj_px[-1]), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # 4. 범례 추가
    cv2.putText(canvas, f"Inlier: {len(map_xy_inlier)}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    cv2.putText(canvas, f"Outlier: {len(map_xy_outlier)}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    cv2.putText(canvas, f"Trajectory: {len(trajectory_xy)}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

    return canvas


def save_run_artifacts(output_dir: str, stats: Slam2DStats, trajectory_xy: np.ndarray, topdown_img: np.ndarray, map_xy_inlier: np.ndarray = None, map_xy_outlier: np.ndarray = None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savetxt(out / "trajectory_xy.txt", trajectory_xy, fmt="%.6f", delimiter=",")
    cv2.imwrite(str(out / "topdown_map.png"), topdown_img)

    # 포인트 클라우드를 PLY 형식으로 저장 (CloudCompare 등에서 시각화 가능)
    if map_xy_inlier is not None and len(map_xy_inlier) > 0:
        _save_ply(out / "map_points_inlier.ply", map_xy_inlier, color=(0, 255, 100))
    if map_xy_outlier is not None and len(map_xy_outlier) > 0:
        _save_ply(out / "map_points_outlier.ply", map_xy_outlier, color=(150, 150, 150))
    if trajectory_xy is not None and len(trajectory_xy) > 0:
        _save_ply(out / "trajectory.ply", trajectory_xy, color=(255, 180, 0))

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)


def _save_ply(filepath: Path, points: np.ndarray, color: tuple[int, int, int] = (0, 0, 0)):
    """점들을 PLY 형식으로 저장 (2D 또는 3D 지원)
    
    PLY는 텍스트 포맷으로 CloudCompare, MeshLab 등에서 열 수 있습니다.
    - 2D 포인트: z=0으로 자동 확장
    - 3D 포인트: 그대로 저장 (벽, 천장 등 수직 구조 포함)
    """
    if len(points) == 0:
        return
    
    # 2D 또는 3D 포인트 처리
    if points.shape[1] == 2:
        # 2D → 3D 확장 (z=0)
        points_3d = np.column_stack([points, np.zeros(len(points))])
    elif points.shape[1] == 3:
        # 이미 3D
        points_3d = points
    else:
        raise ValueError(f"포인트는 2D 또는 3D여야 합니다. 현재: {points.shape[1]}D")
    
    r, g, b = color
    
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt in points_3d:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")

