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


def render_topdown_map(
    trajectory_xy: np.ndarray,
    map_xy: np.ndarray,
    canvas_size: tuple[int, int] = (800, 800),
    margin: int = 40,
) -> np.ndarray:
    h, w = canvas_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

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

    if len(map_xy) > 0:
        map_px = project(map_xy)
        for pt in map_px:
            cv2.circle(canvas, tuple(pt), 1, (180, 180, 180), -1, lineType=cv2.LINE_AA)

    if len(trajectory_xy) > 1:
        traj_px = project(trajectory_xy)
        cv2.polylines(canvas, [traj_px], isClosed=False, color=(0, 180, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(traj_px[0]), 5, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(traj_px[-1]), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return canvas


def save_run_artifacts(output_dir: str, stats: Slam2DStats, trajectory_xy: np.ndarray, topdown_img: np.ndarray):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savetxt(out / "trajectory_xy.txt", trajectory_xy, fmt="%.6f", delimiter=",")
    cv2.imwrite(str(out / "topdown_map.png"), topdown_img)

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
