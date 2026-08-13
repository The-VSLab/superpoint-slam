from __future__ import annotations

import cv2
import numpy as np


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
    show_features: bool = True,
) -> np.ndarray:
    """탑다운 맵 렌더링 (경로 + 선택적 특징점 표시)

    Args:
        trajectory_xy: 카메라 경로 좌표 (N, 2)
        map_xy: 맵 특징점 좌표 (M, 2)
        is_floor_array: 바닥 여부 배열 (M,), True면 바닥(빨간색), False면 사물(파란색)
        canvas_size: 출력 이미지 크기 (H, W)
        margin: 여백
        show_features: False면 특징점을 그리지 않고 경로만 표시 (뷰 스케일도 경로 기준)

    Legend:
        - 파란색 점: 사물(구조물) 위치
        - 빨간색 점: 바닥 위치
        - 녹색 선: 카메라 이동 경로
        - 주황점: 시작점 / 자주점: 끝점
    """
    h, w = canvas_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)  # 흰 배경

    draw_features = show_features and len(map_xy) > 0

    if len(trajectory_xy) == 0 and len(map_xy) == 0:
        return canvas

    # 뷰 스케일 계산: 특징점을 '그리지 않아도' 맵 범위를 포함해 스케일을 잡아야
    # 특징점 표시 여부와 무관하게 궤적의 크기·위치가 동일하게 유지된다.
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

    # 1. 특징점 필터링(범례 계산에 필요) + 원 그리기(show_features=False면 그리기만 생략).
    #    → 특징점을 빼도 뷰 스케일/범례는 특징점 버전과 완전히 동일하게 유지됨.
    filtered_map_xy = np.empty((0, 2))
    if len(map_xy) > 0:
        filtered_map_xy = filter_sparse_points(map_xy, cell_size=1.0, min_count=1)
        if show_features:
            map_px = project(filtered_map_xy)
            # display every 2nd point to reduce clutter
            map_px = map_px[::2]

            # 바닥/사물 정보가 있으면 필터링된 인덱스 추적
            if is_floor_array is not None and len(is_floor_array) == len(map_xy):
                for i, pt in enumerate(map_px):
                    if is_floor_array[i if i < len(is_floor_array) else -1]:
                        cv2.circle(canvas, tuple(pt), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)  # 빨간색 (바닥)
                    else:
                        cv2.circle(canvas, tuple(pt), 1, (255, 0, 0), -1, lineType=cv2.LINE_AA)  # 파란색 (사물)
            else:
                # 기본: 모두 파란색
                for pt in map_px:
                    cv2.circle(canvas, tuple(pt), 1, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    # 2. 경로 표시 - 진한 녹색 선 (두께 증가)
    if len(trajectory_xy) > 1:
        traj_px = project(trajectory_xy)
        cv2.polylines(canvas, [traj_px], isClosed=False, color=(0, 150, 0), thickness=3, lineType=cv2.LINE_AA)
        # 시작점 - 주황색, 끝점 - 자주색
        cv2.circle(canvas, tuple(traj_px[0]), 7, (0, 165, 255), -1, lineType=cv2.LINE_AA)    # 시작: 주황
        cv2.circle(canvas, tuple(traj_px[-1]), 7, (255, 0, 255), -1, lineType=cv2.LINE_AA)   # 끝: 자주

    # 3. 범례 추가 — show_features 여부와 무관하게 동일 (특징점만 안 그릴 뿐)
    feature_count = len(filtered_map_xy) if len(map_xy) > 0 else 0
    floor_count = int(np.sum(is_floor_array)) if is_floor_array is not None else 0
    object_count = feature_count - floor_count

    cv2.putText(canvas, f"Features: {feature_count} (Floor: {floor_count}, Objects: {object_count})", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Blue=Objects | Red=Floor", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, f"Path: {len(trajectory_xy)}", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 1)

    return canvas
