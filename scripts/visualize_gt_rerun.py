"""
KITTI GT 경로 Rerun 시각화 스크립트
사용법: python visualize_gt_rerun.py --gt 08.txt
"""

import argparse
import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def load_kitti_poses(filepath: str) -> list[np.ndarray]:
    """KITTI GT 파일 로드 → 4x4 변환행렬 리스트"""
    poses = []
    with open(filepath, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def main():
    parser = argparse.ArgumentParser(description="KITTI GT Rerun Viewer")
    parser.add_argument("--gt", type=str, required=True, help="KITTI GT .txt 파일 경로")
    parser.add_argument("--name", type=str, default="KITTI GT", help="시각화 이름")
    args = parser.parse_args()

    # ── Rerun 초기화 ─────────────────────────────────────────
    rr.init(args.name, spawn=True)

    # 블루프린트: 3D 뷰 + 상단 정보 패널
    blueprint = rrb.Vertical(
        rrb.Spatial3DView(name="GT Trajectory 3D", origin="/world"),
        rrb.Spatial2DView(name="Top-Down View (XZ)", origin="/world/topdown"),
        row_shares=[3, 1],
    )
    rr.send_blueprint(blueprint)

    # ── GT 포즈 로드 ─────────────────────────────────────────
    poses = load_kitti_poses(args.gt)
    print(f"[INFO] Loaded {len(poses)} poses from {args.gt}")

    # ── 궤적 포인트 추출 (translation) ───────────────────────
    traj = np.array([T[:3, 3] for T in poses])   # (N, 3) — X, Y, Z

    # ── 3D 궤적 선 ───────────────────────────────────────────
    rr.log(
        "world/gt_trajectory",
        rr.LineStrips3D(
            [traj],
            colors=[[0, 255, 100]],   # 초록
            radii=0.05,
        ),
    )

    # ── 각 포즈마다 카메라 프러스텀 + 타임스텝 ───────────────
    for i, T in enumerate(poses):
        rr.set_time_sequence("frame", i)

        # 카메라 포즈 (월드 → 카메라 변환의 역)
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=T[:3, 3],
                mat3x3=T[:3, :3],
            ),
        )

        # 현재 위치 점
        rr.log(
            "world/current_pos",
            rr.Points3D(
                [T[:3, 3]],
                colors=[[255, 50, 50]],   # 빨강
                radii=0.15,
            ),
        )

    # ── 전체 궤적 포인트 (높이별 색상) ───────────────────────
    y_vals = traj[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    y_norm = (y_vals - y_min) / (y_max - y_min + 1e-8)

    colors = np.zeros((len(traj), 3), dtype=np.uint8)
    colors[:, 0] = (y_norm * 255).astype(np.uint8)        # R: 높을수록 빨강
    colors[:, 2] = ((1 - y_norm) * 255).astype(np.uint8)  # B: 낮을수록 파랑

    rr.log(
        "world/gt_points",
        rr.Points3D(
            traj,
            colors=colors,
            radii=0.05,
        ),
    )

    # ── 탑다운 (XZ 평면) ─────────────────────────────────────
    traj_xz = np.stack([traj[:, 0], traj[:, 2]], axis=1)  # (N, 2)
    rr.log(
        "world/topdown/trajectory",
        rr.LineStrips2D(
            [traj_xz],
            colors=[[0, 200, 255]],
            radii=1.0,
        ),
    )

    # 시작/끝 마킹
    rr.log(
        "world/topdown/start",
        rr.Points2D([traj_xz[0]], colors=[[0, 255, 0]], radii=5.0),
    )
    rr.log(
        "world/topdown/end",
        rr.Points2D([traj_xz[-1]], colors=[[255, 0, 0]], radii=5.0),
    )

    print("[INFO] Done. Rerun viewer should be open.")
    print(f"       Total distance: {np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.1f} m")
    print(f"       Frames: {len(poses)}")


if __name__ == "__main__":
    main()
