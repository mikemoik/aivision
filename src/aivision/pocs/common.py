"""Shared eval + render harness for POC SLAM benchmarks on KITTI seq 04.

Reused by every `pocs/<method>.py` wrapper so each new POC is just a thin
subprocess shim + a driver.

Three pieces:
    load_tum_trajectory(path)    - TUM `tx ty tz qx qy qz qw` format (ORB-SLAM3)
    load_kitti_trajectory(path)  - 12-float-per-row 3x4 cam-to-world (KITTI GT)
    load_xyz_rgb(path)           - "x y z r g b" per line (ORB-SLAM3 patched dump)
    evaluate(poses, seq)         - Sim(3)-aligned ATE/RPE + per-axis residuals
    render_debug(...)            - trajectory PNG + orbit/reproj mp4s
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..kitti import (
    KittiSequence,
    absolute_trajectory_error,
    relative_pose_error,
    umeyama_similarity,
)


# ---------------------------------------------------------------------------
# Trajectory loaders
# ---------------------------------------------------------------------------


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """TUM/Hamilton quaternion (qx, qy, qz, qw) -> 3x3 rotation."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_tum_trajectory(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load a TUM-format trajectory (one `t tx ty tz qx qy qz qw` per line).

    Returns (poses, timestamps) where poses is (N, 4, 4) cam-to-world and
    timestamps is (N,) seconds. Skips comment / blank lines.
    """
    path = Path(path)
    rows: list[tuple[float, float, float, float, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append(tuple(float(x) for x in parts[:8]))  # type: ignore[arg-type]

    if not rows:
        raise ValueError(f"No trajectory rows parsed from {path}")

    arr = np.asarray(rows, dtype=np.float64)
    times = arr[:, 0].copy()
    poses = np.tile(np.eye(4), (len(arr), 1, 1))
    for i, (_, tx, ty, tz, qx, qy, qz, qw) in enumerate(rows):
        poses[i, :3, :3] = _quat_to_rot(qx, qy, qz, qw)
        poses[i, :3, 3] = (tx, ty, tz)
    return poses, times


def colorize_cloud_from_video(
    points_xyz: np.ndarray,
    cam_to_world: np.ndarray,
    K: np.ndarray,
    frame_paths: list[Path],
    z_min: float = 1.0,
    z_max: float = 200.0,
) -> np.ndarray:
    """Sample per-point RGB from the source frames.

    For each map point, iterate all frames once; whenever a frame puts the
    point at a smaller (positive) depth than any frame seen so far AND
    inside the image, replace its colour with that frame's pixel value.
    Result: each point ends up coloured from the frame whose camera was
    closest to it — the cleanest signal across the sequence.

    Output is `(N, 3)` floats in [0, 1], RGB order (matches Open3D).
    Points never visible (in front of camera + in-image) keep the default
    mid-grey.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    n_pts = points_xyz.shape[0]
    rgb = np.full((n_pts, 3), 0.5, dtype=np.float64)
    best_z = np.full(n_pts, np.inf, dtype=np.float64)
    pts_T = points_xyz.T  # (3, N) for the gemm

    for i in range(min(len(frame_paths), len(cam_to_world))):
        R_cw = cam_to_world[i, :3, :3]
        t_cw = cam_to_world[i, :3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        x_cam = (R_wc @ pts_T).T + t_wc
        z = x_cam[:, 2]
        closer = (z > z_min) & (z < z_max) & (z < best_z)
        if not closer.any():
            continue

        u = fx * x_cam[closer, 0] / z[closer] + cx
        v = fy * x_cam[closer, 1] / z[closer] + cy

        img = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not in_img.any():
            continue

        u_in, v_in = u[in_img].astype(int), v[in_img].astype(int)
        idx = np.flatnonzero(closer)[in_img]
        bgr = img[v_in, u_in]  # (M, 3) uint8
        rgb[idx] = bgr[:, ::-1] / 255.0  # BGR -> RGB float
        best_z[idx] = z[closer][in_img]

    return rgb


def load_xyz_rgb(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load `"x y z r g b"`-per-line point dumps. RGB is normalised to [0, 1]."""
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 3:
        raise ValueError(f"Expected at least 3 cols in {path}, got {raw.shape[1]}")
    xyz = raw[:, :3].copy()
    if raw.shape[1] >= 6:
        rgb = (raw[:, 3:6] / 255.0).clip(0.0, 1.0)
    else:
        rgb = np.full_like(xyz, 0.5)
    return xyz, rgb


def load_kitti_trajectory(path: Path | str) -> np.ndarray:
    """Load a KITTI 12-float-per-row 3x4 cam-to-world trajectory file."""
    path = Path(path)
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] != 12:
        raise ValueError(f"Expected 12 cols in {path}, got {raw.shape[1]}")
    n = raw.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = raw.reshape(n, 3, 4)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _per_axis_residuals(
    est_xyz: np.ndarray, gt_xyz: np.ndarray
) -> dict[str, float]:
    """RMS of the residual after Sim(3) alignment, decomposed onto GT axes.

    Treats the trajectory tangent as forward and recovers right/up via the
    initial GT pose orientation if available; absent that we fall back to
    world-frame X/Y/Z. The number we care about most for seq 04 is the
    *lateral* (cross-track) residual — it's what straight-line ambiguity
    inflates.
    """
    s, R, t = umeyama_similarity(est_xyz, gt_xyz)
    aligned = (s * R @ est_xyz.T).T + t
    res = aligned - gt_xyz
    return {
        "rms_x": float(np.sqrt((res[:, 0] ** 2).mean())),
        "rms_y": float(np.sqrt((res[:, 1] ** 2).mean())),
        "rms_z": float(np.sqrt((res[:, 2] ** 2).mean())),
    }


def evaluate(
    est_cam_to_world: np.ndarray,
    seq: KittiSequence,
    delta_rpe: int = 10,
) -> dict[str, object]:
    """Sim(3)-aligned ATE + RPE + per-axis residuals against the sequence GT.

    `est_cam_to_world` is (N, 4, 4). N must match the number of frames the
    method actually tracked — caller is responsible for slicing/mapping the
    GT to the right frame indices if the SLAM dropped frames.
    """
    if seq.gt_poses is None:
        raise ValueError(f"Sequence at {seq.seq_dir} has no GT poses")
    if est_cam_to_world.ndim != 3 or est_cam_to_world.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N,4,4) poses, got {est_cam_to_world.shape}")

    n_est = est_cam_to_world.shape[0]
    n_gt = seq.gt_poses.shape[0]
    n = min(n_est, n_gt)
    est_xyz = est_cam_to_world[:n, :3, 3]
    gt_xyz = seq.gt_positions()[:n]

    ate = absolute_trajectory_error(est_xyz, gt_xyz)
    rpe = relative_pose_error(est_xyz, gt_xyz, delta=delta_rpe)
    axes = _per_axis_residuals(est_xyz, gt_xyz)

    return {
        "n_est_frames": int(n_est),
        "n_gt_frames": int(n_gt),
        "n_evaluated": int(n),
        "ate": ate,
        "rpe": rpe,
        "axis_rms": axes,
    }


# ---------------------------------------------------------------------------
# Debug rendering
# ---------------------------------------------------------------------------


def _trajectory_png(
    est_xyz_aligned: np.ndarray,
    gt_xyz: np.ndarray,
    scale: float,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    # KITTI cam0: +X right, +Z forward -> top-down = X-Z
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], "-o", color="#1aa260", lw=2, ms=3, label="GT (RTK)")
    ax.plot(
        est_xyz_aligned[:, 0], est_xyz_aligned[:, 2],
        "-o", color="#e57373", lw=2, ms=3, label=f"est (Sim(3)-aligned, s={scale:.3f})",
    )
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 2], color="black", s=80, zorder=5, label="start")
    ax.set_xlabel("X (m, right)")
    ax.set_ylabel("Z (m, forward)")
    ax.set_aspect("equal", "datalim")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()


def _reproj_video(
    cam_to_world: np.ndarray,
    points_xyz: np.ndarray,
    points_rgb: np.ndarray | None,
    K: np.ndarray,
    frame_paths: list[Path],
    width: int,
    height: int,
    out_path: Path,
    trail_len: int = 60,
) -> None:
    """Project cloud + camera trail onto each source frame.

    `trail_len` = how many past camera centres to project as the ego-motion
    polyline (0 disables the trail). The trail covers the most recent N
    centres only so the line stays readable on long sequences.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    centers = cam_to_world[:, :3, 3]  # (N, 3) world frame

    tmp_dir = out_path.parent / (out_path.stem + "_frames")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    n = min(len(frame_paths), len(cam_to_world))
    for i in range(n):
        img = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # cam_to_world -> world_to_cam
        R_cw = cam_to_world[i, :3, :3]
        t_cw = cam_to_world[i, :3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw

        # --- 1. project cloud -----------------------------------------------
        x_cam = (R_wc @ points_xyz.T).T + t_wc
        z = x_cam[:, 2]
        keep = (z > 1.0) & (z < 200.0)
        n_kept = 0
        if keep.any():
            zc = z[keep]
            u = fx * x_cam[keep, 0] / zc + cx
            v = fy * x_cam[keep, 1] / zc + cy
            in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            u, v, zc = u[in_img], v[in_img], zc[in_img]
            n_kept = int(in_img.sum())
            if points_rgb is not None:
                # Use the cloud's own per-point RGB (BGR for OpenCV draw)
                rgb_sel = points_rgb[keep][in_img]
                colors = (rgb_sel[:, ::-1] * 255.0).clip(0, 255).astype(np.uint8)
            else:
                # Fall back to depth colourmap (turbo, near=red, far=blue)
                d_norm = np.clip(zc / 50.0, 0.0, 1.0)
                colors = cv2.applyColorMap(
                    (d_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
                ).reshape(-1, 3)
            for px, py, col in zip(u.astype(int), v.astype(int), colors):
                cv2.circle(img, (int(px), int(py)), 2, col.tolist(), -1)

        # --- 2. project ego-motion trail (past trail_len centres) -----------
        if trail_len > 0 and i > 0:
            start = max(0, i - trail_len)
            past = centers[start : i + 1]  # include current
            past_cam = (R_wc @ past.T).T + t_wc  # in current cam frame
            uv = []
            for pc in past_cam:
                if pc[2] > 0.05:
                    uu = fx * pc[0] / pc[2] + cx
                    vv = fy * pc[1] / pc[2] + cy
                    uv.append((uu, vv))
                else:
                    uv.append(None)
            # Polyline through the projected centres; segments with either
            # endpoint behind the camera are skipped.
            for k in range(1, len(uv)):
                a, b = uv[k - 1], uv[k]
                if a is None or b is None:
                    continue
                age = (len(uv) - k) / max(1, len(uv) - 1)  # 0=newest, 1=oldest
                col = (0, int(255 * (1 - age)), int(255 * age))  # BGR: green→red
                cv2.line(
                    img,
                    (int(a[0]), int(a[1])),
                    (int(b[0]), int(b[1])),
                    col, 2, cv2.LINE_AA,
                )
            # Current camera centre marker
            if uv[-1] is not None:
                cv2.circle(img, (int(uv[-1][0]), int(uv[-1][1])), 5,
                           (0, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(
            img, f"frame {i:03d}  pts={n_kept}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(tmp_dir / f"f_{i:05d}.png"), img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "10",
            "-i", str(tmp_dir / "f_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(out_path),
        ],
        check=True,
    )
    shutil.rmtree(tmp_dir)


def _orbit_video(
    points_xyz: np.ndarray,
    points_rgb: np.ndarray | None,
    centers: np.ndarray,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    n_frames: int = 180,
    fps: int = 30,
) -> None:
    """Open3D orbit flythrough of the point cloud + camera trail.

    Done inline (rather than calling aivision.render) because the POC inputs
    are raw arrays, not a vo.Scene / pycolmap.Reconstruction.
    """
    import open3d as o3d  # lazy: keeps import cost off the eval-only path

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_xyz)
    if points_rgb is not None:
        pc.colors = o3d.utility.Vector3dVector(points_rgb)

    n = len(centers)
    lines = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile((1.0, 0.8, 0.0), (len(lines), 1)))

    xyz = np.asarray(pc.points)
    lo, hi = np.percentile(xyz, [2, 98], axis=0)
    scene_center = 0.5 * (lo + hi)
    extent = float(np.linalg.norm(hi - lo))
    radius = 0.5 * extent
    cam_height = -0.5 * extent

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0.04, 0.05, 0.08, 1.0])

    mat_pc = o3d.visualization.rendering.MaterialRecord()
    mat_pc.shader = "defaultUnlit"
    mat_pc.point_size = 4.0
    scene.add_geometry("points", pc, mat_pc)

    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0
    scene.add_geometry("trajectory", ls, mat_line)

    fov_deg = 60.0
    near, far = max(extent * 1e-3, 0.01), max(extent * 10.0, 1.0)
    tmp_dir = out_path.parent / (out_path.stem + "_frames")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    up = np.array([0.0, -1.0, 0.0])

    for i in range(n_frames):
        angle = 2.0 * np.pi * i / n_frames
        eye = scene_center + np.array(
            [radius * np.cos(angle), cam_height, radius * np.sin(angle)]
        )
        renderer.setup_camera(fov_deg, scene_center, eye, up)
        scene.camera.set_projection(
            fov_deg, width / height, near, far,
            o3d.visualization.rendering.Camera.FovType.Vertical,
        )
        img = renderer.render_to_image()
        o3d.io.write_image(str(tmp_dir / f"f_{i:05d}.png"), img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "f_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(out_path),
        ],
        check=True,
    )
    shutil.rmtree(tmp_dir)


def render_debug(
    cam_to_world: np.ndarray,
    points_xyz: np.ndarray | None,
    points_rgb: np.ndarray | None,
    K: np.ndarray,
    frame_paths: list[Path],
    out_dir: Path,
    gt_xyz: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    title: str = "POC trajectory",
) -> dict[str, Path]:
    """Trajectory PNG + (if cloud present) orbit + reproj videos.

    Returns a dict of {name: path} for the artefacts that were actually
    written. POCs without a point cloud (ORB-SLAM3 Phase A) pass
    `points_xyz=None` and get just the trajectory PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artefacts: dict[str, Path] = {}

    n_est = cam_to_world.shape[0]
    n = min(n_est, len(gt_xyz))
    est_xyz = cam_to_world[:n, :3, 3]
    gt_used = gt_xyz[:n]

    s, R, t = umeyama_similarity(est_xyz, gt_used)
    est_aligned = (s * R @ est_xyz.T).T + t

    traj_png = out_dir / "trajectory.png"
    _trajectory_png(est_aligned, gt_used, s, traj_png, title)
    artefacts["trajectory_png"] = traj_png

    if points_xyz is not None and len(points_xyz) > 0:
        # Reproj uses the unscaled est poses against the unscaled points (the
        # SLAM solved them in the same frame). Sim(3) is for GT comparison only.
        if width is None or height is None:
            sample = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
            height, width = sample.shape[:2]
        reproj_mp4 = out_dir / "reproj.mp4"
        _reproj_video(
            cam_to_world, points_xyz, points_rgb, K,
            frame_paths, width, height, reproj_mp4,
        )
        artefacts["reproj_mp4"] = reproj_mp4

        centers = cam_to_world[:, :3, 3]
        orbit_mp4 = out_dir / "orbit.mp4"
        _orbit_video(points_xyz, points_rgb, centers, orbit_mp4)
        artefacts["orbit_mp4"] = orbit_mp4

    return artefacts
