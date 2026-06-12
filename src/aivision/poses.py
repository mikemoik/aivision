"""Extract per-frame camera poses + intrinsics from a COLMAP reconstruction
and render an AR overlay (3D axes + ground grid) onto the original video.

The exported poses follow the COLMAP convention: ``cam_from_world`` is a
right-handed rigid transform such that ``X_cam = R @ X_world + t``. The
camera optical axis is +Z, +X right, +Y down. World units are metres
(GPS-anchored).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation
from tqdm import tqdm


_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def _frame_index(name: str) -> int:
    m = _FRAME_RE.search(name)
    if not m:
        raise ValueError(f"unexpected image name: {name}")
    return int(m.group(1))


def _intrinsics_from_camera(cam: pycolmap.Camera) -> dict:
    """Flatten a pycolmap camera to a JSON-friendly dict."""
    out = {
        "model": cam.model_name,
        "width": int(cam.width),
        "height": int(cam.height),
        "params": [float(p) for p in cam.params],
    }
    # Convenience fields for the common SIMPLE_RADIAL case (f, cx, cy, k1).
    if cam.model_name == "SIMPLE_RADIAL":
        f, cx, cy, k1 = cam.params
        out.update({"fx": float(f), "fy": float(f), "cx": float(cx),
                    "cy": float(cy), "k1": float(k1)})
    return out


def export_poses(
    reconstruction_dir: Path, out_json: Path,
) -> dict[str, dict]:
    """Dump per-frame camera poses + intrinsics to JSON."""
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    cam = rec.cameras[next(iter(rec.cameras))]
    intr = _intrinsics_from_camera(cam)

    frames: list[dict] = []
    for img_id, img in rec.images.items():
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=np.float64)
        center = (-R.T @ t).tolist()
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        frames.append({
            "name": img.name,
            "image_id": int(img_id),
            "frame_index": _frame_index(img.name),
            "R_cam_from_world": R.tolist(),
            "t_cam_from_world": t.tolist(),
            "qvec_xyzw": [float(qx), float(qy), float(qz), float(qw)],
            "camera_center_world": [float(c) for c in center],
        })
    frames.sort(key=lambda d: d["frame_index"])

    payload = {"intrinsics": intr, "frames": frames}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[poses] wrote {out_json} — {len(frames)} frames, "
          f"camera={intr['model']} {intr['width']}x{intr['height']}")
    return payload


def _project_points(
    pts_world: np.ndarray, R: np.ndarray, t: np.ndarray, intr: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP-style projection with SIMPLE_RADIAL distortion. Returns (uv,
    depth) — depth is camera Z, used to clip points behind the camera."""
    X_cam = pts_world @ R.T + t  # (N, 3)
    z = X_cam[:, 2]
    safe = np.maximum(z, 1e-6)
    x_n = X_cam[:, 0] / safe
    y_n = X_cam[:, 1] / safe
    r2 = x_n * x_n + y_n * y_n
    k1 = intr.get("k1", 0.0)
    radial = 1.0 + k1 * r2
    x_d = x_n * radial
    y_d = y_n * radial
    u = intr["fx"] * x_d + intr["cx"]
    v = intr["fy"] * y_d + intr["cy"]
    return np.stack([u, v], axis=1), z


def _make_ground_grid(
    centers: np.ndarray,
    cell: float = 2.0,
    extent: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a metric ground grid in world coordinates around the camera
    track. Returns (line_starts, line_ends), both (N, 3)."""
    cx, _, cz = centers.mean(axis=0)
    # Estimate ground height as the lowest 5th percentile of camera Y (since
    # we anchored to GPS positions which lie on the road, this is roughly
    # the ground plane). Camera positions are roughly on top of the road, so
    # offset downward by 1.5 m to approximate the ground.
    ground_y = float(np.percentile(centers[:, 1], 50)) + 1.5

    xs = np.arange(cx - extent, cx + extent + cell, cell)
    zs = np.arange(cz - extent, cz + extent + cell, cell)
    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    for x in xs:
        starts.append((float(x), ground_y, float(zs[0])))
        ends.append((float(x), ground_y, float(zs[-1])))
    for z in zs:
        starts.append((float(xs[0]), ground_y, float(z)))
        ends.append((float(xs[-1]), ground_y, float(z)))
    return np.asarray(starts), np.asarray(ends)


def _draw_3d_axes(
    img: np.ndarray, origin: np.ndarray, R: np.ndarray, t: np.ndarray,
    intr: dict, length: float = 1.0,
) -> None:
    """Project a right-handed XYZ triad anchored at `origin` (world coords)."""
    pts = np.stack([
        origin,
        origin + np.array([length, 0, 0]),
        origin + np.array([0, length, 0]),
        origin + np.array([0, 0, length]),
    ], axis=0)
    uv, z = _project_points(pts, R, t, intr)
    if (z < 0.1).any():
        return
    o = tuple(int(round(c)) for c in uv[0])
    cv2.line(img, o, tuple(int(round(c)) for c in uv[1]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(img, o, tuple(int(round(c)) for c in uv[2]), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(img, o, tuple(int(round(c)) for c in uv[3]), (255, 0, 0), 3, cv2.LINE_AA)


def _user_frame_from_first_camera(
    first_cam_from_world: pycolmap.Rigid3d,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the rigid transform that maps a user-frame point to the recon's
    world frame, where the user frame is defined by the first camera as:

        +x  → camera forward  (OpenCV camera +Z)
        +y  → camera left     (OpenCV camera −X)
        +z  → camera up       (OpenCV camera −Y)
        origin → first camera's projection centre

    Returns ``(R, t)`` so that ``p_world_recon = R @ p_user + t``.
    """
    # Columns of R_cam1_from_user express user-axes in cam1's OpenCV basis.
    R_cam1_from_user = np.array([
        [0.0, -1.0,  0.0],   # user +x = +z_cam1; user +y = -x_cam1; user +z = -y_cam1
        [0.0,  0.0, -1.0],
        [1.0,  0.0,  0.0],
    ])
    R_world_from_cam1 = first_cam_from_world.rotation.matrix().T
    t_world_from_cam1 = -R_world_from_cam1 @ np.asarray(first_cam_from_world.translation)
    R = R_world_from_cam1 @ R_cam1_from_user
    return R, t_world_from_cam1


def _build_xy_checker_cells(
    z: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    cell: float,
) -> list[np.ndarray]:
    """Checker cells parallel to the xy-plane at the given height ``z``,
    expressed in the user frame. Cells are axis-aligned with x (forward) and
    y (left). Returns the painted-cell corner arrays (every other tile)."""
    nx = int(round((x_range[1] - x_range[0]) / cell))
    ny = int(round((y_range[1] - y_range[0]) / cell))
    cells: list[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                continue
            x0 = x_range[0] + i * cell
            x1 = x_range[0] + (i + 1) * cell
            y0 = y_range[0] + j * cell
            y1 = y_range[0] + (j + 1) * cell
            cells.append(np.array([
                [x0, y0, z],
                [x1, y0, z],
                [x1, y1, z],
                [x0, y1, z],
            ], dtype=np.float64))
    return cells


def render_checker_overlay(
    reconstruction_dir: Path,
    frames_dir: Path,
    out_mp4: Path,
    fps: int = 30,
    plane_z: float = 3.5,
    cell: float = 2.0,
    x_range: tuple[float, float] = (-10.0, 80.0),
    y_range: tuple[float, float] = (-15.0, 15.0),
    color_bgr: tuple[int, int, int] = (220, 90, 20),  # translucent blue (BGR)
    alpha: float = 0.45,
) -> Path:
    """Project a static translucent blue checkerboard onto every frame.

    The world coordinate system is defined by the first frame's camera:
    origin at that camera's projection centre, +x = forward (where the car
    was pointing), +y = left, +z = up. The checkerboard lies in the plane
    ``z = plane_z`` (default 3.5 m above the first camera, i.e. ~5 m above
    the road assuming a 1.5 m camera-above-road offset), spans
    ``x_range`` along the forward axis and ``y_range`` laterally, and stays
    fixed in this world frame as the camera moves through it."""
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    cam = rec.cameras[next(iter(rec.cameras))]
    intr = _intrinsics_from_camera(cam)

    img_pairs = sorted(
        ((img_id, img) for img_id, img in rec.images.items()),
        key=lambda kv: _frame_index(kv[1].name),
    )

    first_pose = img_pairs[0][1].cam_from_world()
    R_world_from_user, t_world_from_user = _user_frame_from_first_camera(first_pose)

    cells_user = _build_xy_checker_cells(
        z=plane_z, x_range=x_range, y_range=y_range, cell=cell,
    )
    # Pre-transform every cell into recon-world coords once — the board is
    # static in the user frame, so it's static in the recon world too.
    cells_world = [
        (corners @ R_world_from_user.T) + t_world_from_user for corners in cells_user
    ]
    print(
        f"[checker] static checker at z={plane_z:.1f} m in first-camera frame "
        f"(x∈{x_range}, y∈{y_range}, cell={cell} m → {len(cells_world)} painted cells)"
    )

    sample = cv2.imread(str(frames_dir / img_pairs[0][1].name))
    h, w = sample.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter.fourcc(*"mp4v"),
                              fps, (w, h))

    for _, img in tqdm(img_pairs, desc=f"render {out_mp4.name}"):
        frame = cv2.imread(str(frames_dir / img.name))
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=np.float64)

        overlay = frame.copy()
        any_drawn = False
        for corners in cells_world:
            uv, z = _project_points(corners, R, t, intr)
            if (z < 0.5).any():
                continue
            if not np.isfinite(uv).all():
                continue
            poly = np.round(uv).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly], color_bgr, lineType=cv2.LINE_AA)
            cv2.polylines(overlay, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
            any_drawn = True

        if any_drawn:
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

        cv2.putText(frame, img.name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"checker z={plane_z:.1f} m (first-camera frame)",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[checker] wrote {out_mp4}")
    return out_mp4


def render_camera_balls_overlay(
    reconstruction_dir: Path,
    frames_dir: Path,
    out_mp4: Path,
    fps: int = 30,
    ball_diameter: float = 0.20,
    near_clip: float = 5.0,
    max_radius_px: int = 80,
) -> Path:
    """Project every reconstructed camera centre as a static sphere onto the
    original video. Each ball stays anchored to the world location where that
    camera was originally recovered, so the result is a visual proof that the
    trajectory is metric and self-consistent. Each ball is coloured once via
    a TURBO colormap on its distance from the frame-0 camera centre."""
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    cam = rec.cameras[next(iter(rec.cameras))]
    intr = _intrinsics_from_camera(cam)

    img_pairs = sorted(
        ((img_id, img) for img_id, img in rec.images.items()),
        key=lambda kv: _frame_index(kv[1].name),
    )

    ball_centers = np.stack([
        (-img.cam_from_world().rotation.matrix().T
         @ np.asarray(img.cam_from_world().translation, dtype=np.float64))
        for _, img in img_pairs
    ], axis=0)

    dist_from_frame0 = np.linalg.norm(ball_centers - ball_centers[0], axis=1)
    d_max = float(dist_from_frame0.max()) if len(dist_from_frame0) else 1.0
    norm = np.clip(dist_from_frame0 / max(d_max, 1e-6), 0.0, 1.0)
    ramp = (norm * 255.0).astype(np.uint8).reshape(-1, 1, 1)
    ball_colors = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO).reshape(-1, 3)
    print(f"[balls] {len(ball_centers)} camera centres, diameter={ball_diameter} m, "
          f"distance range 0–{d_max:.1f} m (TURBO colormap)")

    sample = cv2.imread(str(frames_dir / img_pairs[0][1].name))
    h, w = sample.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter.fourcc(*"mp4v"),
                              fps, (w, h))

    radius_world = ball_diameter * 0.5
    fx = intr["fx"]

    for _, img in tqdm(img_pairs, desc=f"render {out_mp4.name}"):
        frame = cv2.imread(str(frames_dir / img.name))
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=np.float64)

        uv, z = _project_points(ball_centers, R, t, intr)
        keep = (z >= near_clip) & np.isfinite(uv).all(axis=1)
        idxs = np.where(keep)[0]
        # Sort back-to-front so closer balls occlude farther ones.
        idxs = idxs[np.argsort(-z[idxs])]
        for i in idxs:
            u, v = uv[i]
            zi = z[i]
            radius_px = max(1, min(max_radius_px,
                                   int(round(fx * radius_world / zi))))
            center = (int(round(u)), int(round(v)))
            color = tuple(int(c) for c in ball_colors[i])
            cv2.circle(frame, center, radius_px, color, -1, cv2.LINE_AA)
            cv2.circle(frame, center, radius_px, (0, 0, 0), 1, cv2.LINE_AA)
            # Small inner highlight (upper-left) to read as 3D.
            hi_r = max(1, radius_px // 3)
            hi_off = max(1, radius_px // 3)
            hi_center = (center[0] - hi_off, center[1] - hi_off)
            cv2.circle(frame, hi_center, hi_r, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(frame, img.name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"camera balls: {len(ball_centers)} cameras, "
                    f"{int(round(ball_diameter * 100))} cm dia",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[balls] wrote {out_mp4}")
    return out_mp4


def render_camera_trail_overlay(
    reconstruction_dir: Path,
    frames_dir: Path,
    out_mp4: Path,
    fps: int = 30,
    ground_offset_m: float = 1.5,
    radius_world_m: float = 0.20,
    near_clip: float = 0.5,
    max_radius_px: int = 60,
) -> Path:
    """Render the camera trajectory projected onto the road surface.

    For each frame, the world camera centres of every other frame are
    pushed down by ``ground_offset_m`` along the *first camera's* +y axis
    (OpenCV camera +y = image-down, so this gives a point on the road
    directly below the original camera position). All these ground points
    are then projected onto the current frame using its pose + intrinsics.

    Colour scheme:
      - past frames (already driven over): blue → cyan, fading by recency
      - future frames (camera hasn't reached yet): green → yellow
      - current frame: white centre, bright outline

    If the recon's forward motion is geometrically correct, the trail
    should form a smooth line that recedes into the distance ahead of
    the car and passes under the bumper at the current frame.
    """
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    cam = rec.cameras[next(iter(rec.cameras))]
    intr = _intrinsics_from_camera(cam)

    img_pairs = sorted(
        ((img_id, img) for img_id, img in rec.images.items()),
        key=lambda kv: _frame_index(kv[1].name),
    )

    # Drop each camera centre along ITS OWN +y axis (image-down in the
    # camera frame). This avoids the world-frame gauge ambiguity that
    # happens when the GPS Umeyama anchoring only constrains positions,
    # not orientation around the motion axis.
    centres = np.stack([
        (-img.cam_from_world().rotation.matrix().T
         @ np.asarray(img.cam_from_world().translation, dtype=np.float64))
        for _, img in img_pairs
    ], axis=0)
    cam_downs_world = np.stack([
        img.cam_from_world().rotation.matrix().T @ np.array([0.0, 1.0, 0.0])
        for _, img in img_pairs
    ], axis=0)
    ground = centres + ground_offset_m * cam_downs_world

    n = len(img_pairs)
    # Per-marker BGR colour by relative time. We compute it per-current-frame
    # inside the loop because past/future flip.
    print(f"[trail] {n} camera positions, ground_offset={ground_offset_m:.2f} m, "
          f"radius={radius_world_m*100:.0f} cm")

    sample = cv2.imread(str(frames_dir / img_pairs[0][1].name))
    h, w = sample.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter.fourcc(*"mp4v"),
                              fps, (w, h))

    fx = intr["fx"]
    # Pre-compute cumulative arc length along the trajectory for distance labels.
    step_lens = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    arc_from_0 = np.concatenate([[0.0], np.cumsum(step_lens)])  # (n,)

    for current_i, (_, img) in enumerate(tqdm(img_pairs, desc=f"render {out_mp4.name}")):
        frame = cv2.imread(str(frames_dir / img.name))
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=np.float64)

        uv, z = _project_points(ground, R, t, intr)
        keep = (z >= near_clip) & np.isfinite(uv).all(axis=1)

        # Sort back-to-front so closer markers occlude farther ones.
        order = np.where(keep)[0]
        order = order[np.argsort(-z[order])]

        # Draw connecting line between consecutive trail points first
        # (so markers sit on top of the line).
        for k in range(n - 1):
            if not (keep[k] and keep[k+1]):
                continue
            u0, v0 = uv[k]; u1, v1 = uv[k+1]
            cv2.line(frame,
                     (int(round(u0)), int(round(v0))),
                     (int(round(u1)), int(round(v1))),
                     (220, 220, 220), 1, cv2.LINE_AA)

        for k in order:
            u, v = uv[k]; zi = z[k]
            radius_px = max(2, min(max_radius_px,
                                   int(round(fx * radius_world_m / zi))))
            centre = (int(round(u)), int(round(v)))

            if k == current_i:
                # Current frame: white interior, yellow ring
                cv2.circle(frame, centre, radius_px + 3, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, centre, radius_px, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, centre, radius_px, (0, 0, 0), 1, cv2.LINE_AA)
                continue

            # Label markers at "round" arc-length distances from the current
            # camera so the user can read distance directly off the image.
            arc_dist = arc_from_0[k] - arc_from_0[current_i]
            arc_round = abs(arc_dist - 5 * round(arc_dist / 5))
            if arc_round < step_lens.mean() * 0.5:
                lbl = f"{arc_dist:+.0f}m"
                pt = (centre[0] + radius_px + 6, centre[1] - 6)
                cv2.putText(frame, lbl, pt, cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, lbl, pt, cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if k < current_i:
                # Past — cyan-blue, fading older
                age = (current_i - k) / max(n - 1, 1)
                col = (int(255 * (1 - 0.4 * age)),  # B
                       int(180 * (1 - 0.5 * age)),  # G
                       0)                             # R
            else:
                # Future — green-yellow
                ahead = (k - current_i) / max(n - 1, 1)
                col = (0,                             # B
                       int(200 + 55 * (1 - ahead)),   # G
                       int(120 * ahead))              # R (more yellow further out)
            cv2.circle(frame, centre, radius_px, col, -1, cv2.LINE_AA)
            cv2.circle(frame, centre, radius_px, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, img.name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"camera trail (past=blue, future=green, current=white)",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[trail] wrote {out_mp4}")
    return out_mp4


def render_point_cloud_overlay(
    reconstruction_dir: Path,
    frames_dir: Path,
    out_mp4: Path,
    fps: int = 30,
    near_clip: float = 1.0,
    far_clip: float = 300.0,
    radius_px: int = 2,
    alpha: float = 0.75,
    colour: str = "rgb",
    min_track_len: int = 1,
    max_reproj_err: float = float("inf"),
    depth_band: tuple[float, float] | None = None,
) -> Path:
    """Project every 3D point in the reconstruction onto each frame and
    write the result as an MP4. Each point uses its stored COLMAP RGB
    (``colour="rgb"``) or a TURBO colormap on depth (``colour="depth"``).

    The visual check: world-anchored points should appear "frozen" to the
    scene as the camera moves through it. Drift in the recon shows up as
    points sliding off their real-world locations frame by frame.
    """
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    cam = rec.cameras[next(iter(rec.cameras))]
    intr = _intrinsics_from_camera(cam)

    img_pairs = sorted(
        ((img_id, img) for img_id, img in rec.images.items()),
        key=lambda kv: _frame_index(kv[1].name),
    )

    selected = [
        p for p in rec.points3D.values()
        if len(p.track.elements) >= min_track_len
        and (not p.has_error or p.error <= max_reproj_err)
    ]
    if depth_band is not None:
        d_lo, d_hi = depth_band
        pose_by_img = {
            img_id: (img.cam_from_world().rotation.matrix(),
                     np.asarray(img.cam_from_world().translation, dtype=np.float64))
            for img_id, img in rec.images.items()
        }
        in_band = []
        for p in selected:
            ds = []
            for elem in p.track.elements:
                R, t = pose_by_img[elem.image_id]
                z = float((R @ np.asarray(p.xyz) + t)[2])
                if z > 0: ds.append(z)
            if ds and d_lo <= np.median(ds) <= d_hi:
                in_band.append(p)
        selected = in_band
    n_total = rec.num_points3D()
    if not selected:
        raise RuntimeError(
            f"no points passed filter (track_len≥{min_track_len}, "
            f"err≤{max_reproj_err}, depth_band={depth_band}) out of {n_total}"
        )
    pts_xyz = np.stack(
        [np.asarray(p.xyz, dtype=np.float64) for p in selected], axis=0,
    )
    if colour == "rgb":
        # COLMAP stores RGB; OpenCV draws in BGR.
        pts_bgr = np.stack(
            [np.asarray(p.color, dtype=np.int32)[[2, 1, 0]] for p in selected],
            axis=0,
        )
    elif colour == "depth":
        # Pre-compute depth in the *first* camera as a stable global colour.
        first_pose = img_pairs[0][1].cam_from_world()
        R0 = first_pose.rotation.matrix()
        t0 = np.asarray(first_pose.translation, dtype=np.float64)
        z0 = pts_xyz @ R0.T[:, 2] + t0[2]
        z_clip = np.clip(z0, near_clip, far_clip)
        norm = (np.log(z_clip) - np.log(near_clip)) / (np.log(far_clip) - np.log(near_clip))
        ramp = (np.clip(norm, 0, 1) * 255).astype(np.uint8).reshape(-1, 1, 1)
        pts_bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO).reshape(-1, 3).astype(np.int32)
    else:
        raise ValueError(f"unknown colour mode: {colour!r}")

    print(f"[cloud] {len(pts_xyz)}/{n_total} points passed filter "
          f"(track_len≥{min_track_len}, err≤{max_reproj_err}), "
          f"{len(img_pairs)} frames, colour={colour}")

    sample = cv2.imread(str(frames_dir / img_pairs[0][1].name))
    h, w = sample.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter.fourcc(*"mp4v"),
                              fps, (w, h))

    for _, img in tqdm(img_pairs, desc=f"render {out_mp4.name}"):
        frame = cv2.imread(str(frames_dir / img.name))
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=np.float64)

        uv, z = _project_points(pts_xyz, R, t, intr)
        keep = (
            (z >= near_clip) & (z <= far_clip)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0) & (uv[:, 0] < w)
            & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )
        if not keep.any():
            writer.write(frame)
            continue

        uvi = np.round(uv[keep]).astype(np.int32)
        bgr_i = pts_bgr[keep]
        if alpha < 1.0:
            overlay = frame.copy()
            for (u, v), c in zip(uvi, bgr_i):
                cv2.circle(overlay, (int(u), int(v)), radius_px,
                           (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
        else:
            for (u, v), c in zip(uvi, bgr_i):
                cv2.circle(frame, (int(u), int(v)), radius_px,
                           (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)

        cv2.putText(frame, img.name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"point cloud ({len(pts_xyz)} pts, colour={colour})",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[cloud] wrote {out_mp4}")
    return out_mp4


def render_ar_overlay(
    reconstruction_dir: Path,
    frames_dir: Path,
    out_mp4: Path,
    fps: int = 30,
    grid_extent: float = 40.0,
    grid_cell: float = 2.0,
    anchor_offset_forward: float = 8.0,
) -> Path:
    """Render an AR overlay: 3D axes anchored a few metres ahead of the
    first camera + a metric ground grid spanning the route, drawn on each
    original frame using the reconstruction's poses + intrinsics.

    If the camera motion is rigid and the BA is GPS-anchored, the grid and
    axes should stay locked to the road surface across the whole clip.
    """
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    cam = rec.cameras[next(iter(rec.cameras))]
    intr = _intrinsics_from_camera(cam)

    # Sort images by frame index.
    img_pairs = sorted(
        ((img_id, img) for img_id, img in rec.images.items()),
        key=lambda kv: _frame_index(kv[1].name),
    )
    centers = np.stack([
        (-img.cam_from_world().rotation.matrix().T @ img.cam_from_world().translation)
        for _, img in img_pairs
    ], axis=0)

    grid_starts, grid_ends = _make_ground_grid(centers, cell=grid_cell, extent=grid_extent)

    # Anchor the axes at a fixed world point ahead of the first camera, on
    # the ground plane.
    first_R = img_pairs[0][1].cam_from_world().rotation.matrix()
    forward_world = first_R.T @ np.array([0.0, 0.0, 1.0])
    forward_world[1] = 0.0  # project onto horizontal plane
    n = np.linalg.norm(forward_world)
    if n < 1e-6:
        forward_world = np.array([1.0, 0.0, 0.0])
    else:
        forward_world /= n
    ground_y = float(np.percentile(centers[:, 1], 50)) + 1.5
    anchor = centers[0].copy()
    anchor[1] = ground_y
    anchor = anchor + forward_world * anchor_offset_forward

    sample = cv2.imread(str(frames_dir / img_pairs[0][1].name))
    h, w = sample.shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter.fourcc(*"mp4v"),
                              fps, (w, h))

    for img_id, img in tqdm(img_pairs, desc=f"render {out_mp4.name}"):
        frame = cv2.imread(str(frames_dir / img.name))
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=np.float64)

        # Project grid lines, clipping segments where either endpoint is behind cam.
        uv_s, z_s = _project_points(grid_starts, R, t, intr)
        uv_e, z_e = _project_points(grid_ends, R, t, intr)
        for (us, vs), (ue, ve), zs_, ze_ in zip(uv_s, uv_e, z_s, z_e):
            if zs_ < 0.5 or ze_ < 0.5:
                continue
            if not (np.isfinite(us) and np.isfinite(vs) and np.isfinite(ue) and np.isfinite(ve)):
                continue
            cv2.line(frame,
                     (int(round(us)), int(round(vs))),
                     (int(round(ue)), int(round(ve))),
                     (180, 180, 180), 1, cv2.LINE_AA)

        _draw_3d_axes(frame, anchor, R, t, intr, length=1.5)

        cv2.putText(frame, img.name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[ar] wrote {out_mp4}")
    return out_mp4
