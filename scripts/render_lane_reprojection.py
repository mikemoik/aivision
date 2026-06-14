"""Reproject the fitted 3D ego-lane polylines back into the RGB video.

Takes the two open polylines fit through the time-integrated lane map
(`outputs/ego_lanes/lane_map/lane_polylines.json`, in the first-camera / world
frame, metric metres) and projects them into every frame of the colour-left
sequence (`image_2`) using the estimated cam0->world poses + the rectified K.

Per frame t with pose T_t = [R_t | C_t] (cam0->world):
    X_cam = R_t.T @ (X_world - C_t)        # world -> cam0 of frame t
    u = fx*X/Z + cx,  v = fy*Y/Z + cy      # project (Z>0)
The image_2/image_0 ~6.6 cm baseline is negligible vs stereo+pose noise (see
build_lane_map.py), so we project straight into the cam0 frame the poses live in
and overlay on the colour-left frames (render-on-RGB convention).

Segments crossing the near plane are clipped so the polyline doesn't pop as the
car drives past vertices. Output: an mp4 with left (blue) + right (pink) borders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence

# BGR draw colours matching the lane-map (ego-left blue, ego-right pink)
COLOR = {"left": (255, 130, 30), "right": (190, 60, 255)}
ZNEAR = 5.0  # metres; clip near segments (very-close verts + pose-drift wobble blow up in px)


def read_calib(calib_path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in calib_path.read_text().strip().splitlines():
        k, *vals = line.replace(":", "").split()
        out[k] = np.array(vals, np.float64).reshape(3, 4)
    return out


def read_poses(path: Path) -> np.ndarray:
    """KITTI 3x4 cam0->world rows -> (N,4,4)."""
    raw = np.loadtxt(path, dtype=np.float64)
    n = raw.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :4] = raw.reshape(n, 3, 4)
    return T


def project_polyline(Pw: np.ndarray, Rinv: np.ndarray, C: np.ndarray,
                     fx, fy, cx, cy) -> list[np.ndarray]:
    """World polyline -> list of pixel polylines (split at near-plane crossings)."""
    Pc = (Pw - C) @ Rinv.T                      # (M,3) in cam0 frame of this view
    segs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []

    def px(p):
        return (fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy)

    for i in range(len(Pc) - 1):
        a, b = Pc[i], Pc[i + 1]
        za, zb = a[2], b[2]
        if za >= ZNEAR and zb >= ZNEAR:                 # both in front
            if not cur:
                cur.append(px(a))
            cur.append(px(b))
        elif za < ZNEAR and zb < ZNEAR:                 # both behind
            if cur:
                segs.append(cur); cur = []
        else:                                           # crosses near plane -> clip
            t = (ZNEAR - za) / (zb - za)
            clip = a + t * (b - a)
            if za >= ZNEAR:                             # a in front, b behind
                if not cur:
                    cur.append(px(a))
                cur.append(px(clip))
                segs.append(cur); cur = []
            else:                                       # a behind, b in front
                if cur:
                    segs.append(cur)
                cur = [px(clip), px(b)]
    if cur:
        segs.append(cur)
    return [np.round(np.array(s)).astype(np.int32) for s in segs if len(s) >= 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/04"))
    ap.add_argument("--poses", type=Path,
                    default=Path("outputs/pocs/orb_slam3/CameraTrajectory.txt"))
    ap.add_argument("--polylines", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_polylines.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_reprojection.mp4"))
    ap.add_argument("--fps", type=float, default=10.0)   # KITTI odometry is 10 Hz
    ap.add_argument("--thickness", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    seq = load_sequence(args.seq_dir, camera="image_2")   # colour-left (render-on-RGB)
    W, H, N = seq.width, seq.height, seq.n_frames
    end = N if args.end < 0 else min(args.end, N)

    calib = read_calib(args.seq_dir / "calib.txt")
    K = calib["P2"][:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    poses = read_poses(args.poses)
    if len(poses) < end:
        raise SystemExit(f"poses ({len(poses)}) fewer than frames ({end})")

    data = json.loads(args.polylines.read_text())
    lanes = {k: np.array(data[k], np.float64) for k in ("left", "right")}
    print(f"[reproj] left={len(lanes['left'])} right={len(lanes['right'])} verts; "
          f"fx={fx:.1f} {W}x{H} {end} frames")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"),
                             args.fps, (W, H))
    drawn = 0
    for t in tqdm(range(args.start, end), desc="lane reproj"):
        im = cv2.imread(str(seq.image_paths[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        Rinv = R.T
        any_seg = False
        for name in ("left", "right"):
            for seg in project_polyline(lanes[name], Rinv, C, fx, fy, cx, cy):
                cv2.polylines(im, [seg], isClosed=False, color=COLOR[name],
                              thickness=args.thickness, lineType=cv2.LINE_AA)
                any_seg = True
        drawn += any_seg
        writer.write(im)
    writer.release()
    print(f"[reproj] wrote {args.out} ({drawn}/{end - args.start} frames had lane overlay)")


if __name__ == "__main__":
    main()
