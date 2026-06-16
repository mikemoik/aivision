"""Render the cam2 (image_2, left colour) RGB stream with overlay guide lines.

Always draws a yellow vertical image centre line (top to bottom). Optionally also
projects, into cam2, two ground-referenced guides:

  * ``--ego-centerline`` — the GREEN ego-lane centreline (the per-frame midline of
    the map's left/right ego-lane borders, ``map_adas.json``, f_veh_t0), lifted into
    the current view via the bundle's vehicle poses.
  * ``--fveh-arrow`` — a RED arrow of ``--arrow-len`` m along the f_veh +X (forward)
    axis, lying on the road plane (z = -0.93 m in f_veh), drawn in the CURRENT
    vehicle frame so it always points straight ahead of the car.

Projection uses the single-source cam2 extrinsics + rectified intrinsics; polylines
are clipped at the near plane. Renders on RGB per the project convention.

    uv run python scripts/render_cam2_midline.py [--map kitti-seq04]
        [--width-px 10] [--ego-centerline] [--fveh-arrow] [--arrow-len 20]
        [--fps 10] [--out map/kitti-seq04/output/cam2_midline.mp4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.calibration import KittiCalibration
from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas
from aivision.map.objects import MapClass

YELLOW = (0, 255, 255)  # BGR
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GROUND_Z_FVEH = -0.93   # road plane height in f_veh (origin is 0.93 m above road)
ZNEAR = 0.5


def _T(rows3x4: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :4] = np.asarray(rows3x4, float).reshape(3, 4)
    return T


def _resample(P: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points uniformly in arc length."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(s, d, P[:, i]) for i in range(3)])


def ego_centerline(amap: MapAdas, n: int = 160) -> np.ndarray | None:
    """Midline of the two ego-lane borders (f_veh_t0), arc-length matched."""
    lanes = {o.attributes.get("side", "left"):
             np.array([v.mean for v in o.geometry["line"].verts], float)
             for o in amap.of_class(MapClass.LANE_MARKING)}
    L, R = lanes.get("left"), lanes.get("right")
    if L is None or R is None or len(L) < 2 or len(R) < 2:
        return None
    return 0.5 * (_resample(L, n) + _resample(R, n))


def project_polyline(Pc: np.ndarray, K, znear: float = ZNEAR) -> list[np.ndarray]:
    """Camera-frame polyline -> pixel segments, split at near-plane crossings."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def px(p):
        return (fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy)

    segs, cur = [], []
    for i in range(len(Pc) - 1):
        a, b = Pc[i], Pc[i + 1]
        za, zb = a[2], b[2]
        if za >= znear and zb >= znear:
            if not cur:
                cur.append(px(a))
            cur.append(px(b))
        elif za < znear and zb < znear:
            if cur:
                segs.append(cur); cur = []
        else:
            t = (znear - za) / (zb - za)
            clip = a + t * (b - a)
            if za >= znear:
                if not cur:
                    cur.append(px(a))
                cur.append(px(clip)); segs.append(cur); cur = []
            else:
                if cur:
                    segs.append(cur)
                cur = [px(clip), px(b)]
    if cur:
        segs.append(cur)
    return [np.round(np.array(s)).astype(np.int32) for s in segs if len(s) >= 2]


def fveh_arrow_pts(length: float) -> list[np.ndarray]:
    """Arrow polylines in f_veh: a shaft along +X on the ground plane, plus barbs."""
    zg = GROUND_Z_FVEH
    barb, half = min(1.5, 0.1 * length), max(0.4, 0.03 * length)
    shaft = np.array([[0.0, 0.0, zg], [length, 0.0, zg]])
    headL = np.array([[length, 0.0, zg], [length - barb, +half, zg]])
    headR = np.array([[length, 0.0, zg], [length - barb, -half, zg]])
    return [shaft, headL, headR]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--width-px", type=int, default=10, help="centre-line thickness in pixels")
    ap.add_argument("--no-midline", action="store_true",
                    help="omit the yellow image centre line")
    ap.add_argument("--ego-centerline", action="store_true",
                    help="project the green ego-lane centreline into cam2")
    ap.add_argument("--fveh-arrow", action="store_true",
                    help="draw the red f_veh +X ground arrow")
    ap.add_argument("--arrow-len", type=float, default=20.0, help="arrow length [m]")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    frames = sorted((layout.input / "odometry" / "image_2").glob("*.png"))
    if not frames:
        raise SystemExit("no image_2 frames found")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    xc = w // 2
    out = args.out or (layout.output / "cam2_midline.mp4")

    cal = KittiCalibration.load()
    K = cal.K("cam2")
    T_cam2_fveh = np.linalg.inv(cal.T_fveh("cam2"))     # f_veh(t) -> cam2

    # world (f_veh_t0) ego centreline + per-frame vehicle poses (only if needed)
    centerline_w = poses_w_fveh = None
    if args.ego_centerline:
        centerline_w = ego_centerline(MapAdas.load(layout.db))
        if centerline_w is None:
            raise SystemExit("map has no left+right ego-lane borders for a centreline")
        bundle = json.loads((layout.root / "camera.json").read_text())
        poses_w_fveh = np.array(bundle["motion"]["poses_vehicle"]).reshape(-1, 3, 4)

    arrow = fveh_arrow_pts(args.arrow_len) if args.fveh_arrow else None

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (w, h))
    n = len(frames) if poses_w_fveh is None else min(len(frames), len(poses_w_fveh))
    for t in tqdm(range(n), desc="cam2 overlay"):
        im = cv2.imread(str(frames[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))

        # green ego-lane centreline: world -> f_veh(t) -> cam2
        if centerline_w is not None:
            T_cam2_w = T_cam2_fveh @ np.linalg.inv(_T(poses_w_fveh[t]))
            Pc = (centerline_w @ T_cam2_w[:3, :3].T) + T_cam2_w[:3, 3]
            for seg in project_polyline(Pc, K):
                cv2.polylines(im, [seg], False, GREEN, 3, cv2.LINE_AA)

        # red f_veh +X ground arrow: f_veh(t) -> cam2 (no world pose needed)
        if arrow is not None:
            for poly in arrow:
                Pc = (poly @ T_cam2_fveh[:3, :3].T) + T_cam2_fveh[:3, 3]
                for seg in project_polyline(Pc, K):
                    cv2.polylines(im, [seg], False, RED, 3, cv2.LINE_AA)

        # yellow image centre line
        if not args.no_midline:
            cv2.line(im, (xc, 0), (xc, h), YELLOW, args.width_px)
        writer.write(im)
    writer.release()

    extras = []
    if not args.no_midline:
        extras.append(f"{args.width_px}px yellow centre line at x={xc}")
    if centerline_w is not None:
        extras.append("green ego centreline")
    if arrow is not None:
        extras.append(f"red {args.arrow_len:g}m f_veh arrow")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  "
          f"{n} frames {w}x{h}, {' + '.join(extras) or 'no overlays'}")


if __name__ == "__main__":
    main()
