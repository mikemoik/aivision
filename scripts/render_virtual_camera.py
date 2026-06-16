"""Render ONLY the virtual-camera sensor output onto the RGB stream.

The virtual sensor (``detections_by_frame.json``, read via
:class:`~aivision.map.virtual_camera.VirtualCameraSensor`) serves, per frame, the
static objects the camera saw that frame as :class:`MapObject` instances already in
CAMERA coordinates. So — unlike the map reprojection — no SLAM pose is needed: each
detection's geometry projects straight through K. This draws those projected objects
on the colour frames (``image_2``, per the project's render-on-RGB convention),
nothing else (no map DB, no dynamic layer):

  * poles          cyan stick      (shaft LineSegment)
  * guide posts     white stick     (point + height)
  * traffic signs   magenta box     (mount + face extents) + name/score
  * traffic lights  green housing   (mount + box extents)
  * ego lanes       blue/pink line  (LaneMarking polyline, ego-left/right)
  * road arrows     yellow glyph    (RoadArrow glyph polygon)

Each glyph is sized by the per-frame depth, so it lands on the actual detection in
the image (single-frame depth noise and all — that is the point of the sensor).

    uv run python scripts/render_virtual_camera.py [--map kitti-seq04] [--run-id RUN]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.map.layout import MapLayout
from aivision.map.virtual_camera import PixelNoise, VirtualCameraSensor

UP = np.array([0.0, -1.0, 0.0])  # up in the camera frame (-Y)
ZNEAR = 1.0

COL = {
    "pole": (230, 230, 0),           # cyan
    "guide_post": (245, 245, 245),   # white
    "sign": (220, 60, 220),          # magenta
    "traffic_light": (60, 220, 60),  # green
    "lane_left": (255, 130, 30),     # blue (ego-left)
    "lane_right": (190, 60, 255),    # pink (ego-right)
    "road_arrow": (0, 220, 255),     # yellow
}


def proj(P, fx, fy, cx, cy) -> np.ndarray | None:
    """Camera-frame points (N,3) -> int pixels; None if any vertex is too near."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if np.any(P[:, 2] < ZNEAR):
        return None
    u = fx * P[:, 0] / P[:, 2] + cx
    v = fy * P[:, 1] / P[:, 2] + cy
    return np.round(np.column_stack([u, v])).astype(np.int32)


def proj_polyline(P, fx, fy, cx, cy) -> list[np.ndarray]:
    """Camera-frame polyline -> pixel segments split at the near plane."""
    P = np.asarray(P, dtype=float)
    segs, cur = [], []
    for i in range(len(P)):
        if P[i, 2] >= ZNEAR:
            cur.append((fx * P[i, 0] / P[i, 2] + cx, fy * P[i, 1] / P[i, 2] + cy))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return [np.round(np.array(s)).astype(np.int32) for s in segs if len(s) >= 2]


def draw_detection(im, obj, fx, fy, cx, cy, show_range: bool) -> None:
    cls = obj.cls.value
    col = COL.get(cls, (200, 200, 200))
    a = obj.attributes
    label = None

    if cls == "pole":
        s = obj.geometry["shaft"]
        seg = proj([s.b.mean, s.a.mean], fx, fy, cx, cy)  # bottom, top
        if seg is None:
            return
        cv2.line(im, tuple(seg[0]), tuple(seg[1]), col, 2, cv2.LINE_AA)
        anchor = tuple(seg[1])
    elif cls == "guide_post":
        c = obj.geometry["point"].mean
        h = float(a.get("height") or 1.0)
        seg = proj([c - 0.5 * h * UP, c + 0.5 * h * UP], fx, fy, cx, cy)
        if seg is None:
            return
        cv2.line(im, tuple(seg[0]), tuple(seg[1]), col, 3, cv2.LINE_AA)
        cv2.circle(im, tuple(seg[1]), 3, (0, 0, 0), -1, cv2.LINE_AA)
        anchor = tuple(seg[1])
    elif cls == "sign":
        c = obj.geometry["mount"].translation
        half = float(obj.geometry["face"].extents[0])
        ctr = proj(c[None], fx, fy, cx, cy)
        if ctr is None:
            return
        u, v = int(ctr[0, 0]), int(ctr[0, 1])
        rpx = max(4, int(fx * half / float(c[2])))
        cv2.rectangle(im, (u - rpx, v - rpx), (u + rpx, v + rpx), col, 2, cv2.LINE_AA)
        anchor = (u - rpx, v - rpx)
        nm = a.get("name")
        if nm:
            sc = a.get("score")
            label = f"{nm}" + (f" {sc:.2f}" if sc is not None else "")
    elif cls == "traffic_light":
        c = obj.geometry["box"].pose.translation
        h = 2.0 * float(obj.geometry["box"].extents[1])
        seg = proj([c - 0.5 * h * UP, c + 0.5 * h * UP], fx, fy, cx, cy)
        if seg is None:
            return
        (_, v0), (u1, v1) = seg[0], seg[1]
        wpx = max(3, abs(v0 - v1) // 4)
        cv2.rectangle(im, (u1 - wpx, v1), (u1 + wpx, v0), col, 2, cv2.LINE_AA)
        anchor = (u1 - wpx, min(v0, v1))
    elif cls == "lane_marking":
        side = a.get("side", "left")
        col = COL["lane_right"] if side.endswith("right") else COL["lane_left"]
        verts = np.array([v.mean for v in obj.geometry["line"].verts])
        for seg in proj_polyline(verts, fx, fy, cx, cy):
            cv2.polylines(im, [seg], False, col, 3, cv2.LINE_AA)
        return  # polyline carries no single label anchor
    elif cls == "road_arrow":
        verts = np.array([v.mean for v in obj.geometry["glyph"].verts])
        poly = proj(verts, fx, fy, cx, cy)
        if poly is None:
            return
        ov = im.copy()
        cv2.fillPoly(ov, [poly], col)
        cv2.addWeighted(ov, 0.5, im, 0.5, 0, dst=im)
        anchor = (int(poly[:, 0].min()), int(poly[:, 1].min()))
        nm = a.get("cls")
        if nm:
            label = nm.replace("_arrow", "")
    else:
        return

    if label is None and show_range:
        label = f"{a.get('range_m', 0.0):.0f}m"
    if label:
        cv2.putText(im, label, (anchor[0], max(10, anchor[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def banner(im, t, n, ndet) -> None:
    ov = im.copy()
    cv2.rectangle(ov, (0, 0), (im.shape[1], 24), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, im, 0.45, 0, dst=im)
    txt = (f"virtual camera sensor  frame {t:03d}/{n - 1}  [{ndet} det]"
           "   pole sign light guidepost  blue/pink=lane  yellow=arrow")
    cv2.putText(im, txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (240, 240, 240), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--run-id", default=None,
                    help="run holding the sensor artifact (default: map's provenance.run_id)")
    ap.add_argument("--artifact", default=None,
                    help="path to detections_by_frame.json (overrides --map/--run-id)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-range", action="store_true", help="hide the per-object range label")
    ap.add_argument("--pixel-noise", type=float, default=0.0,
                    help="image-sensor pixel noise [px, 1-sigma] jittering every detection")
    ap.add_argument("--pixel-noise-seed", type=int, default=0)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    noise = (PixelNoise(sigma_px=args.pixel_noise, seed=args.pixel_noise_seed)
             if args.pixel_noise > 0 else None)
    if args.artifact:
        sensor = VirtualCameraSensor(args.artifact, noise=noise)
    else:
        sensor = VirtualCameraSensor.from_map(args.map, run_id=args.run_id, noise=noise)
    out = Path(args.out) if args.out else layout.output / "virtual_camera.mp4"

    fx, fy = sensor.K[0, 0], sensor.K[1, 1]
    cx, cy = sensor.K[0, 2], sensor.K[1, 2]
    frames = sorted((layout.input / "odometry" / sensor.camera).glob("*.png"))
    n = min(len(frames), sensor.n_frames)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    total = sum(len(sensor.detections(t)) for t in range(n))
    print(f"[vcam] {sensor!r}")
    print(f"[vcam] sources={[p.name for p in sensor.paths]}")
    print(f"[vcam] {n} frames {w}x{h} on {sensor.camera}  {total} detections")

    # Detections are now in the moving vehicle frame f_veh(t); to project straight
    # through K we bring them back to camera (cam0) coords via T_cam0_fveh.
    from aivision.calibration import KittiCalibration
    T_cam_fveh = np.linalg.inv(KittiCalibration.load().T_fveh("cam0"))

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (w, h))
    for t in tqdm(range(n), desc="vcam"):
        im = cv2.imread(str(frames[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        dets = [o.transformed(T_cam_fveh) for o in sensor.detections(t)]
        for obj in dets:
            draw_detection(im, obj, fx, fy, cx, cy, not args.no_range)
        banner(im, t, n, len(dets))
        writer.write(im)
    writer.release()
    print(f"[vcam] wrote {out}")


if __name__ == "__main__":
    main()
