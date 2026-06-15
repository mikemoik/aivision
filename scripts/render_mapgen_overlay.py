"""T7 acceptance render: reproject a regenerated ADAS map onto the RGB video.

This is the *visual* acceptance gate for a map-generation run. It loads the
final ``map_adas.json`` (the product, not intermediate artifacts), the run's own
SLAM trajectory (so the reprojection lives in the same map frame ``M`` the
objects were triangulated in), and the colour-left frames (``image_2``), then
draws every map object reprojected into each view:

  * poles          cyan stick            (shaft LineSegment)
  * guide posts     white stick + cap     (point + height)
  * traffic signs   magenta box + class   (mount + face extents)
  * traffic lights  green housing         (mount + box extents)
  * lane markings   blue/pink polyline    (filtered lane model; the DB holds it)
  * road arrows     yellow filled glyph   (glyph Polyline)

Plus the per-frame DYNAMIC layer (from the standalone ``dynamic`` step's mask
cache, if the run has one): moving pixels tinted red, pot_dyn pixels orange.
This is NOT in the map DB (the map is static-only) — it is overlaid live from the
dynamic step's per-frame labels so the static map and the movers are seen together.

Projection (cam0->world pose ``T_t = [R_t | C_t]``):
    X_cam = R_t.T @ (X_world - C_t);  u = fx*X/Z + cx, v = fy*Y/Z + cy (Z>0)
Near segments are clipped at ``ZNEAR`` so glyphs don't pop as the car passes.
Renders on RGB per the project convention (see feedback_render_rgb).

    uv run python scripts/render_mapgen_overlay.py [--map kitti-seq04] [--run-id RUN]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas
from aivision.map.objects import MapClass

UP = np.array([0.0, -1.0, 0.0])  # world up in the cam frame
ZNEAR = 5.0

COL = {
    "pole": (230, 230, 0),       # cyan
    "guide_post": (245, 245, 245),  # white
    "sign": (220, 60, 220),      # magenta
    "traffic_light": (60, 220, 60),  # green
    "lane_left": (255, 130, 30),  # blue
    "lane_right": (190, 60, 255),  # pink
    "arrow": (0, 220, 255),      # yellow
}

# Dynamic-step label values (steps_dynamic.CLASS_NAMES) → BGR tints.
DYN_MOVING, DYN_POT_DYN = 1, 3
DYN_COL = {DYN_MOVING: (0, 0, 255), DYN_POT_DYN: (0, 140, 255)}  # red, orange


def load_dynamic_label(run, t: int, h: int, w: int):
    """Per-frame dynamic label map (HxW uint8) for source frame t, or None.

    The dynamic cache has one entry per pair (0 … N-2), so the last frame has no
    label. Returns None when the run has no dynamic step or t is out of range.
    """
    import cv2

    p = run.step_out("dynamic") / f"dynamic_{t:06d}.npz"
    if not p.exists():
        return None
    lab = np.load(p)["label"]
    if lab.shape != (h, w):
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
    return lab


def paint_dynamic(im, label, alpha: float = 0.5) -> None:
    """Tint moving (red) + pot_dyn (orange) pixels of the dynamic label in place."""
    ov = im.copy()
    touched = np.zeros(label.shape, bool)
    for val, col in DYN_COL.items():
        sel = label == val
        ov[sel] = col
        touched |= sel
    blend = (touched[..., None] * alpha)
    im[:] = (im * (1 - blend) + ov * blend).astype(np.uint8)


def load_dynamic_boxes(run):
    """Per-frame tracked mover boxes from the dynamic step's sidecar, or None."""
    import json
    p = run.step_out("dynamic") / "dynamic_boxes.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["frames"]


def paint_dynamic_boxes(im, label, boxes, alpha: float = 0.5) -> None:
    """Box variant: draw the detector boxes and tint label pixels ONLY inside them.

    Same box convention as render_motion_seg_full (red=dynamic, orange=pot_dyn).
    Restricting the tint to inside the boxes drops the dense classifier's
    off-object speckle (FOE/epipolar noise) so only detected movers are painted.
    """
    h, w = label.shape
    box_mask = np.zeros((h, w), bool)
    rects = []
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b["box"])
        x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
        y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        box_mask[y0:y1, x0:x1] = True
        rects.append((x0, y0, x1, y1, b))
    # tint moving/pot_dyn label pixels, but only where they fall inside a box
    ov = im.copy()
    touched = np.zeros((h, w), bool)
    for val, col in DYN_COL.items():
        sel = (label == val) & box_mask
        ov[sel] = col
        touched |= sel
    blend = (touched[..., None] * alpha)
    im[:] = (im * (1 - blend) + ov * blend).astype(np.uint8)
    # box outline + tag, coloured by the track-latched verdict
    for x0, y0, x1, y1, b in rects:
        col = DYN_COL[DYN_MOVING] if b["state"] == "moving" else DYN_COL[DYN_POT_DYN]
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
        tag = f"{b['name']} {'DYN' if b['state'] == 'moving' else 'pot_dyn'}"
        cv2.putText(im, tag, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)


def read_calib(p: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in p.read_text().strip().splitlines():
        k, *vals = line.replace(":", "").split()
        out[k] = np.array(vals, np.float64).reshape(3, 4)
    return out


def poses_from_traj(path: Path) -> np.ndarray:
    """trajectory.json (N,12 cam0->world rows) -> (N,4,4)."""
    raw = np.asarray(json.loads(path.read_text())["poses"], np.float64)
    n = raw.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :4] = raw.reshape(n, 3, 4)
    return T


def project_pts(Pw, Rinv, C, fx, fy, cx, cy) -> np.ndarray | None:
    """World pts -> int pixels; None if any vertex is behind the near plane."""
    Pc = (np.atleast_2d(Pw) - C) @ Rinv.T
    if np.any(Pc[:, 2] < ZNEAR):
        return None
    u = fx * Pc[:, 0] / Pc[:, 2] + cx
    v = fy * Pc[:, 1] / Pc[:, 2] + cy
    return np.round(np.column_stack([u, v])).astype(np.int32)


def project_polyline(Pw, Rinv, C, fx, fy, cx, cy) -> list[np.ndarray]:
    """World polyline -> pixel polylines split at near-plane crossings."""
    Pc = (np.asarray(Pw) - C) @ Rinv.T
    segs: list[list] = []
    cur: list = []

    def px(p):
        return (fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy)

    for i in range(len(Pc) - 1):
        a, b = Pc[i], Pc[i + 1]
        za, zb = a[2], b[2]
        if za >= ZNEAR and zb >= ZNEAR:
            if not cur:
                cur.append(px(a))
            cur.append(px(b))
        elif za < ZNEAR and zb < ZNEAR:
            if cur:
                segs.append(cur)
                cur = []
        else:
            t = (ZNEAR - za) / (zb - za)
            clip = a + t * (b - a)
            if za >= ZNEAR:
                if not cur:
                    cur.append(px(a))
                cur.append(px(clip))
                segs.append(cur)
                cur = []
            else:
                if cur:
                    segs.append(cur)
                cur = [px(clip), px(b)]
    if cur:
        segs.append(cur)
    return [np.round(np.array(s)).astype(np.int32) for s in segs if len(s) >= 2]


def collect(m: MapAdas) -> dict:
    """Pull each object's draw geometry (in frame M) out of the map.

    Lanes are the map's ``LaneMarking`` polylines — the DB now holds the
    filtered coupled lane model (assemble ingests ``lane_polylines_model.json``),
    so this is the clean, width-consistent road-surface representation.
    """
    poles, gposts, signs, tls, arrows = [], [], [], [], []
    lanes: dict[str, np.ndarray] = {}
    for o in m.of_class(MapClass.POLE):
        s = o.geometry["shaft"]
        poles.append(np.array([s.a.mean, s.b.mean]))  # top, bottom
    for o in m.of_class(MapClass.GUIDE_POST):
        c = o.geometry["point"].mean
        h = float(o.attributes.get("height") or 1.0)
        gposts.append((c, h))
    for o in m.of_class(MapClass.SIGN):
        c = o.geometry["mount"].translation
        half = float(o.geometry["face"].extents[0])
        signs.append((c, half, o.attributes.get("name") or ""))
    for o in m.of_class(MapClass.TRAFFIC_LIGHT):
        c = o.geometry["box"].pose.translation
        h = 2.0 * float(o.geometry["box"].extents[1])
        tls.append((c, h))
    for o in m.of_class(MapClass.LANE_MARKING):
        lanes[o.attributes.get("side", "left")] = np.array(
            [v.mean for v in o.geometry["line"].verts])
    for o in m.of_class(MapClass.ROAD_ARROW):
        arrows.append(np.array([v.mean for v in o.geometry["glyph"].verts]))
    return {"poles": poles, "gposts": gposts, "signs": signs,
            "tls": tls, "lanes": lanes, "arrows": arrows}


def draw(im, g, Rinv, C, fx, fy, cx, cy, los: float) -> None:
    args = (Rinv, C, fx, fy, cx, cy)

    def zc(p):
        return float(((np.asarray(p) - C) @ Rinv.T)[2])

    # lanes (blue/pink)
    for side, pts in g["lanes"].items():
        col = COL[f"lane_{side}"]
        for seg in project_polyline(pts, *args):
            cv2.polylines(im, [seg], False, col, 3, cv2.LINE_AA)
    # arrows (filled yellow)
    ov = im.copy()
    drew = False
    for glyph in g["arrows"]:
        if ZNEAR < zc(glyph.mean(0)) <= los:
            poly = project_pts(glyph, *args)
            if poly is not None:
                cv2.fillPoly(ov, [poly], COL["arrow"])
                drew = True
    if drew:
        cv2.addWeighted(ov, 0.45, im, 0.55, 0, dst=im)
    # poles (cyan stick) + guideposts (white) + lights (green) + signs (magenta)
    for top, bot in g["poles"]:
        if not (ZNEAR < zc(0.5 * (top + bot)) <= los):
            continue
        seg = project_pts(np.array([bot, top]), *args)
        if seg is not None:
            cv2.line(im, tuple(seg[0]), tuple(seg[1]), COL["pole"], 2, cv2.LINE_AA)
    for c, h in g["gposts"]:
        if not (ZNEAR < zc(c) <= los):
            continue
        seg = project_pts(np.array([c - 0.5 * h * UP, c + 0.5 * h * UP]), *args)
        if seg is not None:
            cv2.line(im, tuple(seg[0]), tuple(seg[1]), COL["guide_post"], 3, cv2.LINE_AA)
            cv2.circle(im, tuple(seg[1]), 3, (0, 0, 0), -1, cv2.LINE_AA)
    for c, h in g["tls"]:
        if not (ZNEAR < zc(c) <= los):
            continue
        seg = project_pts(np.array([c - 0.5 * h * UP, c + 0.5 * h * UP]), *args)
        if seg is not None:
            (_, v0), (u1, v1) = seg[0], seg[1]
            wpx = max(3, abs(v0 - v1) // 4)
            cv2.rectangle(im, (u1 - wpx, v1), (u1 + wpx, v0), COL["traffic_light"], 2, cv2.LINE_AA)
    for c, half, name in g["signs"]:
        if not (ZNEAR < zc(c) <= los):
            continue
        ctr = project_pts(c[None], *args)
        if ctr is None:
            continue
        u, v = int(ctr[0, 0]), int(ctr[0, 1])
        zc_m = zc(c)
        rpx = max(4, int(fx * half / zc_m))
        cv2.rectangle(im, (u - rpx, v - rpx), (u + rpx, v + rpx), COL["sign"], 2, cv2.LINE_AA)
        if name:
            cv2.putText(im, name, (u - rpx, v - rpx - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COL["sign"], 1, cv2.LINE_AA)


def banner(im, t, n, has_dyn: bool) -> None:
    ov = im.copy()
    cv2.rectangle(ov, (0, 0), (im.shape[1], 24), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, im, 0.45, 0, dst=im)
    txt = f"map_adas reprojection  frame {t:03d}/{n - 1}"
    if has_dyn:
        txt += "   [red=dynamic orange=pot_dyn]"
    cv2.putText(im, txt, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


def resolve_run(layout: MapLayout, m: MapAdas, run_id: str | None) -> str:
    """The run whose SLAM trajectory shares the map's frame ``M``.

    Default to the run that produced this map (``provenance.run_id``): its
    objects were triangulated in that run's gauge-free ORB world, so any other
    run's trajectory would be rotated relative to it (the ORB gauge freedom) and
    the reprojection would not line up.
    """
    if run_id:
        return run_id
    rid = getattr(m.provenance, "run_id", "")
    if rid and (layout.run(rid).step_out("slam") / "trajectory.json").exists():
        return rid
    raise SystemExit(
        f"map has no usable provenance.run_id ('{rid}'); pass --run-id explicitly")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--run-id", default=None, help="run dir for the SLAM trajectory (default: latest)")
    ap.add_argument("--db", default=None, help="map_adas.json to render (default: map's tracked result)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--los", type=float, default=100.0, help="line-of-sight cull (m)")
    ap.add_argument("--no-dynamic", action="store_true",
                    help="skip the dynamic moving/pot_dyn overlay even if cached")
    ap.add_argument("--dynamic-mode", choices=("pixels", "boxes"), default="pixels",
                    help="pixels=tint every classified mover pixel; "
                         "boxes=draw detector boxes + tint only inside them")
    ap.add_argument("--dyn-alpha", type=float, default=0.5)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    db = Path(args.db) if args.db else layout.db
    suffix = "_boxes" if args.dynamic_mode == "boxes" else ""
    out = Path(args.out) if args.out else layout.output / f"mapgen_overlay{suffix}.mp4"
    seq = layout.input / "odometry"

    m = MapAdas.load(db)
    run_id = resolve_run(layout, m, args.run_id)
    run = layout.run(run_id)
    g = collect(m)
    poses = poses_from_traj(run.step_out("slam") / "trajectory.json")
    calib = read_calib(seq / "calib.txt")
    K = calib["P2"][:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    frames = sorted((seq / "image_2").glob("*.png"))
    n = min(len(frames), len(poses))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    has_dyn = (not args.no_dynamic) and (run.step_out("dynamic") / "manifest.json").exists()
    dyn_boxes = load_dynamic_boxes(run) if (has_dyn and args.dynamic_mode == "boxes") else None
    if has_dyn and args.dynamic_mode == "boxes" and dyn_boxes is None:
        raise SystemExit(f"--dynamic-mode boxes needs dynamic_boxes.json in run {run_id} "
                         "(re-run the dynamic step to produce it)")
    print(f"[overlay] map={db}  run={run_id}  {m!r}")
    print(f"[overlay] {n} frames {w}x{h}  poles={len(g['poles'])} gp={len(g['gposts'])} "
          f"signs={len(g['signs'])} tl={len(g['tls'])} lanes={len(g['lanes'])} arrows={len(g['arrows'])}")
    print(f"[overlay] dynamic layer: "
          f"{f'ON ({args.dynamic_mode}; red=moving orange=pot_dyn)' if has_dyn else 'off'}")

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (w, h))
    for t in tqdm(range(n), desc="overlay"):
        im = cv2.imread(str(frames[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        if has_dyn:
            lab = load_dynamic_label(run, t, h, w)
            if lab is not None and args.dynamic_mode == "pixels":
                paint_dynamic(im, lab, alpha=args.dyn_alpha)
            elif lab is not None and dyn_boxes is not None:
                paint_dynamic_boxes(im, lab, dyn_boxes[t] if t < len(dyn_boxes) else [],
                                    alpha=args.dyn_alpha)
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        draw(im, g, R.T, C, fx, fy, cx, cy, args.los)
        banner(im, t, n, has_dyn)
        writer.write(im)
    writer.release()
    print(f"[overlay] wrote {out}")


if __name__ == "__main__":
    main()
