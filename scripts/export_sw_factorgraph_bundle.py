"""Dump the SLIDING-WINDOW factor-graph state per display frame for Blender.

Re-runs the sliding-window localizer with recording on (per-step marginal pose
covariances + active factor edges), then builds a per-display-frame view where:

  * each pose is a DOT whose radius = its 1-sigma positional uncertainty;
  * poses inside the current window are "live" (their state at this step);
  * once a pose leaves the window it FREEZES at its last in-window state
    (position + uncertainty), and stays there.

Everything is georeferenced to ENU and recentered by the SAME origin the locviz
scene uses (``/tmp/locviz_data.json``). Output ``/tmp/sw_factorgraph_data.json``:
  - landmarks: map node positions (recentered ENU)
  - frames[t]: {dots:[{p,r,live}], gnss, edges_lm, edges_lane, edges_odom, edges_gnss}

    uv run python scripts/export_sw_factorgraph_bundle.py [--window 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.ingest import load_oxts_fveh_enu
from aivision.map.layout import MapLayout
from gtsam import Point3

from aivision.map.localize_sw import (
    POINT_CLASSES, SWConfig, localize_sliding_window, odometry_from_poses,
)
from aivision.map.localize import (
    LANE_SIDE, map_landmarks, map_lane_curves, _nearest_on_polyline,
)
from aivision.map.map_adas import MapAdas
from aivision.map.objects import MapClass
from aivision.map.virtual_camera import VirtualCameraSensor


def match_current(pose, dets, by_cls, lanes_M, gate=3.0, lane_gate=1.5,
                  z_min=3.0, z_max=45.0):
    """Current-frame measurement matching against the map (point + lane).

    Returns (assoc, unassoc): assoc = (pose_pos, target) edges for measurements that
    associate to a map landmark / lane point; unassoc = (pose_pos, det_world) edges
    for in-range point detections that find NO map match this frame."""
    p0 = np.asarray(pose.translation(), float)
    pred = {}
    for cls, lms in by_cls.items():
        for lm in lms:
            pc = np.asarray(pose.transformTo(Point3(*lm.p_M)), float)
            if z_min < pc[0] < z_max:
                pred.setdefault(cls, []).append((lm, pc))
    assoc, unassoc = [], []
    for obj in dets:
        cls = obj.cls
        if cls == MapClass.LANE_MARKING:
            side = LANE_SIDE.get(obj.attributes.get("side"))
            V = lanes_M.get(side)
            if V is None or len(V) < 2:
                continue
            for vert in obj.geometry["line"].verts:
                vc = np.asarray(vert.mean, float)
                if not (z_min < vc[0] < z_max):
                    continue
                q, _t, d = _nearest_on_polyline(np.asarray(pose.transformFrom(Point3(*vc))), V)
                if q is not None and d <= lane_gate:
                    assoc.append([p0.tolist(), np.asarray(q).tolist()])
            continue
        c = np.asarray(VirtualCameraSensor.center(obj), float)
        infront = z_min < c[0] < z_max
        best, bd = None, gate
        for lm, pc in pred.get(cls, []):
            dd = float(np.linalg.norm(pc - c))
            if dd < bd:
                best, bd = lm, dd
        if best is not None:
            assoc.append([p0.tolist(), best.p_M.tolist()])
        elif cls in POINT_CLASSES and infront:
            unassoc.append([p0.tolist(), np.asarray(pose.transformFrom(Point3(*c))).tolist()])
    return assoc, unassoc

DEFAULT_OXTS = Path("data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--run-id", default="vcam-test")
    ap.add_argument("--oxts", type=Path, default=DEFAULT_OXTS)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--gnss-sigma", type=float, default=5.0)
    ap.add_argument("--pixel-noise", type=float, default=0.0,
                    help="virtual-camera detection noise (px); must match the localize_sw run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--locviz", type=Path, default=Path("/tmp/locviz_data.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/sw_factorgraph_data.json"))
    args = ap.parse_args()

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    # drop the map-display enu_offset so edges/landmarks match the OXTS-ENU localization frame.
    amap.georef.enu_offset = np.zeros(3)
    from aivision.map.virtual_camera import PixelNoise
    noise = PixelNoise(sigma_px=args.pixel_noise, seed=args.seed) if args.pixel_noise > 0 else None
    sensor = VirtualCameraSensor.from_map(args.map, run_id=args.run_id, noise=noise)
    n = sensor.n_frames
    clean, _ = load_oxts_fveh_enu(args.oxts, n)
    rng = np.random.default_rng(args.seed)
    gnss = clean.copy(); gnss[:, :2] += rng.normal(0.0, args.gnss_sigma, size=(n, 2))

    bundle = json.loads((layout.root / "camera.json").read_text())
    P = np.array(bundle["motion"]["poses_vehicle"]).reshape(-1, 3, 4)[:n]
    Tw = np.tile(np.eye(4), (n, 1, 1)); Tw[:, :3, :4] = P
    odom = odometry_from_poses(Tw, 0.05, 0.3, seed=args.seed)

    cfg = SWConfig(window=args.window, gnss_sigma=args.gnss_sigma)
    if args.pixel_noise > 0:                      # match localize_sw: widen bearing sigma + gate
        cfg.sigma_bearing = args.pixel_noise / float(sensor.K[0, 0])
        cfg.gate_m = max(cfg.gate_m, 4.0)
    res = localize_sliding_window(amap, sensor, gnss, odom, cfg, record=True)

    shift = np.array(json.loads(args.locviz.read_text())["origin_enu"], float)

    def enu(p_M):
        return (amap.georef.to_enu(np.atleast_2d(p_M))[0] - shift).tolist()

    def enu_edge(e):
        return [enu(e[0]), enu(e[1])]

    by_t = {s["t"]: s for s in res.states}
    W = args.window
    last_t = max(by_t) if by_t else 0

    # landmark/lane lookups for current-frame measurement matching
    landmarks = map_landmarks(amap, POINT_CLASSES)
    by_cls = {}
    for lm in landmarks:
        by_cls.setdefault(lm.cls, []).append(lm)
    lanes_M = map_lane_curves(amap)

    frames = []
    for t in range(n):
        a = max(0, t - W + 1)
        dots = []
        for i in range(t + 1):
            live = i >= a and t >= 1
            src = t if live else min(i + W - 1, last_t)
            src = max(src, 1)
            s = by_t.get(src)
            if s is None or i not in s["poses"]:
                # fallback: latest recorded state that contains pose i
                cand = [tt for tt, ss in by_t.items() if i in ss["poses"]]
                if not cand:
                    continue
                s = by_t[max(cand)]
            pos, r = s["poses"][i]
            dots.append({"p": enu(pos), "r": float(r), "live": bool(live)})
        # current-frame measurement edges from the causal pose at t
        assoc, unassoc = [], []
        if t >= 1:
            assoc, unassoc = match_current(res.online_poses[t], sensor.detections(t),
                                           by_cls, lanes_M)
        s = by_t.get(t)
        frames.append({
            "dots": dots,
            "gnss": (enu(s["gnss"]) if (s and s["gnss"] is not None) else None),
            "edges_assoc": [enu_edge(e) for e in assoc],      # current measurement matched (dark green)
            "edges_unassoc": [enu_edge(e) for e in unassoc],  # current measurement unmatched (red)
            "edges_odom": [enu_edge(e) for e in s["edges_odom"]] if s else [],
            "edges_gnss": [enu_edge(e) for e in s["edges_gnss"]] if s else [],
            "edges_lm": [enu_edge(e) for e in s["edges_lm"]] if s else [],  # all in-window pose->landmark obs
        })

    lm_nodes = [{"p": enu(lm.p_M), "cls": lm.cls.value}
                for lm in map_landmarks(amap, POINT_CLASSES)]
    out = {"origin_enu": shift.tolist(), "landmarks": lm_nodes, "frames": frames, "n": n}
    args.out.write_text(json.dumps(out))
    rr = [d["r"] for f in frames for d in f["dots"]]
    print(f"wrote {args.out}  landmarks={len(lm_nodes)}  frames={n}  "
          f"dot-radius (m) min/med/max: {min(rr):.3f}/{np.median(rr):.3f}/{max(rr):.3f}")


if __name__ == "__main__":
    main()
