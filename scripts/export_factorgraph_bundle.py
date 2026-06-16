"""Dump the localization factor-graph state per frame for the Blender viz.

Reproduces the default localization run, then re-derives the FINAL factor graph
under the converged poses: which map landmarks each pose takes a bearing/range
factor to, which lane-curve points it snaps to (point-to-curve factors), and the
loose GNSS fix it is pulled toward. Everything is georeferenced to ENU and shifted
by the SAME origin the locviz bundle used (``/tmp/locviz_data.json``), so the graph
overlays the existing scene exactly.

Output ``/tmp/factorgraph_data.json``:
  - landmarks: all point-landmark node positions (recentered ENU) + class
  - frames[t]: {pose, gnss, lm:[...], lane:[...]}  (recentered ENU edges from pose)

    uv run python scripts/export_factorgraph_bundle.py [--gnss-sigma 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.ingest import load_oxts_fveh_enu
from aivision.map.layout import MapLayout
from aivision.map.localize import (
    LocalizeConfig, associate, associate_lanes, localize,
    map_landmarks, map_lane_curves,
)
from aivision.map.map_adas import MapAdas
from aivision.map.virtual_camera import VirtualCameraSensor

DEFAULT_OXTS = Path("data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--run-id", default="vcam-test")
    ap.add_argument("--oxts", type=Path, default=DEFAULT_OXTS)
    ap.add_argument("--gnss-sigma", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--locviz", type=Path, default=Path("/tmp/locviz_data.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/factorgraph_data.json"))
    args = ap.parse_args()

    amap = MapAdas.load(MapLayout(args.map).db)
    sensor = VirtualCameraSensor.from_map(args.map, run_id=args.run_id)
    n = sensor.n_frames
    clean_enu, _ = load_oxts_fveh_enu(args.oxts, n)
    rng = np.random.default_rng(args.seed)
    gnss_enu = clean_enu.copy()
    gnss_enu[:, :2] += rng.normal(0.0, args.gnss_sigma, size=(n, 2))

    cfg = LocalizeConfig(gnss_sigma=args.gnss_sigma)
    res = localize(amap, sensor, gnss_enu, cfg)
    poses = res.poses_M

    # final factor graph under the converged poses
    landmarks = map_landmarks(amap)
    lanes_M = map_lane_curves(amap)
    assoc = associate(poses, sensor, landmarks, gate_m=cfg.gate_m,
                      sigma_bearing=cfg.sigma_bearing)
    lane_assoc = associate_lanes(poses, sensor, lanes_M, gate_m=cfg.lane_gate_m,
                                 sigma_perp=cfg.lane_sigma_perp)

    # same recenter origin as the locviz scene
    shift = np.array(json.loads(args.locviz.read_text())["origin_enu"], float)

    def enu(p_M):
        return (amap.georef.to_enu(np.atleast_2d(p_M))[0] - shift).tolist()

    pose_enu = amap.georef.to_enu(res.centers_M) - shift
    gnss_rc = gnss_enu - shift            # gnss_enu is already ENU -> just recenter

    by_frame_lm: dict[int, list] = {}
    for a in assoc:
        by_frame_lm.setdefault(a.frame, []).append(enu(a.landmark.p_M))
    by_frame_lane: dict[int, list] = {}
    for a in lane_assoc:
        by_frame_lane.setdefault(a.frame, []).append(enu(a.q_M))

    frames = []
    for t in range(n):
        frames.append({
            "pose": pose_enu[t].tolist(),
            "gnss": gnss_rc[t].tolist(),
            "lm": by_frame_lm.get(t, []),
            "lane": by_frame_lane.get(t, []),
        })

    lm_nodes = [{"p": enu(lm.p_M), "cls": lm.cls.value} for lm in landmarks]
    bundle = {"origin_enu": shift.tolist(), "landmarks": lm_nodes, "frames": frames, "n": n}
    args.out.write_text(json.dumps(bundle))
    tot_lm = sum(len(f["lm"]) for f in frames)
    tot_ln = sum(len(f["lane"]) for f in frames)
    print(f"wrote {args.out}  landmarks={len(lm_nodes)}  "
          f"bearing-factor edges={tot_lm}  lane-factor edges={tot_ln}  frames={n}")


if __name__ == "__main__":
    main()
