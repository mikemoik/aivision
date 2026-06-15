"""Fit a coupled lane model to the ego-lane point cloud (constant-ish width).

Thin CLI over the shared builder ``aivision.map.steps_lanes.fit_lane_model_from_pts``
(the map-gen T4 lanes step's core): a shared centerline c(z) plus a half-width hw(z)
with borders left = c - hw, right = c + hw, fit under RANSAC border rejection +
Whittaker (penalised-2nd-difference) smoothing. This script just splits the
``lane_map.ply`` cloud by colour (ego-left blue / ego-right pink) and feeds the
per-class points to the builder.

Frame: first-camera / world frame, metric metres (same as lane_polylines.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.steps_lanes import fit_lane_model_from_pts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=Path, default=Path("outputs/ego_lanes/lane_map/lane_map.ply"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_polylines_model.json"))
    ap.add_argument("--dz", type=float, default=2.0, help="Z-bin size (m)")
    ap.add_argument("--minpts", type=int, default=12, help="min points per bin")
    ap.add_argument("--ransac-deg", type=int, default=3)
    ap.add_argument("--ransac-thresh", type=float, default=0.35,
                    help="lateral inlier band for RANSAC border fit (m)")
    ap.add_argument("--lam-center", type=float, default=20.0,
                    help="centerline rigidity (lower = follows road bends)")
    ap.add_argument("--lam-width", type=float, default=400.0,
                    help="WIDTH rigidity: big = near-constant lane width (start rigid)")
    ap.add_argument("--lam-height", type=float, default=80.0)
    args = ap.parse_args()

    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(args.ply))
    P = np.asarray(pc.points)
    C = np.asarray(pc.colors)
    pts = {0: P[C[:, 2] > C[:, 0]],   # blue = ego-left
           1: P[C[:, 0] > C[:, 2]]}   # pink = ego-right

    out = fit_lane_model_from_pts(
        pts, dz=args.dz, minpts=args.minpts, ransac_deg=args.ransac_deg,
        ransac_thresh=args.ransac_thresh, lam_center=args.lam_center,
        lam_width=args.lam_width, lam_height=args.lam_height)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"[lanemodel] wrote {args.out}  ({len(out['left'])} verts/side)")


if __name__ == "__main__":
    main()
