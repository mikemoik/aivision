"""Dump a compact ENU (Z-up) data bundle for the Blender localization viz.

Pulls the ego-lane borders from ``map_adas.json`` and a localization trajectory
(``scripts/localize.py --out`` artifact) and georeferences everything into ENU
(x=East, y=North, z=Up). Recenters by the first localization fix (E0, N0) and by
the median lane height so the road sits near z=0. Writes ``/tmp/locviz_data.json``
for the Blender MCP scene builder to read.

    uv run python scripts/export_locviz_bundle.py
        [--map kitti-seq04] [--traj map/kitti-seq04/output/localize_traj.json]
        [--out /tmp/locviz_data.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas
from aivision.map.objects import MapClass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--traj", type=Path,
                    default=Path("map/kitti-seq04/output/localize_traj.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/locviz_data.json"))
    args = ap.parse_args()

    amap = MapAdas.load(MapLayout(args.map).db)
    if amap.georef is None:
        raise SystemExit("map has no georef")
    # drop the map-display enu_offset so this bundle matches the OXTS-ENU localization frame
    # (the localizers score offset-free; keeping it here would shift the viz ~1.37 m N).
    amap.georef.enu_offset = np.zeros(3)
    data = json.loads(args.traj.read_text())
    est = np.array(data["est_enu"], float)            # (n,3) ENU
    fwd = np.array(data.get("est_fwd_enu", []), float)  # (n,3) heading unit, ENU

    # ego-lane borders -> ENU
    lanes = {}
    for o in amap.of_class(MapClass.LANE_MARKING):
        side = o.attributes.get("side", "left")
        verts = np.array([v.mean for v in o.geometry["line"].verts], float)
        lanes[side] = amap.georef.to_enu(verts)

    # recenter: origin = first localization fix (E,N); ground = median lane Up
    e0, n0 = float(est[0, 0]), float(est[0, 1])
    all_lane_u = np.concatenate([v[:, 2] for v in lanes.values()]) if lanes else np.array([0.0])
    u0 = float(np.median(all_lane_u))
    shift = np.array([e0, n0, u0])

    def rc(p):
        return (np.asarray(p, float) - shift).tolist()

    bundle = {
        "map": args.map,
        "traj": str(args.traj),
        "origin_enu": [e0, n0, u0],
        "lane_left": rc(lanes["left"]) if "left" in lanes else [],
        "lane_right": rc(lanes["right"]) if "right" in lanes else [],
        "track": rc(est),                              # loc positions, recentered ENU
        "fwd": fwd.tolist(),                           # heading unit vectors (ENU, direction)
        "n": int(len(est)),
    }
    args.out.write_text(json.dumps(bundle))
    print(f"wrote {args.out}  n={bundle['n']}  "
          f"lanes L/R={len(bundle['lane_left'])}/{len(bundle['lane_right'])}  "
          f"origin ENU=({e0:.1f},{n0:.1f},{u0:.2f})")
    # quick extents for camera sizing
    t = np.array(bundle["track"])
    print(f"track extent E[{t[:,0].min():.1f},{t[:,0].max():.1f}] "
          f"N[{t[:,1].min():.1f},{t[:,1].max():.1f}] U[{t[:,2].min():.2f},{t[:,2].max():.2f}]")


if __name__ == "__main__":
    main()
