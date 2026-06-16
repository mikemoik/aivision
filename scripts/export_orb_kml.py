"""Export the ORB-SLAM3 track as KML for Google Earth.

The ORB-SLAM3 stereo trajectory *defines* the map frame M (frame 0 = identity).
Its per-frame poses live in the camera bundle (``camera.json`` ->
``motion.poses_twc``, flattened 3x4 ``T_world_cam0`` rows). Points are
georeferenced with the map's own ``GeoReference`` (M -> WGS84) and drawn as a
single tessellated line.

``--ref`` selects which physical point the track follows:

  * ``cam0``    — the reference camera (image_0, left gray) origin (default).
    On this rig cam0 sits on the car's lateral centreline (KITTI Fig. 3).
  * ``vehicle`` — the OXTS IMU/GPS origin. Offset from cam0 by the cam0->IMU
    lever arm (``calib_imu_to_velo`` o ``velo_to_cam0``); matches the raw OXTS
    GPS track to ~0.3 m. NB the IMU is mounted 0.32 m LEFT of centre.
  * ``fveh``    — the canonical vehicle frame ``f_veh``: the IMU frame shifted
    ``(0, -0.32, 0)`` m in IMU axes (x-fwd, y-left, z-up), i.e. moved 0.32 m
    right onto the lateral centreline. Keeps the IMU fore/aft + height; x-fwd /
    y-left / z-up. Lands within ~5 mm of the cam0/velodyne centreline.

    uv run python scripts/export_orb_kml.py [--map kitti-seq04] [--ref cam0|vehicle|fveh]
        [--out map/kitti-seq04/output/orb_track.kml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas


def _coords(llh: np.ndarray) -> str:
    return " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon, _ in llh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--ref", choices=("cam0", "vehicle", "fveh"), default="vehicle",
                    help="track the f_veh vehicle body (default) or cam0; vehicle==fveh now")
    ap.add_argument("--camera", type=Path, default=None,
                    help="camera bundle (default map/<map>/camera.json)")
    ap.add_argument("--color", default="green",
                    help="line colour: a name (green/red/blue/yellow/orange) or raw KML aabbggrr")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    NAMED = {"green": "ff00ff00", "red": "ff0000ff", "blue": "ffff0000",
             "yellow": "ff00ffff", "orange": "ff0080ff"}
    color = NAMED.get(args.color, args.color)

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    if amap.georef is None:
        raise SystemExit("map has no georef — cannot export lat/lon")

    # The bundle now carries both pose sets directly in f_veh_t0: poses_vehicle
    # (f_veh body) and poses_twc (cam0). No lever-arm reconstruction needed.
    bundle = json.loads((args.camera or (layout.root / "camera.json")).read_text())
    motion = bundle["motion"]
    if args.ref == "cam0":
        rows, label = motion["poses_twc"], "cam0 (left camera)"
    else:
        rows, label = motion.get("poses_vehicle", motion["poses_twc"]), "f_veh (vehicle)"
    pts_world = np.array(rows)[:, [3, 7, 11]]            # body centres in f_veh_t0
    llh = amap.georef.to_wgs84(pts_world)
    out = args.out or (layout.output / f"orb_track_{args.ref}.kml")
    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} ORB-SLAM3 track ({label})</name>',
         f'<Style id="orb"><LineStyle><color>{color}</color><width>4</width></LineStyle></Style>',
         f'<Placemark><name>ORB-SLAM3 track — {label}</name><styleUrl>#orb</styleUrl>'
         '<LineString><tessellate>1</tessellate>'
         f'<coordinates>{_coords(llh)}</coordinates></LineString></Placemark>',
         '</Document></kml>']

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  "
          f"ORB-SLAM3 track [{label}], {len(pts_world)} vertices")


if __name__ == "__main__":
    main()
