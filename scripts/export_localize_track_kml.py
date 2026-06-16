"""Export the localization estimate track as KML — dots + pink heading tails.

Reads the artifact from ``scripts/localize.py --out`` and draws the fused
estimate per frame as a dot at the estimated vehicle position with a 1 m PINK
heading tail along the ESTIMATED vehicle forward (``est_fwd_enu``, the f_veh +X
axis from the factor-graph pose). ENU -> WGS84 about the map georef datum.

    uv run python scripts/export_localize_track_kml.py
        [--traj map/kitti-seq04/output/localize_traj.json] [--map kitti-seq04]
        [--out map/kitti-seq04/output/localize_track_heading.kml]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas

_R = 6378137.0


def enu_to_wgs84(enu: np.ndarray, datum) -> np.ndarray:
    lat0, lon0, _ = datum
    enu = np.atleast_2d(np.asarray(enu, float))
    lat = lat0 + np.degrees(enu[:, 1] / _R)
    lon = lon0 + np.degrees(enu[:, 0] / (_R * math.cos(math.radians(lat0))))
    return np.stack([lat, lon], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path,
                    default=Path("map/kitti-seq04/output/localize_traj.json"))
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.traj.exists():
        raise SystemExit(f"{args.traj} not found — run scripts/localize.py --out first")
    data = json.loads(args.traj.read_text())
    if "est_fwd_enu" not in data:
        raise SystemExit(f"{args.traj} has no est_fwd_enu — re-run scripts/localize.py --out")
    datum = MapAdas.load(MapLayout(args.map).db).georef.datum

    est = np.array(data["est_enu"], float)
    fwd = np.array(data["est_fwd_enu"], float)
    tip = enu_to_wgs84(est, datum)               # dot = estimated position
    tail = enu_to_wgs84(est - fwd, datum)        # 1 m behind along estimated heading
    n = len(est)

    out = args.out or (MapLayout(args.map).output / "localize_track_heading.kml")
    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} localization track (heading)</name>',
         '<Style id="tail"><LineStyle><color>ffb469ff</color><width>2</width></LineStyle></Style>',
         '<Style id="dot"><IconStyle><color>ffb469ff</color><scale>0.4</scale>'
         '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>'
         '</Icon></IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>']

    tails = "".join(
        '<LineString><tessellate>1</tessellate><coordinates>'
        f'{tail[i, 1]:.8f},{tail[i, 0]:.8f},0 {tip[i, 1]:.8f},{tip[i, 0]:.8f},0'
        '</coordinates></LineString>' for i in range(n))
    P.append(f'<Placemark><name>estimated heading tails 1m ({n})</name><styleUrl>#tail</styleUrl>'
             f'<MultiGeometry>{tails}</MultiGeometry></Placemark>')
    P.append('<Folder><name>localization estimate (per frame)</name>')
    for i, (lat, lon) in enumerate(tip):
        head = np.degrees(np.arctan2(*(tip[i] - tail[i])[::-1])) % 360.0
        P.append(f'<Placemark><name>{i}</name><description>heading {head:.1f} deg</description>'
                 f'<styleUrl>#dot</styleUrl>'
                 f'<Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point></Placemark>')
    P.append('</Folder></Document></kml>')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  "
          f"localization dots & 1m pink heading tails, {n} frames")


if __name__ == "__main__":
    main()
