"""Export the NOISY GNSS track that the localizer actually consumed, as KML.

Reads the artifact written by ``scripts/localize.py --out`` and draws the
per-frame noisy GNSS fix (``gnss_enu`` = clean OXTS + seeded N(0, sigma^2),
the loose 2D input sensor) as RED dots joined by a thin red line, so it can be
overlaid against the f_veh track and the localization estimate to *see* how far
the factor graph pulled the input back onto the trajectory.

ENU -> WGS84 about the map georef datum (matches the other localize exporters).

    uv run python scripts/export_localize_gnss_kml.py
        [--traj map/kitti-seq04/output/localize_traj.json] [--map kitti-seq04]
        [--out map/kitti-seq04/output/localize_gnss.kml]
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
    if "gnss_enu" not in data:
        raise SystemExit(f"{args.traj} has no gnss_enu — re-run scripts/localize.py --out")
    datum = MapAdas.load(MapLayout(args.map).db).georef.datum

    gnss = np.array(data["gnss_enu"], float)
    llh = enu_to_wgs84(gnss, datum)
    n = len(gnss)
    sigma = data.get("gnss_sigma", "?")

    out = args.out or (MapLayout(args.map).output / "localize_gnss.kml")
    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} noisy GNSS input (sigma {sigma} m)</name>',
         '<Style id="line"><LineStyle><color>660000ff</color><width>2</width></LineStyle></Style>',
         '<Style id="dot"><IconStyle><color>ff0000ff</color><scale>0.4</scale>'
         '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>'
         '</Icon></IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>']

    line = " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in llh)
    P.append('<Placemark><name>noisy GNSS input (line)</name><styleUrl>#line</styleUrl>'
             '<LineString><tessellate>1</tessellate>'
             f'<coordinates>{line}</coordinates></LineString></Placemark>')
    P.append('<Folder><name>noisy GNSS fixes (per frame)</name>')
    for i, (lat, lon) in enumerate(llh):
        P.append(f'<Placemark><name>{i}</name><styleUrl>#dot</styleUrl>'
                 f'<Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point></Placemark>')
    P.append('</Folder></Document></kml>')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  "
          f"noisy GNSS input (red dots + line), {n} frames")


if __name__ == "__main__":
    main()
