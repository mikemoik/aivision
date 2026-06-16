"""Export the f_veh vehicle track as a KML for Google Earth.

The pipeline's reconstructed vehicle-centreline trajectory (camera bundle
``motion.poses_vehicle`` in f_veh_t0), drawn per-frame as a GREEN dot with a 1 m
YELLOW heading tail. The heading is the vehicle forward axis (f_veh +X = column 0
of each pose) — the tail trails 1 m behind, tip at the dot — georeferenced
f_veh_t0 -> WGS84 via the map's own georef.

    uv run python scripts/export_gnss_fveh_kml.py [--map kitti-seq04]
        [--out map/kitti-seq04/output/fveh_track.kml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    if amap.georef is None:
        raise SystemExit("map has no georef — cannot export lat/lon")

    # f_veh poses (f_veh_t0) from the bundle: centre + forward (col 0 = +X)
    bundle = json.loads((layout.root / "camera.json").read_text())
    P_veh = np.array(bundle["motion"]["poses_vehicle"]).reshape(-1, 3, 4)
    n = len(P_veh)
    centre = P_veh[:, :, 3]                      # f_veh position in f_veh_t0
    fwd = P_veh[:, :, 0]                         # f_veh forward (+X) in f_veh_t0
    tip = amap.georef.to_wgs84(centre)[:, :2]              # dot = vehicle position
    tail = amap.georef.to_wgs84(centre - fwd)[:, :2]       # 1 m behind along heading

    out = args.out or (layout.output / "fveh_track.kml")
    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} f_veh track</name>',
         '<Style id="tail"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle></Style>',
         '<Style id="dot"><IconStyle><color>ff00ff00</color><scale>0.4</scale>'
         '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>'
         '</Icon></IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>']

    # f_veh: one MultiGeometry of 1 m heading tails (base -> tip), then dots at tips
    tails = "".join(
        '<LineString><tessellate>1</tessellate><coordinates>'
        f'{tail[i, 1]:.8f},{tail[i, 0]:.8f},0 {tip[i, 1]:.8f},{tip[i, 0]:.8f},0'
        '</coordinates></LineString>' for i in range(n))
    P.append(f'<Placemark><name>f_veh heading tails 1m ({n})</name><styleUrl>#tail</styleUrl>'
             f'<MultiGeometry>{tails}</MultiGeometry></Placemark>')
    P.append('<Folder><name>f_veh track (vehicle centreline)</name>')
    for i, (lat, lon) in enumerate(tip):
        head = np.degrees(np.arctan2(*(tip[i] - tail[i])[::-1])) % 360.0  # lat,lon delta
        P.append(f'<Placemark><name>{i}</name><description>heading {head:.1f} deg</description>'
                 f'<styleUrl>#dot</styleUrl>'
                 f'<Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point></Placemark>')
    P.append('</Folder></Document></kml>')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  "
          f"f_veh dots & 1m heading tails, {n} frames")


if __name__ == "__main__":
    main()
