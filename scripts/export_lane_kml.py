"""Export the ADAS map's ego-lane boundaries as KML for Google Earth.

Reads the lane-marking polylines straight from ``map_adas.json`` (frame M) and
georeferences each vertex with the map's own ``GeoReference`` (M -> WGS84). Each
lane boundary becomes one tessellated LineString; left is drawn red, right blue.

    uv run python scripts/export_lane_kml.py [--map kitti-seq04]
        [--out map/kitti-seq04/output/lane_boundaries.kml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas
from aivision.map.objects import MapClass

# lane side -> (KML aabbggrr colour, label)
SIDE_STYLE = {"left": ("ff0000ff", "red"), "right": ("ffff0000", "blue")}


def _coords(llh: np.ndarray) -> str:
    return " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon, _ in llh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    if amap.georef is None:
        raise SystemExit("map has no georef — cannot export lat/lon")

    lanes = amap.of_class(MapClass.LANE_MARKING)
    if not lanes:
        raise SystemExit("map has no lane markings")

    out = args.out or (layout.output / "lane_boundaries.kml")

    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} lane boundaries</name>']
    for side, (color, _) in SIDE_STYLE.items():
        P.append(f'<Style id="{side}"><LineStyle><color>{color}</color>'
                 '<width>4</width></LineStyle></Style>')

    for o in lanes:
        side = o.attributes.get("side", "left")
        style = side if side in SIDE_STYLE else "left"
        verts_M = np.array([v.mean for v in o.geometry["line"].verts])
        llh = amap.georef.to_wgs84(verts_M)
        P.append(f'<Placemark><name>{o.id} ({side})</name><styleUrl>#{style}</styleUrl>'
                 '<LineString><tessellate>1</tessellate>'
                 f'<coordinates>{_coords(llh)}</coordinates></LineString></Placemark>')
        print(f"  {o.id:16s} side={side:5s} {len(verts_M)} verts")

    P.append('</Document></kml>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  {len(lanes)} lane boundaries")


if __name__ == "__main__":
    main()
