"""Export the ADAS map's road arrows as KML for Google Earth.

Reads the road-arrow glyph footprints straight from ``map_adas.json`` (frame M)
and georeferences each vertex with the map's own ``GeoReference`` (M -> WGS84).
Each arrow becomes one closed, tessellated LineString clamped to the ground,
drawn red.

    uv run python scripts/export_arrows_kml.py [--map kitti-seq04]
        [--out map/kitti-seq04/output/arrows.kml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas
from aivision.map.objects import MapClass


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

    arrows = amap.of_class(MapClass.ROAD_ARROW)
    if not arrows:
        raise SystemExit("map has no road arrows")

    out = args.out or (layout.output / "arrows.kml")

    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} road arrows</name>',
         '<Style id="arrow"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle>'
         '<PolyStyle><color>400000ff</color></PolyStyle></Style>']

    for o in arrows:
        verts_M = np.array([v.mean for v in o.geometry["glyph"].verts])
        verts_M = np.vstack([verts_M, verts_M[0]])          # close the glyph
        llh = amap.georef.to_wgs84(verts_M)
        cls = o.attributes.get("cls", "arrow")
        P.append(f'<Placemark><name>{o.id} ({cls})</name><styleUrl>#arrow</styleUrl>'
                 '<LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>'
                 f'<coordinates>{_coords(llh)}</coordinates></LineString></Placemark>')
        print(f"  {o.id:16s} {cls:16s} {len(verts_M) - 1} verts")

    P.append('</Document></kml>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  {len(arrows)} road arrows")


if __name__ == "__main__":
    main()
