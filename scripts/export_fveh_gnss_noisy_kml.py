"""Synthesize a noisy GNSS track from the f_veh track and export it as KML.

Takes the pipeline's reconstructed vehicle-centreline trajectory (camera bundle
``motion.poses_vehicle`` in f_veh_t0 — the near-GT track) and adds seeded
Gaussian noise (default sigma 1 m) to the HORIZONTAL position of each frame, the
same way ``scripts/localize.py`` perturbs the clean OXTS into a loose GNSS fix.
Drawn as RED dots joined by a thin red line, georeferenced f_veh_t0 -> WGS84 via
the map's own georef.

    uv run python scripts/export_fveh_gnss_noisy_kml.py [--map kitti-seq04]
        [--sigma 1.0] [--seed 0]
        [--out map/kitti-seq04/output/fveh_gnss_noisy.kml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--sigma", type=float, default=1.0, help="horizontal GNSS noise [m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    if amap.georef is None:
        raise SystemExit("map has no georef — cannot export lat/lon")

    # f_veh poses (f_veh_t0): position = column 3; x-fwd / y-left / z-up
    bundle = json.loads((layout.root / "camera.json").read_text())
    P_veh = np.array(bundle["motion"]["poses_vehicle"]).reshape(-1, 3, 4)
    centre = P_veh[:, :, 3].copy()
    n = len(centre)

    # noisy GNSS = clean horizontal position + seeded Gaussian noise (x, y only;
    # f_veh_t0 x-y is the horizontal plane since z is up). Height left untouched.
    rng = np.random.default_rng(args.seed)
    centre[:, :2] += rng.normal(0.0, args.sigma, size=(n, 2))

    llh = amap.georef.to_wgs84(centre)[:, :2]   # (lat, lon)

    out = args.out or (layout.output / "fveh_gnss_noisy.kml")
    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} noisy GNSS from f_veh (sigma {args.sigma:g} m)</name>',
         '<Style id="line"><LineStyle><color>660000ff</color><width>2</width></LineStyle></Style>',
         '<Style id="dot"><IconStyle><color>ff0000ff</color><scale>0.4</scale>'
         '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>'
         '</Icon></IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>']

    line = " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in llh)
    P.append('<Placemark><name>noisy GNSS (line)</name><styleUrl>#line</styleUrl>'
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
          f"noisy GNSS from f_veh (sigma {args.sigma:g} m, seed {args.seed}), {n} frames")


if __name__ == "__main__":
    main()
