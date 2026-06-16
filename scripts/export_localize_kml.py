"""Export the factor-graph localization track as KML for Google Earth.

Reads the artifact written by ``scripts/localize.py --out`` and draws the fused
estimate (the camera trajectory localized in the ADAS map) as a single
tessellated GREEN line, georeferenced through the map datum.

ENU -> WGS84 is the spherical equirectangular about the map georef datum (the
exact inverse of the OXTS->ENU forward projection), matching ``GeoReference``.

    uv run python scripts/export_localize_kml.py
        [--traj map/kitti-seq04/output/localize_traj.json] [--map kitti-seq04]
        [--out map/kitti-seq04/output/localize_track.kml]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas

_R = 6378137.0  # spherical Earth radius [m] (matches GeoReference / OXTS forward)


def enu_to_wgs84(enu: np.ndarray, datum) -> np.ndarray:
    """ENU metres about ``datum`` (lat0, lon0, alt0) -> (lat, lon, alt) degrees/m."""
    lat0, lon0, alt0 = datum
    enu = np.atleast_2d(np.asarray(enu, float))
    lat = lat0 + np.degrees(enu[:, 1] / _R)
    lon = lon0 + np.degrees(enu[:, 0] / (_R * math.cos(math.radians(lat0))))
    alt = alt0 + enu[:, 2]
    return np.stack([lat, lon, alt], axis=1)


def _coords(llh: np.ndarray) -> str:
    return " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon, _ in llh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path,
                    default=Path("map/kitti-seq04/output/localize_traj.json"))
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.traj.exists():
        raise SystemExit(
            f"{args.traj} not found — run:\n"
            f"  uv run python scripts/localize.py --out {args.traj}")
    data = json.loads(args.traj.read_text())
    datum = MapAdas.load(MapLayout(args.map).db).georef.datum

    est = enu_to_wgs84(np.array(data["est_enu"]), datum)
    rep = data.get("report", {})

    out = args.out or args.traj.with_suffix("").with_name(
        args.traj.stem.replace("_traj", "") + "_track.kml")

    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} localization track</name>',
         f'<description>fused estimate (camera localized in the ADAS map); '
         f'ATE2D {rep.get("ate_2d", float("nan")):.2f} m, '
         f'RPE {rep.get("rpe_t", float("nan")):.2f} m, '
         f'heading {rep.get("heading_deg_med", float("nan")):.2f} deg</description>',
         '<Style id="est"><LineStyle><color>ff00ff00</color><width>4</width></LineStyle></Style>']

    P.append('<Placemark><name>localization track</name><styleUrl>#est</styleUrl>'
             '<LineString><tessellate>1</tessellate>'
             f'<coordinates>{_coords(est)}</coordinates></LineString></Placemark>')
    P.append('</Document></kml>')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  "
          f"localization track, {len(est)} vertices")


if __name__ == "__main__":
    main()
