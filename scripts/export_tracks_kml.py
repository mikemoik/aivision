"""Export all trajectory tracks into ONE KML for side-by-side comparison.

Draws each track as a distinctly-coloured tessellated line in a single Google
Earth document, all georeferenced through the map's own ``GeoReference``:

  * GREEN   — fused localization estimate (from scripts/localize.py --out)
  * ORANGE  — ORB-SLAM3 cam0 (left-camera) track
  * MAGENTA — ORB-SLAM3 vehicle (f_veh) track

Optionally add the clean OXTS GPS ground truth with --with-oxts (BLUE).

    uv run python scripts/export_tracks_kml.py [--map kitti-seq04]
        [--traj map/kitti-seq04/output/localize_traj.json] [--with-oxts]
        [--out map/kitti-seq04/output/tracks_compare.kml]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aivision.map.ingest import load_oxts_fveh_enu
from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas

DEFAULT_OXTS = Path("data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data")
_R = 6378137.0


def _coords_from_wgs84(llh: np.ndarray) -> str:
    return " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon, _ in llh)


def _enu_to_wgs84(enu: np.ndarray, datum) -> np.ndarray:
    lat0, lon0, alt0 = datum
    enu = np.atleast_2d(np.asarray(enu, float))
    lat = lat0 + np.degrees(enu[:, 1] / _R)
    lon = lon0 + np.degrees(enu[:, 0] / (_R * math.cos(math.radians(lat0))))
    return np.stack([lat, lon, alt0 + enu[:, 2]], axis=1)


def _placemark(name: str, style: str, llh: np.ndarray) -> str:
    return (f'<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl>'
            '<LineString><tessellate>1</tessellate>'
            f'<coordinates>{_coords_from_wgs84(llh)}</coordinates></LineString></Placemark>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--traj", type=Path,
                    default=Path("map/kitti-seq04/output/localize_traj.json"))
    ap.add_argument("--camera", type=Path, default=None)
    ap.add_argument("--with-oxts", action="store_true",
                    help="also draw the clean OXTS GPS ground truth (blue)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    if amap.georef is None:
        raise SystemExit("map has no georef — cannot export lat/lon")
    gr = amap.georef

    tracks: list[tuple[str, str, np.ndarray]] = []  # (name, style, wgs84)

    # ORB cam0 + vehicle from the camera bundle
    bundle = json.loads((args.camera or (layout.root / "camera.json")).read_text())
    poses = np.array(bundle["motion"]["poses_twc"]).reshape(-1, 3, 4)
    cam0_M = poses[:, :3, 3]
    tracks.append(("ORB-SLAM3 — cam0 (left camera)", "cam0", gr.to_wgs84(cam0_M)))
    # vehicle (f_veh) poses are now carried directly by the bundle in f_veh_t0
    veh_rows = bundle["motion"].get("poses_vehicle")
    if veh_rows is not None:
        veh_M = np.array(veh_rows)[:, [3, 7, 11]]
        tracks.append(("ORB-SLAM3 — vehicle (f_veh)", "veh", gr.to_wgs84(veh_M)))

    # fused localization estimate
    if args.traj.exists():
        est_enu = np.array(json.loads(args.traj.read_text())["est_enu"])
        tracks.insert(0, ("localization (fused estimate)", "loc",
                          _enu_to_wgs84(est_enu, gr.datum)))
    else:
        print(f"note: {args.traj} missing — run scripts/localize.py --out first")

    if args.with_oxts and DEFAULT_OXTS.exists():
        oxts_enu, datum = load_oxts_fveh_enu(DEFAULT_OXTS, len(cam0_M))
        tracks.append(("OXTS GPS (ground truth)", "oxts", _enu_to_wgs84(oxts_enu, datum)))

    styles = {"loc": "ff00ff00", "cam0": "ff0080ff", "veh": "ffff00ff",
              "oxts": "ffff6400"}  # aabbggrr: green / orange / magenta / blue
    P = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{args.map} track comparison</name>']
    for sid, color in styles.items():
        P.append(f'<Style id="{sid}"><LineStyle><color>{color}</color>'
                 '<width>4</width></LineStyle></Style>')
    for name, sid, llh in tracks:
        P.append(_placemark(name, sid, llh))
    P.append('</Document></kml>')

    out = args.out or (layout.output / "tracks_compare.kml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)  {len(tracks)} tracks:")
    for name, _, llh in tracks:
        print(f"  - {name}  ({len(llh)} pts)")


if __name__ == "__main__":
    main()
