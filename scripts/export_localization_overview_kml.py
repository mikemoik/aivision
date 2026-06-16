"""Combined seq04 KML overview: static map + ego lanes + GNSS GT + localization track.

Layers (Google-Earth styled):
  * static objects (signs / lights / guideposts / poles / road arrows) -> PINK vertical lines
  * ego-lane boundaries                                                -> BLUE ground lines
  * GNSS ground-truth track (f_veh OXTS)                               -> YELLOW, 1 m vertical tails
  * latest localization estimate (localize_sw_traj.json)               -> GREEN,  1 m vertical tails

Georeferencing matches export_static_scene_kml.py: ORB-world -> ENU via an Umeyama fit of the ORB
camera centres against the OXTS GNSS, plus the measured display offset SEQ04_ENU_OFFSET (so the map
lands on the satellite imagery). The GT + estimate tracks are already in OXTS-ENU (the localizer
runs offset-free); we add the SAME display offset and project with the OXTS datum so every layer is
mutually consistent.

    uv run python scripts/export_localization_overview_kml.py
        [--traj map/kitti-seq04/output/localize_sw_traj.json] [--tail 1.0]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OXD = REPO / "data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data"
ORB = REPO / "outputs/pocs/orb_slam3/CameraTrajectory.txt"
LANE = REPO / "outputs/ego_lanes/lane_map/lane_polylines_model.json"
ARROWS = REPO / "outputs/static_env/arrows_lanemodel/arrows_world.json"
PS = REPO / "outputs/static_env/poles_signs/poles_signs_world.json"
R_E = 6378137.0
ENU_OFFSET = np.array([0.20, 1.37, 0.0])      # display nudge (see reference_seq04_gnss_offset)

# KML colours are aabbggrr
PINK, BLUE, YELLOW, GREEN = "ffb469ff", "ffff0000", "ff00ffff", "ff00ff00"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path,
                    default=REPO / "map/kitti-seq04/output/localize_sw_traj.json")
    ap.add_argument("--tail", type=float, default=1.0, help="track tail height [m]")
    ap.add_argument("--out", type=Path, default=REPO / "map/kitti-seq04/output/localization_overview.kml")
    args = ap.parse_args()

    n = 271
    ox = np.array([np.loadtxt(OXD / f"{i:010d}.txt")[:3] for i in range(n)])   # lat,lon,alt
    lat0, lon0, alt0 = ox[0]
    cl = math.cos(math.radians(lat0))

    def enu_to_lonlat(e, n_, u=0.0):
        eo, no, _ = (e, n_, u) + ENU_OFFSET
        return (lon0 + math.degrees(eo / (R_E * cl)), lat0 + math.degrees(no / R_E))

    # --- ORB-world -> ENU similarity fit (camera centres vs GNSS) ---
    gE = np.radians(ox[:, 1] - lon0) * R_E * cl
    gN = np.radians(ox[:, 0] - lat0) * R_E
    enu = np.stack([gE, gN, ox[:, 2] - alt0], 1)
    orb_c = np.loadtxt(ORB).reshape(-1, 3, 4)[:n, :, 3]

    def umeyama(src, dst):
        ms, md = src.mean(0), dst.mean(0); S, D = src - ms, dst - md
        cov = D.T @ S / len(src); U, sv, Vt = np.linalg.svd(cov); Sg = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[-1, -1] = -1
        Rm = U @ Sg @ Vt; sc = np.trace(np.diag(sv) @ Sg) / ((S ** 2).sum() / len(src))
        return sc, Rm, md - sc * Rm @ ms
    s, Rm, t = umeyama(orb_c, enu)

    def orb_to_lonlat(p_orb):
        e, n_, _ = s * (Rm @ np.asarray(p_orb, float)) + t
        return enu_to_lonlat(e, n_)

    # --- KML helpers ---
    def coords_orb(pts):
        return " ".join("{:.8f},{:.8f},0".format(*orb_to_lonlat(p)) for p in pts)

    def line(name, pts, style):
        return (f'<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl><LineString>'
                f'<tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>'
                f'<coordinates>{coords_orb(pts)}</coordinates></LineString></Placemark>')

    def vbar_orb(name, center, h, style):
        lo, la = orb_to_lonlat(center)
        return (f'<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl><LineString>'
                f'<extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>'
                f'<coordinates>{lo:.8f},{la:.8f},0 {lo:.8f},{la:.8f},{h:.2f}</coordinates></LineString></Placemark>')

    def tail(lo, la, h, style):
        return (f'<Placemark><styleUrl>#{style}</styleUrl><LineString><extrude>1</extrude>'
                f'<altitudeMode>relativeToGround</altitudeMode>'
                f'<coordinates>{lo:.8f},{la:.8f},0 {lo:.8f},{la:.8f},{h:.2f}</coordinates></LineString></Placemark>')

    def track(name, enu_pts, style):
        ll = [enu_to_lonlat(p[0], p[1]) for p in enu_pts]
        body = [f'<Folder><name>{name}</name>']
        body.append('<Placemark><styleUrl>#{}</styleUrl><LineString><tessellate>1</tessellate>'
                     '<altitudeMode>clampToGround</altitudeMode><coordinates>{}</coordinates>'
                     '</LineString></Placemark>'.format(
                         style, " ".join(f"{lo:.8f},{la:.8f},0" for lo, la in ll)))
        for lo, la in ll:
            body.append(tail(lo, la, args.tail, style))
        body.append('</Folder>')
        return "".join(body)

    K = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         '<name>seq04 localization overview</name>']
    for sid, col, w in [('pink', PINK, 3), ('blue', BLUE, 2),
                        ('yellow', YELLOW, 2), ('green', GREEN, 2)]:
        K.append(f'<Style id="{sid}"><LineStyle><color>{col}</color><width>{w}</width></LineStyle>'
                 f'<PolyStyle><color>60{col[2:]}</color></PolyStyle></Style>')

    # --- ego lanes (blue) ---
    lm = json.loads(LANE.read_text())
    K.append('<Folder><name>Ego lane (blue)</name>')
    K.append(line('left border', np.array(lm['left'], float), 'blue'))
    K.append(line('right border', np.array(lm['right'], float), 'blue'))
    K.append('</Folder>')

    # --- static objects (pink): arrows + signs + lights + guideposts + poles ---
    ar = json.loads(ARROWS.read_text())
    ps = json.loads(PS.read_text())
    K.append('<Folder><name>Static objects (pink)</name>')
    for i, a in enumerate(ar):
        g = np.array(a['glyph_world'], float); g = np.vstack([g, g[0]])
        K.append(line(a.get('cls', f'arrow {i}'), g, 'pink'))
    for sgn in ps.get('signs', []):
        K.append(vbar_orb('sign', sgn['center'], 2.2, 'pink'))
    for tl in ps.get('traffic_lights', []):
        K.append(vbar_orb('light', tl['center'], 5.0, 'pink'))
    for gp in ps.get('guideposts', []):
        K.append(vbar_orb('guidepost', gp['center'], max(gp.get('height', 1.0), 0.8), 'pink'))
    for p in ps.get('poles', []):
        K.append(vbar_orb('pole', p['center'], min(max(p.get('height', 6.0), 2.0), 12.0), 'pink'))
    K.append('</Folder>')

    # --- GNSS GT (yellow tails) + localization estimate (green tails) ---
    d = json.loads(args.traj.read_text())
    clean = np.array(d['clean_enu'], float)
    est = np.array(d['est_enu'], float)
    K.append(track('GNSS GT (yellow)', clean, 'yellow'))
    K.append(track('Localization track (green)', est, 'green'))

    K.append('</Document></kml>')
    args.out.write_text("".join(K))
    print(f"wrote {args.out}  ({args.out.stat().st_size/1024:.0f} KB): "
          f"{len(ar)} arrows + {len(ps.get('signs',[]))} signs + {len(ps.get('traffic_lights',[]))} lights"
          f" + {len(ps.get('guideposts',[]))} guideposts + {len(ps.get('poles',[]))} poles (pink), "
          f"lanes (blue), GNSS GT (yellow {args.tail:.0f} m tails), est (green {args.tail:.0f} m tails)")


if __name__ == "__main__":
    main()
