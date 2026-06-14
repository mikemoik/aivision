"""Geo-reference the whole static-environment map and export it to KML.

The static maps (lane borders, road arrows, signs, traffic lights, guideposts, poles) live in the
ORB-SLAM3 world frame. We georeference them via a similarity transform fit between the ORB-SLAM3
camera centres and the GNSS (OXTS) positions for the same frames (Umeyama: scale+R+t), then
ORB-world -> ENU -> lat/lon/alt. Footprints (lane, arrows) are clamped to ground; vertical objects
(signs/lights/guideposts/poles) are drawn as extruded vertical lines at their geo position.

seq04 == KITTI raw drive 2011_09_30_drive_0016 (Karlsruhe). OXTS frame i == odometry/ORB frame i.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--objects-only", action="store_true",
                help="export only traffic signs, traffic lights, road arrows and guideposts "
                     "(drop lane borders, poles and the GNSS reference track)")
ap.add_argument("--arrows-only", action="store_true",
                help="export only the road arrows, drawn in red")
ap.add_argument("--lane-only", action="store_true",
                help="export only the ego-lane boundaries, drawn in blue")
ap.add_argument("--pink", action="store_true",
                help="colour all object types (arrows, signs, lights, guideposts) pink")
ap.add_argument("--de", type=float, default=0.0, help="extra East offset [m] added to the ENU compensation")
ap.add_argument("--dn", type=float, default=0.0, help="extra North offset [m] added to the ENU compensation")
ap.add_argument("--du", type=float, default=0.0, help="extra Up offset [m] added to the ENU compensation")
args = ap.parse_args()

REPO = Path(__file__).resolve().parent.parent
OXD = REPO / "data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data"
ORB = REPO / "outputs/pocs/orb_slam3/CameraTrajectory.txt"
LANE = REPO / "outputs/ego_lanes/lane_map/lane_polylines_model.json"
ARROWS = REPO / "outputs/static_env/arrows_lanemodel/arrows_world.json"
PS = REPO / "outputs/static_env/poles_signs/poles_signs_world.json"
OUT = REPO / ("outputs/static_scene_geo_lane.kml" if args.lane_only
              else "outputs/static_scene_geo_arrows.kml" if args.arrows_only
              else ("outputs/static_scene_geo_objects_pink.kml" if args.pink
                    else "outputs/static_scene_geo_objects.kml") if args.objects_only
              else "outputs/static_scene_geo.kml")
R_E = 6378137.0

# --- fit ORB-world -> ENU from camera centres vs GNSS ---
N = 271
ox = np.array([np.loadtxt(OXD / f"{i:010d}.txt")[:3] for i in range(N)])      # lat,lon,alt
lat0, lon0, alt0 = ox[0]
gE = np.radians(ox[:, 1] - lon0) * R_E * math.cos(math.radians(lat0))
gN = np.radians(ox[:, 0] - lat0) * R_E
enu = np.stack([gE, gN, ox[:, 2] - alt0], 1)                                  # GNSS ENU
orb_c = np.loadtxt(ORB).reshape(-1, 3, 4)[:N, :, 3]                           # ORB camera centres

def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); S, D = src - ms, dst - md
    cov = D.T @ S / len(src); U, sv, Vt = np.linalg.svd(cov); Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[-1, -1] = -1
    Rm = U @ Sg @ Vt; sc = np.trace(np.diag(sv) @ Sg) / ((S ** 2).sum() / len(src))
    return sc, Rm, md - sc * Rm @ ms
s, Rm, t = umeyama(orb_c, enu)
resid = np.linalg.norm((s * (Rm @ orb_c.T).T + t) - enu, axis=1)
print(f"[georef] ORB->ENU sim3 fit: scale {s:.4f}, residual mean {resid.mean():.2f} m max {resid.max():.2f} m")

# Measured GNSS along-track bias (truth - ours), ENU metres: from comparing our yellow road-arrow
# tip to the real white painted arrow in satellite imagery near OXTS frame 235 (2026-06-14). The
# raw measurement (dE 0.36, dN 1.96) is fixated at 0.7x then nudged dE -0.05 = (0.20, 1.37, 0.0).
# Added to every georeferenced coordinate so the scene matches the satellite. --de/--dn/--du add metres.
ENU_OFFSET = np.array([0.20, 1.37, 0.0]) + np.array([args.de, args.dn, args.du])
print(f"[georef] applying ENU compensation offset "
      f"dE {ENU_OFFSET[0]:+.2f} dN {ENU_OFFSET[1]:+.2f} dU {ENU_OFFSET[2]:+.2f} m")

def to_latlon(p_orb):                            # ORB-world point -> (lon, lat, up_m)
    e, n_, u = s * (Rm @ np.asarray(p_orb, float)) + t + ENU_OFFSET
    return (lon0 + math.degrees(e / (R_E * math.cos(math.radians(lat0)))),
            lat0 + math.degrees(n_ / R_E), u)

def coords(pts, alt=lambda u: 0.0):              # list of ORB pts -> KML coord string
    out = []
    for p in pts:
        lo, la, u = to_latlon(p); out.append(f"{lo:.8f},{la:.8f},{alt(u):.2f}")
    return " ".join(out)

def line(name, pts, style, clamp=True):
    am = "clampToGround" if clamp else "absolute"
    return (f'<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl><LineString>'
            f'<tessellate>1</tessellate><altitudeMode>{am}</altitudeMode>'
            f'<coordinates>{coords(pts)}</coordinates></LineString></Placemark>')

def vertical(name, center, height, style):       # extruded vertical line at the geo position
    lo, la, _ = to_latlon(center)
    return (f'<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl><LineString>'
            f'<extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>'
            f'<coordinates>{lo:.8f},{la:.8f},0 {lo:.8f},{la:.8f},{height:.2f}</coordinates></LineString></Placemark>')

K = ['<?xml version="1.0" encoding="UTF-8"?>',
     '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
     '<name>seq04 static scene (geo-referenced)</name>']
PINK = 'ffb469ff'                                            # hot pink (KML aabbggrr)
arrow_col = 'ff0000ff' if args.arrows_only else 'ff00ffff'   # red in arrows-only mode, else yellow
lane_col = 'ffff0000' if args.lane_only else 'ffb0b0b0'      # blue in lane-only mode, else grey
sign_col, light_col, guide_col = 'ffff00ff', 'ff00c800', 'ffffffff'
if args.pink:                                                # all object types pink
    arrow_col = sign_col = light_col = guide_col = PINK
for sid, col, w in [('lane', lane_col, 2), ('arrow', arrow_col, 2), ('sign', sign_col, 3),
                    ('light', light_col, 3), ('guide', guide_col, 2), ('pole', 'ff0080ff', 3),
                    ('trk', 'a00000ff', 2)]:
    K.append(f'<Style id="{sid}"><LineStyle><color>{col}</color><width>{w}</width></LineStyle>'
             f'<PolyStyle><color>40{col[2:]}</color></PolyStyle></Style>')

# which sections to include (the *-only flags are mutually exclusive; no flag = full export)
full = not (args.lane_only or args.arrows_only or args.objects_only)
inc_lane = args.lane_only or full
inc_arrows = args.arrows_only or args.objects_only or full
inc_objs = args.objects_only or full        # signs / lights / guideposts
inc_extra = full                            # poles + GNSS track

# lane borders
if inc_lane:
    lm = json.loads(LANE.read_text())
    K.append('<Folder><name>Ego lane</name>')
    K.append(line('left border', np.array(lm['left'], float), 'lane'))
    K.append(line('right border', np.array(lm['right'], float), 'lane'))
    K.append('</Folder>')

# road arrows (closed glyph footprints)
ar = json.loads(ARROWS.read_text())
if inc_arrows:
    K.append('<Folder><name>Road arrows</name>')
    for i, a in enumerate(ar):
        g = np.array(a['glyph_world'], float); g = np.vstack([g, g[0]])
        K.append(line(a.get('cls', f'arrow {i}'), g, 'arrow'))
    K.append('</Folder>')

# signs / lights / guideposts / poles
ps = json.loads(PS.read_text())
def speed(name):
    m = __import__('re').search(r'(\d+)\s*km/h', name); return m.group(1) if m else name
if inc_objs:
    K.append('<Folder><name>Traffic signs</name>')
    for sgn in ps.get('signs', []):
        K.append(vertical(speed(sgn['name']), sgn['center'], 2.2, 'sign'))
    K.append('</Folder><Folder><name>Traffic lights</name>')
    for tl in ps.get('traffic_lights', []):
        K.append(vertical('traffic light', tl['center'], 5.0, 'light'))
    K.append('</Folder><Folder><name>Guideposts</name>')
    for gp in ps.get('guideposts', []):
        K.append(vertical('guidepost', gp['center'], max(gp.get('height', 1.0), 0.8), 'guide'))
    K.append('</Folder>')
if inc_extra:
    K.append('<Folder><name>Poles</name>')
    for p in ps.get('poles', []):
        K.append(vertical('pole', p['center'], min(max(p.get('height', 6.0), 2.0), 12.0), 'pole'))
    K.append('</Folder>')

    # GNSS drive track (reference)
    K.append('<Folder><name>GNSS track (ref)</name>')
    K.append('<Placemark><name>GNSS</name><styleUrl>#trk</styleUrl><LineString><tessellate>1</tessellate>'
             f'<coordinates>{" ".join(f"{ox[i,1]:.8f},{ox[i,0]:.8f},0" for i in range(N))}</coordinates></LineString></Placemark>')
    K.append('</Folder>')
K.append('</Document></kml>')

OUT.write_text("".join(K))
if args.lane_only:
    summary = "ego-lane boundaries (blue)"
elif args.arrows_only:
    summary = f"{len(ar)} arrows (red)"
else:
    extras = "" if args.objects_only else f", {len(ps.get('poles',[]))} poles, lane borders, GNSS track"
    tint = " (pink)" if args.pink else ""
    summary = (f"{len(ar)} arrows, {len(ps.get('signs',[]))} signs, "
               f"{len(ps.get('traffic_lights',[]))} lights, {len(ps.get('guideposts',[]))} guideposts{extras}{tint}")
print(f"[georef] wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB): {summary}")
