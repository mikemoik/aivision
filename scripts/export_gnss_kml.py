"""Export the seq04 GNSS (OXTS) frames as KML for Google Earth — matching the notebook.

Per frame: a red DOT at the GNSS point + a short 1 m HEADING line whose TIP ends AT the dot
(it trails 1 m back along the OXTS yaw). Heading drawn as geometry (icon orientation in Google
Earth is unreliable). Heading from OXTS yaw (0=east, CCW+), verified to match the OXTS velocity
& GPS trajectory to ~0.1 deg.  seq04 == KITTI raw drive 2011_09_30_drive_0016; 271 frames (0..270).
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OX = REPO / "data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data"
OUT = REPO / "outputs" / "seq04_gnss.kml"
N = 271
LINE = 1.0     # metres; heading line runs 1 m behind -> the GNSS point (tip at the dot)
DOT = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"

oxts = np.array([np.loadtxt(OX / f"{i:010d}.txt") for i in range(N)])         # 30-field rows
lat, lon, yaw = oxts[:, 0], oxts[:, 1], oxts[:, 5]                            # yaw 0=east CCW+ rad
e, n = np.cos(yaw), np.sin(yaw)                                               # heading unit (E, N)
R = 6378137.0
def dll(dN, dE, at_lat):                                                       # metre offset -> deg
    return math.degrees(dN / R), math.degrees(dE / (R * math.cos(math.radians(at_lat))))

P = ['<?xml version="1.0" encoding="UTF-8"?>',
     '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
     '<name>seq04 GNSS frames + heading (2011_09_30_drive_0016)</name>',
     '<Style id="hdg"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle></Style>',  # yellow, 2pt
     '<Style id="dot"><IconStyle><color>ff0000ff</color><scale>0.5</scale>'
     f'<Icon><href>{DOT}</href></Icon></IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>']

# heading lines: one MultiGeometry of per-frame 1 m LineStrings (base 1 m behind -> tip at point)
lines = []
for i in range(N):
    dN, dE = dll(LINE * n[i], LINE * e[i], lat[i])
    lines.append('<LineString><tessellate>1</tessellate><coordinates>'
                 f'{lon[i]-dE:.8f},{lat[i]-dN:.8f},0 {lon[i]:.8f},{lat[i]:.8f},0'
                 '</coordinates></LineString>')
P.append(f'<Placemark><name>heading 1m ({N})</name><styleUrl>#hdg</styleUrl>'
         f'<MultiGeometry>{"".join(lines)}</MultiGeometry></Placemark>')

# frame dots: one Placemark per frame (shared #dot style) so each carries frame/heading info
P.append('<Folder><name>GNSS frames</name>')
for i in range(N):
    bdeg = math.degrees(math.atan2(e[i], n[i])) % 360.0
    P.append(f'<Placemark><name>frame {i}</name><description>heading {bdeg:.1f} deg</description>'
             f'<styleUrl>#dot</styleUrl><Point><coordinates>{lon[i]:.8f},{lat[i]:.8f},0</coordinates></Point></Placemark>')
P.append('</Folder></Document></kml>')

OUT.write_text("".join(P))
print(f"wrote {OUT}  (dots + 1 m heading lines, {OUT.stat().st_size/1024:.0f} KB)")
