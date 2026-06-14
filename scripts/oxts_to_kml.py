#!/usr/bin/env python3
"""Render a KITTI oxts track as KML (line + heading arrows) for Google Earth.

KITTI oxts fields: lat lon alt roll pitch yaw ...  (yaw: 0=east, +CCW, rad).
KML heading is degrees clockwise from north, so heading = (90 - deg(yaw)) % 360.
Opening the .kml in Google Earth puts the track on its satellite basemap.
"""
import math
import os
import sys

OXTS_DIR = "data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data"
OUT = "outputs/kitti04_track.kml"
FRAME_LO, FRAME_HI = 0, 270   # KITTI odometry seq04 = drive_0016 frames [0, 270]
ARROW_EVERY = 10              # place a heading arrow every N frames

ARROW_ICON = "http://maps.google.com/mapfiles/kml/shapes/arrow.png"


def read_oxts(d, lo, hi):
    pts = []
    for i in range(lo, hi + 1):
        p = os.path.join(d, f"{i:010d}.txt")
        with open(p) as f:
            v = f.read().split()
        lat, lon, alt, yaw = float(v[0]), float(v[1]), float(v[2]), float(v[5])
        heading = (90.0 - math.degrees(yaw)) % 360.0
        pts.append((i, lat, lon, alt, heading))
    return pts


def kml(pts):
    line = " ".join(f"{lon:.9f},{lat:.9f},0" for _, lat, lon, _, _ in pts)
    arrows = []
    for i, lat, lon, _, hd in pts:
        if i % ARROW_EVERY and i not in (pts[0][0], pts[-1][0]):
            continue
        arrows.append(f"""    <Placemark>
      <name>f{i}</name>
      <Style><IconStyle>
        <heading>{hd:.1f}</heading>
        <scale>0.6</scale>
        <Icon><href>{ARROW_ICON}</href></Icon>
      </IconStyle></Style>
      <Point><coordinates>{lon:.9f},{lat:.9f},0</coordinates></Point>
    </Placemark>""")
    arrows = "\n".join(arrows)
    s, e = pts[0], pts[-1]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>KITTI seq04 (drive_0016) track</name>
  <Style id="trk"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>
  <Placemark>
    <name>track</name><styleUrl>#trk</styleUrl>
    <LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>
      <coordinates>{line}</coordinates>
    </LineString>
  </Placemark>
  <Placemark><name>start</name>
    <Style><IconStyle><color>ff00ff00</color></IconStyle></Style>
    <Point><coordinates>{s[2]:.9f},{s[1]:.9f},0</coordinates></Point>
  </Placemark>
  <Placemark><name>end</name>
    <Style><IconStyle><color>ff0000ff</color></IconStyle></Style>
    <Point><coordinates>{e[2]:.9f},{e[1]:.9f},0</coordinates></Point>
  </Placemark>
{arrows}
</Document>
</kml>
"""


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pts = read_oxts(os.path.join(root, OXTS_DIR), FRAME_LO, FRAME_HI)
    out = os.path.join(root, OUT)
    with open(out, "w") as f:
        f.write(kml(pts))
    print(f"{len(pts)} frames -> {out}")


if __name__ == "__main__":
    main()
