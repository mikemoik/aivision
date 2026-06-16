"""Virtual-camera-sensor test: run the `objects` step, then exercise the reader.

The objects step now emits a second artifact alongside poles_signs_world.json —
`detections_by_frame.json`, the raw per-frame static-object detections in camera
coordinates (the virtual sensor), serialized as MapObject.to_dict() — the SAME data
structures as the map. This test seeds the four objects upstreams from the
GT-producing artifacts (same as test_mapgen_objects), runs objects-only, then loads
VirtualCameraSensor and checks:

  1. artifact present + self-describing header (schema, K, image size, n_frames);
  2. detections reconstruct to real MapObject subclasses (same types as the map),
     span many frames; per-class totals >= aggregated map counts (each physical
     object is seen in several frames);
  3. CORRECTNESS — every det's camera-frame centre reprojects through K back onto its
     stored image_uv attribute (pose-independent proof the geometry is consistent);
  4. depth attr == centre.z, range >= depth, depth within the lift gate (4..zmax).

    uv run python scripts/test_virtual_camera.py [--run-id vcam-test] [--no-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.mapgen import MapGen
from aivision.map.objects import (
    GuidePost, LaneMarking, Pole, RoadArrow, TrafficLight, TrafficSign)
from aivision.map.virtual_camera import VirtualCameraSensor
from test_mapgen_objects import GT_OBJECTS, seed_upstreams  # reuse the T5 seeding

# MapClass value -> expected MapObject subclass (all six can appear in the sensor)
CATEGORIES = {"pole": Pole, "sign": TrafficSign, "traffic_light": TrafficLight,
              "guide_post": GuidePost, "lane_marking": LaneMarking, "road_arrow": RoadArrow}
# aggregated-map json key -> MapClass value (for the four triangulated object classes)
WORLD_KEY = {"poles": "pole", "signs": "sign",
             "traffic_lights": "traffic_light", "guideposts": "guide_post"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="vcam-test")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir")
    ap.add_argument("--uv-tol", type=float, default=0.5,
                    help="max reprojection error pos_cam->image_uv (px)")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    run = gen.layout.run(args.run_id)
    if not args.no_run:
        seed_upstreams(run)
        # lanes first (markings + objects read its model); each emits a *_by_frame.json
        for role in ("lanes", "markings", "objects"):
            gen.run(only=[role], force=True, run_id=args.run_id)

    sensor = VirtualCameraSensor.from_map("kitti-seq04", run_id=args.run_id)
    print(f"\nVirtual camera sensor  (run={args.run_id})")
    print(f"  {sensor!r}")
    print(f"  sources: {[p.name for p in sensor.paths]}")

    ok = True

    # --- header sanity
    if sensor.K.shape != (3, 3) or sensor.n_frames <= 0:
        print("  FAIL: bad header (K / n_frames)"); ok = False
    W, H = sensor.image_size

    # --- gather all detections
    per_frame = [len(sensor.detections(t)) for t in range(sensor.n_frames)]
    total = sum(per_frame)
    n_frames_hit = sum(1 for c in per_frame if c)
    print(f"  {total} detections over {n_frames_hit}/{sensor.n_frames} frames "
          f"(max {max(per_frame)}/frame)")
    if total == 0:
        print("  FAIL: no detections"); ok = False

    # --- per-class totals + type check (all six map classes can appear)
    world = json.loads(GT_OBJECTS.read_text())
    by_cls: dict[str, int] = {}
    bad_type = 0
    for _, dets in sensor:
        for d in dets:
            by_cls[d.cls.value] = by_cls.get(d.cls.value, 0) + 1
            if not isinstance(d, CATEGORIES.get(d.cls.value, ())):
                bad_type += 1
    # the 4 triangulated object classes have an aggregated-map count to compare to
    for cval in ("pole", "sign", "traffic_light", "guide_post"):
        cat = next(k for k, v in WORLD_KEY.items() if v == cval)
        n_map = len(world.get(cat, []))
        n_det = by_cls.get(cval, 0)
        flag = "" if n_det >= n_map else "  <-- fewer dets than map objects?!"
        print(f"  {cval:14s} dets {n_det:4d} | map objects {n_map:3d}{flag}")
    # lanes + arrows are per-frame curves/glyphs (no single aggregated count)
    for cval in ("lane_marking", "road_arrow"):
        print(f"  {cval:14s} dets {by_cls.get(cval, 0):4d}")
    if by_cls.get("lane_marking", 0) == 0:
        print("  FAIL: no ego-lane detections"); ok = False
    if by_cls.get("road_arrow", 0) == 0:
        print("  FAIL: no road-arrow detections"); ok = False
    print(f"  wrong-type reconstructions: {bad_type}")
    if bad_type:
        print("  FAIL: detections did not reconstruct to map types"); ok = False

    # --- correctness: reproject camera-frame centre through K -> image_uv attribute
    #     (only the point-like object detections carry an image_uv; lanes/arrows don't)
    max_uv_err = 0.0
    bad_depth = checked = 0
    for t in range(sensor.n_frames):
        for d in sensor.detections(t):
            if "image_uv" not in d.attributes:
                continue
            checked += 1
            pc = sensor.center(d)
            uvw = sensor.K @ pc
            u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
            uv = d.attributes["image_uv"]
            err = float(np.hypot(u - uv[0], v - uv[1]))
            max_uv_err = max(max_uv_err, err)
            depth, rng = d.attributes["depth_m"], d.attributes["range_m"]
            if abs(depth - pc[2]) > 1e-6 or rng + 1e-6 < depth or not (4.0 < depth < 80.0):
                bad_depth += 1
    print(f"  max reprojection error centre->image_uv: {max_uv_err:.4f} px "
          f"(tol {args.uv_tol}, {checked} point dets checked)")
    print(f"  depth/range inconsistencies: {bad_depth}")
    if max_uv_err > args.uv_tol:
        print("  FAIL: centre does not reproject onto image_uv"); ok = False
    if bad_depth:
        print("  FAIL: depth/range inconsistencies"); ok = False

    # --- show a sample frame that has a lane + an arrow if possible
    def has_extras(t):
        cs = {d.cls.value for d in sensor.detections(t)}
        return "lane_marking" in cs and "road_arrow" in cs
    pick = next((t for t in range(sensor.n_frames) if has_extras(t)),
                int(np.argmax(per_frame)))
    print(f"\n  sample — frame {pick} ({len(sensor.detections(pick))} dets):")
    for d in sensor.detections(pick):
        c = sensor.center(d)
        a = d.attributes
        extra = a.get("name") or a.get("cls") or a.get("side") or ""
        rng = a.get("range_m")
        rs = f"range={rng:5.1f}m" if rng is not None else "          "
        print(f"    {d.cls.value:13s} {type(d).__name__:12s} "
              f"c=({c[0]:6.1f},{c[1]:6.1f},{c[2]:6.1f}) {rs}  {extra}")

    print(f"\n  {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
