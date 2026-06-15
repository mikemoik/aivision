"""T4 isolation test: run the map-gen `lanes` step and compare to the saved GT.

Unlike depth/segmentation, the GT lanes `out/` WAS migrated — both artifacts exist:
  * lane_polylines.json        (independent left/right border polylines → assemble)
  * lane_polylines_model.json  (coupled lane model → objects, markings)
so this is a tolerance-vs-GT test, not just sanity. The `slam` trajectory is seeded
from the GT run first so the lanes step sees identical poses (EgoLanes ONNX + SGBM
are deterministic, so a faithful reconstruction should land within a few cm).

For each artifact, left/right are resampled onto a common Z grid (the overlap of GT
and new Z spans) and the lateral (X) + height (Y) deviations are reported.

    uv run python scripts/test_mapgen_lanes.py [--run-id t4-lanes-test] [--no-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from aivision.map.mapgen import MapGen

GT_DIR = Path("map/kitti-seq04/bindata/mapgen/kitti-seq04-backfill/map-gen")
GT_SLAM_TRAJ = GT_DIR / "slam/out/trajectory.json"


def deviation(gt: dict, new: dict, side: str) -> dict:
    """Resample both polylines' X(z),Y(z) onto the common Z overlap; return dev stats."""
    g = np.asarray(gt[side], dtype=float)
    n = np.asarray(new[side], dtype=float)
    if len(g) < 2 or len(n) < 2:
        return {"n": 0, "x_max": float("nan"), "x_rms": float("nan"),
                "y_max": float("nan"), "y_rms": float("nan")}
    g, n = g[np.argsort(g[:, 2])], n[np.argsort(n[:, 2])]
    z0 = max(g[0, 2], n[0, 2])
    z1 = min(g[-1, 2], n[-1, 2])
    zs = np.linspace(z0, z1, 200)
    gx, nx = np.interp(zs, g[:, 2], g[:, 0]), np.interp(zs, n[:, 2], n[:, 0])
    gy, ny = np.interp(zs, g[:, 2], g[:, 1]), np.interp(zs, n[:, 2], n[:, 1])
    dx, dy = np.abs(gx - nx), np.abs(gy - ny)
    return {"n": len(zs), "z_span": (round(z0, 1), round(z1, 1)),
            "x_max": float(dx.max()), "x_rms": float(np.sqrt((dx ** 2).mean())),
            "y_max": float(dy.max()), "y_rms": float(np.sqrt((dy ** 2).mean()))}


def report(name: str, gt: dict, new: dict) -> bool:
    print(f"\n  {name}")
    worst = 0.0
    for side in ("left", "right"):
        d = deviation(gt, new, side)
        print(f"    {side:5s} z{d.get('z_span', '?')} "
              f"X: max={d['x_max']:.3f} rms={d['x_rms']:.3f}  "
              f"Y: max={d['y_max']:.3f} rms={d['y_rms']:.3f} m  "
              f"(GT {len(gt[side])} / new {len(new[side])} verts)")
        worst = max(worst, d["x_max"], d["y_max"])
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="t4-lanes-test")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir")
    ap.add_argument("--lat-max", type=float, default=0.50,
                    help="max allowed lateral/height deviation vs GT (m)")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    run = gen.layout.run(args.run_id)
    if not args.no_run:
        run.ensure()
        run.ensure_step("slam")
        shutil.copy(GT_SLAM_TRAJ, run.step_out("slam") / "trajectory.json")
        gen.run(only=["lanes"], force=True, run_id=args.run_id)

    out = run.step_out("lanes")
    gt_out = GT_DIR / "lanes/out"
    print(f"\nT4 lanes vs GT  (run={args.run_id}, tol={args.lat_max} m)")

    worst = 0.0
    for fname, label in (("lane_polylines.json", "border polylines (→assemble)"),
                         ("lane_polylines_model.json", "coupled lane model (→objects,markings)")):
        gt = json.loads((gt_out / fname).read_text())
        new = json.loads((out / fname).read_text())
        worst = max(worst, report(label, gt, new))
        if "lane_width_m" in gt:
            print(f"    lane_width_m: GT {gt['lane_width_m']} | new {new['lane_width_m']}")

    ok = worst <= args.lat_max
    print(f"\n  worst deviation: {worst:.3f} m")
    print(f"  verdict: {'PASS' if ok else 'FAIL'} (tol {args.lat_max} m)\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
