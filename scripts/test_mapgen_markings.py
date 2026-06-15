"""T6 isolation test: run the map-gen `markings` step and compare to the saved GT.

The GT markings artifact (arrows_world.json) WAS migrated, so this is a
tolerance-vs-GT test. `markings` has two upstreams; both are seeded from the
artifacts that produced the GT arrows before the step runs, isolating `markings`
from ORB / lane-fit nondeterminism:

  * slam   ← GT trajectory.json
  * lanes  ← GT lane model (lane_polylines_model.json = road plane + ego-lane gate)

The RLMD road-marking detector and the lift/cluster/pose-fit are deterministic
given identical inputs, so a faithful refactor should reproduce GT closely. Each
GT arrow is greedily matched to the nearest new arrow (within --match-r); the 3D
centroid deviation, heading (yaw-to-lane) deviation and count parity are reported.

    uv run python scripts/test_mapgen_markings.py [--run-id t6-markings-test] [--no-run]
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
GT_LANES = GT_DIR / "lanes/out"
GT_MARKINGS = GT_DIR / "markings/out/arrows_world.json"


def seed_upstreams(run) -> None:
    """Populate slam/lanes out/ from the GT-producing artifacts."""
    run.ensure()
    run.ensure_step("slam")
    shutil.copy(GT_SLAM_TRAJ, run.step_out("slam") / "trajectory.json")

    run.ensure_step("lanes")
    for fname in ("lane_polylines.json", "lane_polylines_model.json"):
        shutil.copy(GT_LANES / fname, run.step_out("lanes") / fname)


def match(gt: list, new: list, r: float) -> dict:
    """Greedy nearest-centroid match of GT→new arrows within radius r (m).

    Reports the worst matched centroid distance and the worst |Δyaw| over matches.
    """
    gc = np.array([a["centroid"] for a in gt]) if gt else np.empty((0, 3))
    nc = np.array([a["centroid"] for a in new]) if new else np.empty((0, 3))
    used = set()
    devs, yaws = [], []
    for gi, g in enumerate(gc):
        if not len(nc):
            break
        d = np.linalg.norm(nc - g, axis=1)
        for j in np.argsort(d):
            if j in used:
                continue
            if d[j] <= r:
                used.add(int(j))
                devs.append(float(d[j]))
                yaws.append(abs(gt[gi]["yaw_to_lane_deg"] - new[int(j)]["yaw_to_lane_deg"]))
            break
    devs, yaws = np.asarray(devs), np.asarray(yaws)
    return {"gt": len(gt), "new": len(new), "matched": len(devs),
            "dev_max": float(devs.max()) if len(devs) else 0.0,
            "dev_rms": float(np.sqrt((devs ** 2).mean())) if len(devs) else 0.0,
            "yaw_max": float(yaws.max()) if len(yaws) else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="t6-markings-test")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir")
    ap.add_argument("--match-r", type=float, default=1.5,
                    help="max centroid distance to count a GT arrow as reproduced (m)")
    ap.add_argument("--pos-max", type=float, default=0.50,
                    help="max allowed centroid deviation over matched arrows (m)")
    ap.add_argument("--yaw-max", type=float, default=2.0,
                    help="max allowed |Δyaw-to-lane| over matched arrows (deg)")
    ap.add_argument("--min-match", type=float, default=0.80,
                    help="min fraction of GT arrows that must be matched")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    run = gen.layout.run(args.run_id)
    if not args.no_run:
        seed_upstreams(run)
        gen.run(only=["markings"], force=True, run_id=args.run_id)

    gt = json.loads(GT_MARKINGS.read_text())
    new = json.loads((run.step_out("markings") / "arrows_world.json").read_text())

    print(f"\nT6 markings vs GT  (run={args.run_id}, match_r={args.match_r} m)")
    m = match(gt, new, args.match_r)
    frac = m["matched"] / max(m["gt"], 1)
    print(f"  arrows  GT {m['gt']:3d} | new {m['new']:3d} | matched {m['matched']:3d}  "
          f"dev max={m['dev_max']:.3f} rms={m['dev_rms']:.3f} m | yaw max={m['yaw_max']:.3f}°")
    print(f"\n  matched {m['matched']}/{m['gt']} ({frac:.0%})  "
          f"worst centroid dev {m['dev_max']:.3f} m  worst Δyaw {m['yaw_max']:.3f}°")
    ok = (frac >= args.min_match and m["dev_max"] <= args.pos_max
          and m["yaw_max"] <= args.yaw_max)
    print(f"  verdict: {'PASS' if ok else 'FAIL'} "
          f"(need ≥{args.min_match:.0%} matched, ≤{args.pos_max} m dev, ≤{args.yaw_max}° yaw)\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
