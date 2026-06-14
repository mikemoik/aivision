"""T1 isolation test: run the map-gen `slam` step and compare its trajectory to
the saved GT run via Sim(3) ATE/RPE.

ORB-SLAM3 is non-deterministic, so this is tolerance-based: a fresh stereo run
should land within a few cm of the GT trajectory (itself an ORB3 run). The
contract under test is `map-gen/slam/out/trajectory.json`.

    uv run python scripts/test_mapgen_slam.py [--run-id t1-slam-test] [--ate-max 0.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.kitti import absolute_trajectory_error, relative_pose_error
from aivision.map.mapgen import MapGen

GT_TRAJ = Path("map/kitti-seq04/bindata/mapgen/kitti-seq04-backfill/"
               "map-gen/slam/out/trajectory.json")


def centres(traj_json: Path) -> np.ndarray:
    poses = np.asarray(json.loads(traj_json.read_text())["poses"], dtype=float)
    return poses[:, [3, 7, 11]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="t1-slam-test")
    ap.add_argument("--ate-max", type=float, default=0.5, help="ATE rmse tolerance (m)")
    ap.add_argument("--rpe-max", type=float, default=0.2, help="RPE(Δ=10) rmse tolerance (m)")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir, skip ORB")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    if not args.no_run:
        gen.run(only=["slam"], force=True, run_id=args.run_id)

    new_traj = gen.layout.run(args.run_id).step_out("slam") / "trajectory.json"
    est, gt = centres(new_traj), centres(GT_TRAJ)
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    ate = absolute_trajectory_error(est, gt)
    rpe = relative_pose_error(est, gt, delta=10)
    print(f"\nT1 slam vs GT  ({n} poses)")
    print(f"  ATE  rmse={ate['rmse']:.4f} mean={ate['mean']:.4f} "
          f"median={ate['median']:.4f} max={ate['max']:.4f} scale={ate['scale']:.5f} m")
    print(f"  RPE  rmse={rpe['rmse']:.4f} mean={rpe['mean']:.4f} "
          f"median={rpe['median']:.4f} max={rpe['max']:.4f} m (Δ=10)")

    ok = ate["rmse"] <= args.ate_max and rpe["rmse"] <= args.rpe_max
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} "
          f"(ATE≤{args.ate_max}, RPE≤{args.rpe_max})\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
