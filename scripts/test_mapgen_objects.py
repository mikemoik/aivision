"""T5 isolation test: run the map-gen `objects` step and compare to the saved GT.

The GT objects artifact (poles_signs_world.json) WAS migrated, so this is a
tolerance-vs-GT test. `objects` has four upstreams; depth + segmentation were
never migrated to the GT run, so all four are seeded from the artifacts that
produced the GT objects before the step runs, isolating `objects` from ORB /
depth-fusion / Mask2Former nondeterminism:

  * slam          ← GT trajectory.json
  * depth         ← legacy fused-rho mkv (renamed to the depth-step contract name)
  * segmentation  ← legacy perception cache (frame_NNNNNN.npz)
  * lanes         ← GT lane model (declared input; not used for the JSON)

GTSRB ViT + YOLOv12 delineator inference and the triangulation are deterministic
given identical inputs, so a faithful refactor should reproduce GT closely. Each
GT object is greedily matched to the nearest new object of the same kind (within
--match-r); the 3D centre deviation + count parity are reported.

    uv run python scripts/test_mapgen_objects.py [--run-id t5-objects-test] [--no-run]
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
GT_OBJECTS = GT_DIR / "objects/out/poles_signs_world.json"

LEGACY_RHO = Path("outputs/pocs/orb_slam3/inv_depth_fused_mono_rho_gf_16bit.mkv")
LEGACY_CACHE = Path("outputs/pocs/orb_slam3/perception")

CATEGORIES = ("poles", "signs", "traffic_lights", "guideposts")


def seed_upstreams(run) -> None:
    """Populate slam/depth/segmentation/lanes out/ from the GT-producing artifacts."""
    run.ensure()
    run.ensure_step("slam")
    shutil.copy(GT_SLAM_TRAJ, run.step_out("slam") / "trajectory.json")

    run.ensure_step("depth")
    shutil.copy(LEGACY_RHO, run.step_out("depth") / "inv_depth_rho16.mkv")

    run.ensure_step("segmentation")
    seg_out = run.step_out("segmentation")
    for npz in sorted(LEGACY_CACHE.glob("frame_*.npz")):
        shutil.copy(npz, seg_out / npz.name)

    run.ensure_step("lanes")
    for fname in ("lane_polylines.json", "lane_polylines_model.json"):
        shutil.copy(GT_LANES / fname, run.step_out("lanes") / fname)


def match(gt: list, new: list, r: float) -> dict:
    """Greedy nearest-neighbour match of GT→new centres within radius r (m)."""
    gc = np.array([o["center"] for o in gt]) if gt else np.empty((0, 3))
    nc = np.array([o["center"] for o in new]) if new else np.empty((0, 3))
    used = set()
    devs = []
    for g in gc:
        if not len(nc):
            break
        d = np.linalg.norm(nc - g, axis=1)
        for j in np.argsort(d):
            if j in used:
                continue
            if d[j] <= r:
                used.add(int(j))
                devs.append(float(d[j]))
            break
    devs = np.asarray(devs)
    return {"gt": len(gt), "new": len(new), "matched": len(devs),
            "dev_max": float(devs.max()) if len(devs) else 0.0,
            "dev_rms": float(np.sqrt((devs ** 2).mean())) if len(devs) else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="t5-objects-test")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir")
    ap.add_argument("--match-r", type=float, default=1.0,
                    help="max centre distance to count a GT object as reproduced (m)")
    ap.add_argument("--pos-max", type=float, default=0.50,
                    help="max allowed centre deviation over matched objects (m)")
    ap.add_argument("--min-match", type=float, default=0.80,
                    help="min fraction of GT objects that must be matched")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    run = gen.layout.run(args.run_id)
    if not args.no_run:
        seed_upstreams(run)
        gen.run(only=["objects"], force=True, run_id=args.run_id)

    gt = json.loads(GT_OBJECTS.read_text())
    new = json.loads((run.step_out("objects") / "poles_signs_world.json").read_text())

    print(f"\nT5 objects vs GT  (run={args.run_id}, match_r={args.match_r} m)")
    worst_dev = 0.0
    n_gt = n_matched = 0
    for cat in CATEGORIES:
        m = match(gt.get(cat, []), new.get(cat, []), args.match_r)
        n_gt += m["gt"]
        n_matched += m["matched"]
        worst_dev = max(worst_dev, m["dev_max"])
        print(f"  {cat:15s} GT {m['gt']:3d} | new {m['new']:3d} | matched {m['matched']:3d}  "
              f"dev max={m['dev_max']:.3f} rms={m['dev_rms']:.3f} m")

    frac = n_matched / max(n_gt, 1)
    print(f"\n  matched {n_matched}/{n_gt} ({frac:.0%})  worst centre dev {worst_dev:.3f} m")
    ok = frac >= args.min_match and worst_dev <= args.pos_max
    print(f"  verdict: {'PASS' if ok else 'FAIL'} "
          f"(need ≥{args.min_match:.0%} matched, ≤{args.pos_max} m dev)\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
