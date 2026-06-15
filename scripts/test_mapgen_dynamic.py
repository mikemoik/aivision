"""Dynamic-step sanity test: run the map-gen `dynamic` step and validate the cache.

Dynamic motion-segmentation has no migrated GT artifact, so this is a sanity
test. The four upstreams are seeded from already-produced artifacts so the step
is isolated from re-running slam/depth/flow:

  * slam   ← a prior run's trajectory.json (only RELATIVE pose is used → gauge-free)
  * depth  ← a prior run's stereo-only rho mkv (per-frame SGBM, pose-independent)
  * flow   ← the flow step's cached fwd/bwd npz

Checks: one label map per pair (n_frames-1); values ∈ {0,1,2,3}; the class mix is
plausible for a mostly-static forward drive (static dominant, some moving from the
oncoming car, pot_dyn from tracked vehicles); and at least one frame actually
flags MOVING pixels (the detector/dense core found a real mover).

    uv run python scripts/test_mapgen_dynamic.py [--run-id dynamic-test] [--no-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from aivision.map.mapgen import MapGen

FULL_RUN = Path("map/kitti-seq04/bindata/mapgen/kitti-seq04-20260615T103344/map-gen")
FLOW_RUN = Path("map/kitti-seq04/bindata/mapgen/flow-test/map-gen")


def seed_upstreams(run) -> None:
    run.ensure()
    run.ensure_step("slam")
    shutil.copy(FULL_RUN / "slam/out/trajectory.json", run.step_out("slam") / "trajectory.json")
    run.ensure_step("depth")
    shutil.copy(FULL_RUN / "depth/out/inv_depth_rho16_stereo.mkv",
                run.step_out("depth") / "inv_depth_rho16_stereo.mkv")
    run.ensure_step("flow")
    fdst = run.step_out("flow")
    for f in sorted((FLOW_RUN / "flow/out").glob("*")):
        shutil.copy(f, fdst / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="dynamic-test")
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    run = gen.layout.run(args.run_id)
    if not args.no_run:
        seed_upstreams(run)
        gen.run(only=["dynamic"], force=True, run_id=args.run_id)

    out = run.step_out("dynamic")
    manifest = json.loads((out / "manifest.json").read_text())
    npz = sorted(out.glob("dynamic_*.npz"))
    n_pairs = manifest["n_pairs"]

    print(f"\ndynamic sanity  (run={args.run_id})")
    print(f"  manifest: {manifest['n_frames']} frames → {n_pairs} pairs "
          f"{manifest['w']}x{manifest['h']}  window=±{manifest['window']} "
          f"objects={manifest['do_objects']}")
    print(f"  npz on disk: {len(npz)}  (expected {n_pairs})")
    print(f"  mean class %: {manifest['mean_pct']}")

    count_ok = len(npz) == n_pairs == manifest["n_frames"] - 1

    # scan every label map: value range + how many frames carry MOVING pixels
    vals_ok = True
    frames_with_moving = 0
    max_moving_px = 0
    for p in npz:
        lab = np.load(p)["label"]
        if lab.dtype != np.uint8 or lab.min() < 0 or lab.max() > 3:
            vals_ok = False
        mv = int((lab == 1).sum())
        frames_with_moving += mv > 0
        max_moving_px = max(max_moving_px, mv)

    m = manifest["mean_pct"]
    mix_ok = m["static"] > 40.0 and m["moving"] >= 0.0 and m["moving"] < 30.0
    moving_ok = frames_with_moving > 0 and max_moving_px > 200
    print(f"  frames with MOVING px: {frames_with_moving}/{len(npz)}  "
          f"(max {max_moving_px} px in a frame)")

    ok = count_ok and vals_ok and mix_ok and moving_ok
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} "
          f"(count={count_ok} values={vals_ok} mix={mix_ok} moving={moving_ok})\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
