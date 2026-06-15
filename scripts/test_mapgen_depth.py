"""T2 isolation test: run the map-gen `depth` step and sanity-check its rho mkvs.

The GT depth artifact was never migrated (the backfill step.json only records
"artifact not migrated"), so there is no GT mkv to diff against. Instead this is
a sanity test: a fresh run should produce a well-formed rho stream — sensible
valid-pixel coverage, a plausible median scene depth, and the stereo-only stream
should be a (sky-free) subset of the mono-fused one.

To isolate depth from ORB-SLAM3 nondeterminism, the upstream `slam` trajectory is
seeded from the saved GT run before the depth step runs.

    uv run python scripts/test_mapgen_depth.py [--run-id t2-depth-test] [--no-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from aivision.map.mapgen import MapGen

GT_SLAM_TRAJ = Path("map/kitti-seq04/bindata/mapgen/kitti-seq04-backfill/"
                    "map-gen/slam/out/trajectory.json")


def decode_rho_stream(video: Path, W: int, H: int):
    """Yield successive gray16 rho frames as float32 H×W (0 = invalid)."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE,
    )
    fb = W * H * 2
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            pix = np.frombuffer(raw, dtype=np.uint16).reshape(H, W).astype(np.float32)
            yield np.where(pix > 0, (pix - 1.0) / 65534.0, 0.0)
    finally:
        proc.stdout.close()
        proc.wait()


def stream_stats(video: Path, W: int, H: int) -> dict:
    n = 0
    cover = 0.0
    z_meds = []
    for rho in decode_rho_stream(video, W, H):
        valid = rho > 0
        n += 1
        cover += valid.mean()
        if valid.any():
            z_meds.append(float(np.median(1.0 / rho[valid])))
    return {"n": n, "coverage": cover / max(n, 1),
            "median_Z": float(np.median(z_meds)) if z_meds else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="t2-depth-test")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir, skip depth")
    ap.add_argument("--cover-min", type=float, default=0.70,
                    help="min mean valid coverage for the mono-fused stream")
    ap.add_argument("--z-lo", type=float, default=3.0, help="plausible median Z lower bound (m)")
    ap.add_argument("--z-hi", type=float, default=80.0, help="plausible median Z upper bound (m)")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    run = gen.layout.run(args.run_id)

    if not args.no_run:
        # Seed upstream slam from GT so depth is tested in isolation from ORB.
        run.ensure()
        run.ensure_step("slam")
        shutil.copy(GT_SLAM_TRAJ, run.step_out("slam") / "trajectory.json")
        gen.run(only=["depth"], force=True, run_id=args.run_id)

    out = run.step_out("depth")
    meta = json.loads((out / "inv_depth_rho16.json").read_text())
    W, H = meta["width"], meta["height"]
    fused = stream_stats(out / "inv_depth_rho16.mkv", W, H)
    stereo = stream_stats(out / "inv_depth_rho16_stereo.mkv", W, H)

    print(f"\nT2 depth sanity  (run={args.run_id})")
    print(f"  meta: fx={meta['fx']:.3f} baseline={meta['baseline_m']:.4f}m "
          f"n={meta['n_frames']} affine a={meta['mono_affine_a']:.3f} b={meta['mono_affine_b']:.4f}")
    print(f"  fused  : n={fused['n']} coverage={fused['coverage']:.1%} median_Z={fused['median_Z']:.2f}m")
    print(f"  stereo : n={stereo['n']} coverage={stereo['coverage']:.1%} median_Z={stereo['median_Z']:.2f}m")

    checks = {
        "frame_count_matches_meta": fused["n"] == meta["n_frames"] == stereo["n"],
        "fused_coverage_ok": fused["coverage"] >= args.cover_min,
        "stereo_subset_of_fused": stereo["coverage"] <= fused["coverage"] + 1e-6,
        "median_Z_plausible": args.z_lo <= fused["median_Z"] <= args.z_hi,
    }
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    ok = all(checks.values())
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'}\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
