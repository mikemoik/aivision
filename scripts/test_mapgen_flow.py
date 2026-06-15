"""Flow-step sanity test: run the map-gen `flow` step and validate the cache.

Flow has no migrated GT artifact (it's a new preprocessing role), so this is a
sanity test, not a tolerance-vs-GT test:

  * the cache has one npz per consecutive pair (n_frames - 1), each with `fwd`
    and `bwd` (H, W, 2) float16 fields;
  * flow magnitudes are plausible for a forward-driving sequence (non-trivial
    mean, finite);
  * forward/backward consistency — warping a pixel by fwd then by bwd at the
    landing point returns near the start (small FB residual over most pixels) —
    which is the property the dynamic step's occlusion gate relies on.

    uv run python scripts/test_mapgen_flow.py [--run-id flow-test] [--no-run]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from aivision.map.mapgen import MapGen


def fb_residual(fwd: np.ndarray, bwd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|fwd(x) + bwd(x + fwd(x))| per pixel (px) + an in-bounds validity mask.

    Pixels whose forward flow lands outside the image are genuine outflow/
    occlusion (heavy under forward driving), where FB consistency is undefined —
    the consumer's own occlusion gate drops them — so they are excluded from the
    consistency statistics rather than clipped to the border (which would inflate
    the residual spuriously)."""
    h, w = fwd.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    fx_, fy_ = xx + fwd[..., 0], yy + fwd[..., 1]
    valid = (fx_ >= 0) & (fx_ <= w - 1) & (fy_ >= 0) & (fy_ <= h - 1)
    dx = np.clip(np.round(fx_).astype(int), 0, w - 1)
    dy = np.clip(np.round(fy_).astype(int), 0, h - 1)
    landed = bwd[dy, dx]
    return np.linalg.norm(fwd + landed, axis=-1), valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="flow-test")
    ap.add_argument("--no-run", action="store_true", help="reuse existing run dir")
    ap.add_argument("--fb-tol", type=float, default=1.0,
                    help="px ceiling on the (robust) median in-bounds FB residual")
    ap.add_argument("--fb-report", type=float, default=2.0,
                    help="px threshold for the reported in-bounds frac (informational)")
    args = ap.parse_args()

    gen = MapGen.from_name("kitti-seq04")
    run = gen.layout.run(args.run_id)
    if not args.no_run:
        gen.run(only=["flow"], force=True, run_id=args.run_id)

    out = run.step_out("flow")
    manifest = json.loads((out / "manifest.json").read_text())
    npz = sorted(out.glob("flow_*.npz"))
    n_pairs = manifest["n_pairs"]

    print(f"\nflow sanity  (run={args.run_id})")
    print(f"  manifest: {manifest['n_frames']} frames → {n_pairs} pairs "
          f"{manifest['w']}x{manifest['h']}  backend={manifest['backend']} dtype={manifest['dtype']}")
    print(f"  npz on disk: {len(npz)}  (expected {n_pairs})")

    ok = len(npz) == n_pairs == manifest["n_frames"] - 1

    # inspect a representative mid-sequence pair
    mid = n_pairs // 2
    d = np.load(out / f"flow_{mid:06d}.npz")
    fwd, bwd = d["fwd"].astype(np.float32), d["bwd"].astype(np.float32)
    shape_ok = fwd.shape == (manifest["h"], manifest["w"], 2) and d["fwd"].dtype == np.float16
    mag = np.linalg.norm(fwd, axis=-1)
    fb, valid = fb_residual(fwd, bwd)
    fb_v = fb[valid]
    fb_med = float(np.median(fb_v))
    fb_frac = float((fb_v < args.fb_report).mean())  # informational (scene-dependent)
    print(f"  pair {mid}: fwd shape {fwd.shape} dtype {d['fwd'].dtype}  "
          f"|fwd| mean={mag.mean():.2f} max={mag.max():.1f}px")
    print(f"  FB residual (in-bounds {valid.mean():.0%} of px): median={fb_med:.3f}px  "
          f"(report frac<{args.fb_report}px={fb_frac:.0%}; tail is sky/foliage/movers)")

    mag_ok = 0.1 < mag.mean() < 200.0 and np.isfinite(mag).all()
    fb_ok = fb_med < args.fb_tol
    ok = ok and shape_ok and mag_ok and fb_ok
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'} "
          f"(count={len(npz) == n_pairs} shape={shape_ok} mag={mag_ok} fb={fb_ok})\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
