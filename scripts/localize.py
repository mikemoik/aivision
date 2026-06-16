"""Batch-localize the ego camera in the georeferenced ADAS map.

Sensors: the virtual camera (per-frame static-object detections in camera
coords) + a NOISY 2D GNSS fix (clean OXTS position + seeded N(0, sigma^2)).
ORB odometry is deliberately excluded; frame coupling is the smoothed-trajectory
motion priors inside ``aivision.map.localize``.

Scores the estimate against the CLEAN OXTS track in ENU (ATE / RPE / heading)
and prints the raw-GNSS baseline for comparison.

    uv run python scripts/localize.py [--map kitti-seq04] [--run-id vcam-test]
        [--gnss-sigma 5.0] [--seed 0] [--em-rounds 3] [--out traj.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.ingest import load_oxts_fveh_enu
from aivision.map.layout import MapLayout
from aivision.map.localize import LocalizeConfig, localize, score
from aivision.map.map_adas import MapAdas
from aivision.map.virtual_camera import PixelNoise, VirtualCameraSensor

DEFAULT_OXTS = Path(
    "data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--run-id", default="vcam-test",
                    help="run carrying the *_by_frame.json virtual-sensor artifacts")
    ap.add_argument("--oxts", type=Path, default=DEFAULT_OXTS)
    ap.add_argument("--gnss-sigma", type=float, default=5.0)
    ap.add_argument("--gnss-bias-e", type=float, default=0.0,
                    help="constant GNSS bias along ENU East [m] (added to every fix)")
    ap.add_argument("--gnss-bias-n", type=float, default=0.0,
                    help="constant GNSS bias along ENU North [m] (added to every fix)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--em-rounds", type=int, default=4)
    ap.add_argument("--gate-m", type=float, default=3.0)
    ap.add_argument("--head-window", type=int, default=15)
    ap.add_argument("--pixel-noise", type=float, default=0.0,
                    help="image-sensor pixel noise [px, 1-sigma]; jitters every virtual-camera "
                         "detection's mean + covariance and widens the graph's bearing sigma to match")
    ap.add_argument("--pixel-noise-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional: write estimated ENU trajectory + report JSON")
    ap.add_argument("--apply-display-offset", action="store_true",
                    help="keep the georef enu_offset (map-display nudge) in the localization "
                         "path; default drops it so ATE is scored in the OXTS-ENU frame")
    args = ap.parse_args()

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    # georef.enu_offset is a MAP-DISPLAY nudge (KML/satellite); OXTS GT + GNSS + the estimate all
    # live in the un-nudged OXTS-ENU frame, so applying it injects a constant ~1.37 m N bias into
    # ATE. Drop it from the localization path (see scripts/localize_sw.py for the full rationale).
    if not args.apply_display_offset and amap.georef is not None:
        import numpy as _np
        amap.georef.enu_offset = _np.zeros(3)
    noise = (PixelNoise(sigma_px=args.pixel_noise, seed=args.pixel_noise_seed)
             if args.pixel_noise > 0 else None)
    sensor = VirtualCameraSensor.from_map(args.map, run_id=args.run_id, noise=noise)
    print(amap)
    print(sensor)
    if noise is not None:
        print(f"[sensor] pixel noise: sigma_px={noise.sigma_px} px (seed {noise.seed})")
    if amap.georef is None:
        raise SystemExit("map has no georef — cannot localize")

    n = sensor.n_frames
    clean_enu, datum = load_oxts_fveh_enu(args.oxts, n)   # f_veh (vehicle) GNSS track
    if not np.allclose(datum, amap.georef.datum, atol=1e-6):
        print(f"WARN: oxts datum {datum} != georef datum {amap.georef.datum}")

    # noisy GNSS = clean horizontal position + seeded Gaussian noise + constant bias
    rng = np.random.default_rng(args.seed)
    gnss_enu = clean_enu.copy()
    gnss_enu[:, :2] += rng.normal(0.0, args.gnss_sigma, size=(n, 2))
    if args.gnss_bias_e or args.gnss_bias_n:
        gnss_enu[:, 0] += args.gnss_bias_e
        gnss_enu[:, 1] += args.gnss_bias_n
        print(f"[gnss] constant bias dE {args.gnss_bias_e:+.2f} dN {args.gnss_bias_n:+.2f} m "
              f"(|{np.hypot(args.gnss_bias_e, args.gnss_bias_n):.2f}| m)")

    cfg = LocalizeConfig(gnss_sigma=args.gnss_sigma, em_rounds=args.em_rounds,
                         gate_m=args.gate_m, head_window=args.head_window)
    if args.pixel_noise > 0:                       # graph's bearing sigma tracks the sensor
        cfg.sigma_bearing = args.pixel_noise / float(sensor.K[0, 0])  # px -> rad
        cfg.gate_m = max(cfg.gate_m, 4.0)          # widen gate for the noisier detections
    res = localize(amap, sensor, gnss_enu, cfg)
    print(f"point assoc per EM round: {res.n_assoc}")
    print(f"lane  assoc per EM round: {res.n_lane_assoc}")

    # --- scoring -----------------------------------------------------------
    fused = score(res.centers_M, amap.georef, clean_enu, res.poses_M)

    # baseline: the raw noisy GNSS fix as the trajectory estimate (ENU direct)
    base_enu = gnss_enu.copy()
    base_enu[:, 2] = clean_enu[:, 2]            # height free -> use clean for a fair 2D/3D
    d0 = base_enu - clean_enu
    baseline = {
        "ate_2d": float(np.sqrt(np.mean(np.sum(d0[:, :2] ** 2, axis=1)))),
        "ate_2d_max": float(np.max(np.linalg.norm(d0[:, :2], axis=1))),
    }

    print("\n=== localization vs clean OXTS (ENU) ===")
    print(f"  frames                : {fused['n']}")
    print(f"  ATE 2D (E/N)   rms/max: {fused['ate_2d']:.3f} / {fused['ate_2d_max']:.3f} m")
    print(f"  ATE 3D         rms    : {fused['ate_3d']:.3f} m")
    print(f"  RPE (d={fused['rpe_delta']:>2}) trans  : {fused['rpe_t']:.3f} m")
    print(f"  heading mean/med      : {fused['heading_deg_mean']:.2f} / "
          f"{fused['heading_deg_med']:.2f} deg")
    print(f"  baseline raw-GNSS ATE : {baseline['ate_2d']:.3f} / "
          f"{baseline['ate_2d_max']:.3f} m  (rms/max, 2D)")
    gain = baseline["ate_2d"] / fused["ate_2d"] if fused["ate_2d"] else float("nan")
    print(f"  improvement over GNSS : {gain:.2f}x")

    if args.out:
        from aivision.map.localize import R_world_from_ENU
        est_enu = amap.georef.to_enu(res.centers_M)
        # estimated vehicle forward (f_veh +X) per frame, as a unit vector in ENU
        R_ENU_world = np.linalg.inv(R_world_from_ENU(amap.georef))
        fwd = np.array([R_ENU_world @ p.rotation().matrix()[:, 0] for p in res.poses_M])
        fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "map": args.map, "run_id": args.run_id, "seed": args.seed,
            "gnss_sigma": args.gnss_sigma,
            "report": fused, "baseline": baseline,
            "est_enu": est_enu.tolist(), "est_fwd_enu": fwd.tolist(),
            "clean_enu": clean_enu.tolist(), "gnss_enu": gnss_enu.tolist(),
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
