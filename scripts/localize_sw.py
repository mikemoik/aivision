"""Online sliding-window localization (Wilbers et al. 2019) vs the batch baseline.

Runs the sliding-window factor-graph localizer (truncation window, map-prior
landmarks, odometry between-factors, GNSS for init only) causally over the run and
scores it against the clean OXTS track in ENU. Odometry is derived from the
reconstructed vehicle poses (``camera.json``) with added drift noise, since the
dataset has no wheel encoder. Prints the batch smoother's numbers for comparison.

    uv run python scripts/localize_sw.py [--window 30] [--gnss-mode init|continuous]
        [--odom-sigma-t 0.05] [--map-sigma 0.05] [--out traj.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aivision.map.ingest import load_oxts_fveh_enu
from aivision.map.layout import MapLayout
from aivision.map.localize import LocalizeConfig, R_world_from_ENU, localize, score
from aivision.map.localize_sw import SWConfig, localize_sliding_window, odometry_from_poses
from aivision.map.map_adas import MapAdas
from aivision.map.virtual_camera import VirtualCameraSensor

DEFAULT_OXTS = Path("data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data")


def _poses_world(layout: MapLayout, n: int) -> np.ndarray:
    """T_world_fveh(t) (n,4,4) from the camera bundle's vehicle poses (f_veh_t0)."""
    bundle = json.loads((layout.root / "camera.json").read_text())
    P = np.array(bundle["motion"]["poses_vehicle"]).reshape(-1, 3, 4)[:n]
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :4] = P
    return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--run-id", default="vcam-test")
    ap.add_argument("--oxts", type=Path, default=DEFAULT_OXTS)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--gnss-sigma", type=float, default=5.0)
    ap.add_argument("--gnss-mode", choices=("init", "continuous"), default="init")
    ap.add_argument("--init-frames", type=int, default=12)
    ap.add_argument("--map-sigma", type=float, default=0.05)
    ap.add_argument("--odom-sigma-t", type=float, default=0.05)
    ap.add_argument("--odom-sigma-r", type=float, default=0.3)
    ap.add_argument("--no-lanes", action="store_true")
    ap.add_argument("--pixel-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-batch", action="store_true", help="skip the batch comparison")
    ap.add_argument("--apply-display-offset", action="store_true",
                    help="keep the georef enu_offset (map-display nudge) in the localization "
                         "path; default drops it so ATE is scored in the OXTS-ENU frame")
    args = ap.parse_args()

    layout = MapLayout(args.map)
    amap = MapAdas.load(layout.db)
    if amap.georef is None:
        raise SystemExit("map has no georef")
    # `georef.enu_offset` (SEQ04_ENU_OFFSET = [0.20, 1.37, 0]) is a MAP-DISPLAY nudge for the
    # static-scene KML/satellite overlay. OXTS GT, the GNSS measurements, and the localization
    # estimate all live in the un-nudged OXTS-ENU frame, so feeding it through `to_enu`/
    # `enu_to_world` injects a constant ~1.37 m N bias (est_enu = truth + offset, scored vs an
    # offset-free GT). Drop it from the localization path so ATE measures in-map accuracy.
    if not args.apply_display_offset:
        amap.georef.enu_offset = np.zeros(3)
    from aivision.map.virtual_camera import PixelNoise
    noise = PixelNoise(sigma_px=args.pixel_noise, seed=args.seed) if args.pixel_noise > 0 else None
    sensor = VirtualCameraSensor.from_map(args.map, run_id=args.run_id, noise=noise)
    n = sensor.n_frames

    clean_enu, _ = load_oxts_fveh_enu(args.oxts, n)
    rng = np.random.default_rng(args.seed)
    gnss_enu = clean_enu.copy()
    gnss_enu[:, :2] += rng.normal(0.0, args.gnss_sigma, size=(n, 2))

    poses_world = _poses_world(layout, n)
    odom = odometry_from_poses(poses_world, args.odom_sigma_t, args.odom_sigma_r, seed=args.seed)

    cfg = SWConfig(window=args.window, gnss_sigma=args.gnss_sigma, gnss_mode=args.gnss_mode,
                   init_frames=args.init_frames, map_sigma=args.map_sigma,
                   odom_sigma_t=args.odom_sigma_t, odom_sigma_r=args.odom_sigma_r,
                   use_lanes=not args.no_lanes)
    if args.pixel_noise > 0:
        cfg.sigma_bearing = args.pixel_noise / float(sensor.K[0, 0])
        cfg.gate_m = max(cfg.gate_m, 4.0)

    res = localize_sliding_window(amap, sensor, gnss_enu, odom, cfg)
    sw = score(res.online_centers, amap.georef, clean_enu, res.online_poses)

    print(f"\n=== sliding-window (online, window={args.window}, gnss={args.gnss_mode}) "
          f"vs clean OXTS ===")
    print(f"  frames                : {sw['n']}")
    print(f"  ATE 2D (E/N)   rms/max: {sw['ate_2d']:.3f} / {sw['ate_2d_max']:.3f} m")
    print(f"  ATE 3D         rms    : {sw['ate_3d']:.3f} m")
    print(f"  RPE (d={sw['rpe_delta']:>2}) trans  : {sw['rpe_t']:.3f} m")
    print(f"  heading mean/med      : {sw['heading_deg_mean']:.2f} / {sw['heading_deg_med']:.2f} deg")
    print(f"  mean point/lane assoc : {np.mean(res.n_assoc):.1f} / {np.mean(res.n_lane):.1f} per frame")

    batch = None
    if not args.no_batch:
        bcfg = LocalizeConfig(gnss_sigma=args.gnss_sigma)
        if args.pixel_noise > 0:
            bcfg.sigma_bearing = cfg.sigma_bearing; bcfg.gate_m = cfg.gate_m
        bres = localize(amap, sensor, gnss_enu, bcfg)
        batch = score(bres.centers_M, amap.georef, clean_enu, bres.poses_M)
        print("\n=== batch smoother (offline, whole track) vs clean OXTS ===")
        print(f"  ATE 2D rms/max        : {batch['ate_2d']:.3f} / {batch['ate_2d_max']:.3f} m")
        print(f"  RPE (d={batch['rpe_delta']:>2}) trans  : {batch['rpe_t']:.3f} m")
        print(f"  heading med           : {batch['heading_deg_med']:.2f} deg")

    if args.out:
        est_enu = amap.georef.to_enu(res.online_centers)
        R_ENU_world = np.linalg.inv(R_world_from_ENU(amap.georef))
        fwd = np.array([R_ENU_world @ p.rotation().matrix()[:, 0] for p in res.online_poses])
        fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "map": args.map, "method": "sliding_window", "window": args.window,
            "gnss_mode": args.gnss_mode, "report": sw, "batch": batch,
            "est_enu": est_enu.tolist(), "est_fwd_enu": fwd.tolist(),
            "clean_enu": clean_enu.tolist(), "gnss_enu": gnss_enu.tolist(),
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
