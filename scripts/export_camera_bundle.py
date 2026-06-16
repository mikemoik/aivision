"""Export a curated camera bundle: ego-motion + full intrinsics/extrinsics.

Wraps three things that previously lived apart into one tracked artifact at the
map root (``map/<map>/camera.json``), the companion to ``map_adas.json``:

  * ego-motion   — per-frame ``Twc`` (camera cam0 → world), from the map's own
    generation run's ``slam`` trajectory;
  * intrinsics   — the shared rectified ``K`` (KITTI P0..P3 share it);
  * extrinsics   — each camera's rectified pose relative to cam0 (pure x-baseline
    in the rectified rig) + the velodyne→cam0 transform (Tr).

COORDINATE FRAME: everything is in the SAME constant frame ``M`` as the ADAS map
— ``M`` = cam0 (image_0) at frame 0 (X-right, Y-down, Z-forward, metric), so
``poses_twc[0]`` is identity. The same georef (``T_M_to_ENU``) that places the
map on the Earth is copied in, so the bundle is self-contained for georeferencing
the trajectory too. Motion is taken from the run that produced the map
(``map_adas.json`` provenance.run_id) so it is consistent with the map objects.

    uv run python scripts/export_camera_bundle.py [--map kitti-seq04]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aivision.calibration import KittiCalibration
from aivision.map.layout import MapLayout
from aivision.map.map_adas import MapAdas

CAMERA_ROLES = {
    "image_0": ("gray_left", True),
    "image_1": ("gray_right", False),
    "image_2": ("color_left", False),
    "image_3": ("color_right", False),
}


def read_calib(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text().strip().splitlines():
        k, *vals = line.replace(":", "").split()
        out[k] = np.array(vals, float).reshape(3, 4)
    return out


def to4x4(m3x4: np.ndarray) -> list:
    T = np.eye(4)
    T[:3, :4] = m3x4
    return [[round(float(x), 9) for x in row] for row in T]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="kitti-seq04")
    ap.add_argument("--run-id", default=None, help="run for the trajectory (default: map's provenance run)")
    ap.add_argument("--out", default=None, help="output path (default: map/<map>/camera.json)")
    args = ap.parse_args()

    layout = MapLayout(args.map)
    m = MapAdas.load(layout.db)
    run_id = args.run_id or getattr(m.provenance, "run_id", "")
    if not run_id:
        raise SystemExit("map has no provenance.run_id; pass --run-id")
    run = layout.run(run_id)
    seq = layout.input / "odometry"
    out = Path(args.out) if args.out else layout.root / "camera.json"

    # --- motion: reframe the run's ORB cam0 trajectory (M) into f_veh_t0 -----
    # The SLAM trajectory is T_M_cam0(t) (cam0@frame0 world). We emit BOTH the
    # camera poses (T_fveh0_cam0, for projection/render) and the VEHICLE poses
    # (T_fveh0_fveh, the "ORB poses of the vehicle") in the f_veh_t0 world frame.
    cal = KittiCalibration.load()
    traj = json.loads((run.step_out("slam") / "trajectory.json").read_text())
    rows = np.asarray(traj["poses"], float)
    T_M_cam0 = np.tile(np.eye(4), (len(rows), 1, 1))
    T_M_cam0[:, :3, :4] = rows.reshape(-1, 3, 4)
    poses_cam = [[round(float(x), 6) for x in cal.cam0_pose_to_fveh0(P)[:3, :4].ravel()]
                 for P in T_M_cam0]
    poses_veh = [[round(float(x), 6) for x in cal.cam0_pose_to_vehicle(P)[:3, :4].ravel()]
                 for P in T_M_cam0]

    # --- calibration: shared K + per-camera rectified extrinsics + velo→cam0 -
    calib = read_calib(seq / "calib.txt")
    K = calib["P0"][:3, :3]
    Kinv = np.linalg.inv(K)
    fx = float(K[0, 0])
    img0 = cv2.imread(str(sorted((seq / "image_0").glob("*.png"))[0]))
    h, w = img0.shape[:2]

    cameras = {}
    for cam, (role, is_ref) in CAMERA_ROLES.items():
        P = calib.get(f"P{cam[-1]}")
        if P is None:
            continue
        t = Kinv @ P[:, 3]                 # cam0->cam_i translation (rectified: R=I)
        T_cam0_cam = np.eye(4)
        T_cam0_cam[:3, 3] = t
        cameras[cam] = {
            "role": role,
            "is_reference": is_ref,
            "K": [[round(float(x), 6) for x in r] for r in K],
            "distortion": "rectified (none)",
            "T_cam0_cam": [[round(float(x), 9) for x in r] for r in T_cam0_cam],
            "baseline_m_x": round(float(-P[0, 3] / fx), 6),
        }

    bundle = {
        "schema": "camera_bundle/v1",
        "map": args.map,
        "frame": {
            **m.frame.to_dict(),
            "axes": "X-forward, Y-left, Z-up",
            "origin": "f_veh at frame 0 (vehicle centreline); poses_vehicle[0] = identity",
            "note": "SAME constant frame f_veh_t0 as map_adas.json (the ADAS map objects); "
                    "sensor extrinsics/intrinsics in data/kitti/calibration_fveh.json",
        },
        "georef": m.georef.to_dict() if m.georef else None,
        "reference_camera": "image_0",
        "image_size": {"width": w, "height": h},
        "cameras": cameras,
        "velo_to_cam0": to4x4(calib["Tr"]) if "Tr" in calib else None,
        "motion": {
            "convention": "3x4 row-major [R|t], <body>->world(f_veh_t0), metric metres; "
                          "poses_vehicle = f_veh body pose (T_fveh0_fveh), "
                          "poses_twc = cam0 pose (T_fveh0_cam0)",
            "n_frames": int(traj.get("n_frames", len(poses_veh))),
            "frame_indexing": "i aligned with KITTI image index (image_0/frame_{i:06d}.png)",
            "poses_vehicle": poses_veh,
            "poses_twc": poses_cam,
        },
        "provenance": {
            "run_id": run_id,
            "sources": ["map-gen/slam/out/trajectory.json", "bindata/input/odometry/calib.txt"],
            "note": "ego-motion from the map's generation run; calib from the KITTI input slice",
        },
    }

    out.write_text(json.dumps(bundle, indent=2))
    print(f"[camera] wrote {out}")
    print(f"  frame {m.frame.name} (== map frame), {bundle['motion']['n_frames']} poses "
          f"(vehicle + cam0), {len(cameras)} cameras, {w}x{h}, "
          f"georef={'yes' if m.georef else 'no'}, run={run_id}")


if __name__ == "__main__":
    main()
