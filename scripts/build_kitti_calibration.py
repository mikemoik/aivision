"""Build the single-source KITTI calibration JSON, all extrinsics wrt f_veh.

This is the canonical calibration document for the KITTI rig: every sensor
extrinsic is expressed relative to ``f_veh`` (the centreline vehicle frame), and
all camera intrinsics (raw + rectified) are included. It is generated purely
from KITTI's own raw calibration files so it is reproducible:

    calib_cam_to_cam.txt   per-camera S/K/D/R/T + S_rect/R_rect/P_rect
    calib_velo_to_cam.txt  velodyne -> cam0 (unrectified)
    calib_imu_to_velo.txt  GPS-IMU  -> velodyne

Frames / conventions (KITTI, Geiger et al. 2013 IJRR):
    camera   : x-right, y-down,  z-forward (rectified)
    velodyne : x-forward, y-left, z-up
    imu/gps  : x-forward, y-left, z-up
    f_veh    : the imu frame, origin shifted (0, -0.32, 0) m in imu axes onto the
               car's lateral centreline; same axes (x-fwd, y-left, z-up).

Extrinsics are stored as ``T_fveh_<sensor>`` (4x4): ``p_fveh = T @ p_sensor``.
The rectified-camera relation follows the paper: a point in the rectified cam0
frame maps to camera i via ``P_rect_i``; in rectified coords all cameras share
cam0's orientation and differ only by the baseline ``b_x`` along x.

    uv run python scripts/build_kitti_calibration.py [--date 2011_09_30]
        [--out data/kitti/calibration_fveh.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# f_veh: shift the IMU origin by this (x-fwd, y-left, z-up) onto the centreline.
FVEH_IMU_SHIFT = np.array([0.0, -0.32, 0.0])

CAMERA_ROLES = {0: "gray_left", 1: "gray_right", 2: "color_left", 3: "color_right"}


def _parse_kv(path: Path) -> dict[str, np.ndarray]:
    """Parse a KITTI ``key: v v v ...`` calibration file into name -> float array."""
    out: dict[str, np.ndarray] = {}
    for ln in path.read_text().splitlines():
        if ":" not in ln or ln.startswith("calib_time"):
            continue
        k, v = ln.split(":", 1)
        try:
            out[k.strip()] = np.array([float(x) for x in v.split()])
        except ValueError:
            pass
    return out


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = np.asarray(R).reshape(3, 3)
    M[:3, 3] = np.asarray(t).reshape(3)
    return M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2011_09_30")
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="dir holding the three calib_*.txt (default data/kitti/raw/<date>)")
    ap.add_argument("--out", type=Path, default=Path("data/kitti/calibration_fveh.json"))
    args = ap.parse_args()

    cdir = args.calib_dir or Path(f"data/kitti/raw/{args.date}")
    c2c = _parse_kv(cdir / "calib_cam_to_cam.txt")
    v2c = _parse_kv(cdir / "calib_velo_to_cam.txt")
    i2v = _parse_kv(cdir / "calib_imu_to_velo.txt")

    # --- transform chain into the rectified cam0 frame (the ORB / map frame M) ---
    R_rect0 = c2c["R_rect_00"].reshape(3, 3)
    T_velo_cam0 = _T(R_rect0, np.zeros(3)) @ _T(v2c["R"], v2c["T"])   # velo -> rect cam0
    T_imu_cam0 = T_velo_cam0 @ _T(i2v["R"], i2v["T"])                 # imu  -> rect cam0
    T_fveh_in_cam0 = T_imu_cam0 @ _T(np.eye(3), FVEH_IMU_SHIFT)       # f_veh -> rect cam0
    T_cam0_fveh = np.linalg.inv(T_fveh_in_cam0)                       # rect cam0 -> f_veh

    # rectified camera i: centre [b_x_i, 0, 0] in rect cam0, orientation = identity
    def cam_in_cam0(i: int) -> np.ndarray:
        P = c2c[f"P_rect_0{i}"].reshape(3, 4)
        bx = -P[0, 3] / P[0, 0]
        return _T(np.eye(3), [bx, 0.0, 0.0])

    sensors_in_cam0 = {
        "cam0": cam_in_cam0(0), "cam1": cam_in_cam0(1),
        "cam2": cam_in_cam0(2), "cam3": cam_in_cam0(3),
        "velodyne": T_velo_cam0, "imu": T_imu_cam0,
    }
    extrinsics = {name: (T_cam0_fveh @ T).tolist()           # T_fveh_<sensor>
                  for name, T in sensors_in_cam0.items()}
    extrinsics["f_veh"] = np.eye(4).tolist()

    # --- intrinsics (raw + rectified) per camera --------------------------------
    intrinsics = {}
    for i in range(4):
        P = c2c[f"P_rect_0{i}"].reshape(3, 4)
        intrinsics[f"cam{i}"] = {
            "role": CAMERA_ROLES[i],
            "image_size": [int(c2c[f"S_rect_0{i}"][0]), int(c2c[f"S_rect_0{i}"][1])],
            "K": P[:3, :3].tolist(),                         # rectified intrinsics
            "baseline_m": float(-P[0, 3] / P[0, 0]),         # b_x wrt cam0 (rectified)
            "distortion": "rectified (none)",
            "raw": {
                "image_size": [int(c2c[f"S_0{i}"][0]), int(c2c[f"S_0{i}"][1])],
                "K": c2c[f"K_0{i}"].reshape(3, 3).tolist(),
                "distortion_model": "plumb_bob (k1 k2 p1 p2 k3)",
                "distortion": c2c[f"D_0{i}"].tolist(),
                "R_rect": c2c[f"R_rect_0{i}"].reshape(3, 3).tolist(),
                "P_rect": P.tolist(),
            },
        }

    doc = {
        "schema": "kitti_calibration_fveh/v1",
        "description": (
            "SINGLE SOURCE OF TRUTH for KITTI sensor calibration. Every extrinsic is "
            "expressed relative to f_veh (the centreline vehicle frame). Generated from "
            "KITTI's raw calib_*.txt by scripts/build_kitti_calibration.py — do not hand-edit."),
        "recording_date": args.date,
        "root_frame": "f_veh",
        "conventions": {
            "frames": {
                "f_veh": "imu frame shifted (0,-0.32,0) m in imu axes onto the lateral "
                         "centreline; x-forward, y-left, z-up",
                "camera": "x-right, y-down, z-forward (rectified)",
                "velodyne": "x-forward, y-left, z-up",
                "imu": "x-forward, y-left, z-up",
            },
            "extrinsics": "T_fveh_<sensor> is 4x4 homogeneous; p_fveh = T_fveh_<sensor> @ p_sensor",
            "units": "metres",
            "reference": "Geiger et al. 2013, 'Vision meets Robotics: The KITTI Dataset', IJRR",
        },
        "f_veh": {
            "definition": "GPS/IMU frame, origin translated (0,-0.32,0) m in imu axes (x-fwd,"
                          "y-left,z-up) onto the car lateral centreline",
            "imu_to_fveh_shift_m": FVEH_IMU_SHIFT.tolist(),
            "axes": "x-forward, y-left, z-up",
            "height_above_road_m": 0.93,
        },
        "heights_above_road_m": {"cameras": 1.65, "velodyne": 1.73, "imu": 0.93,
                                 "wheel_axes": 0.30},
        "vehicle": {"wheelbase_m": 2.71, "track_width_m": 1.60},
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "source_files": {
            "cam_to_cam": str(cdir / "calib_cam_to_cam.txt"),
            "velo_to_cam": str(cdir / "calib_velo_to_cam.txt"),
            "imu_to_velo": str(cdir / "calib_imu_to_velo.txt"),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))

    # --- report + sanity checks -------------------------------------------------
    print(f"wrote {args.out}")
    print("  extrinsics T_fveh_<sensor> origins (in f_veh, x-fwd/y-left/z-up):")
    for name in ("cam0", "cam1", "cam2", "cam3", "velodyne", "imu"):
        o = np.array(extrinsics[name])[:3, 3]
        print(f"    {name:9s} {np.round(o, 3)}")
    imu_o = np.array(extrinsics["imu"])[:3, 3]
    cam0_o = np.array(extrinsics["cam0"])[:3, 3]
    print(f"  check: imu lateral (y, +left) = {imu_o[1]:+.3f} m  (expect +0.32, IMU left of centre)")
    print(f"  check: cam0 lateral (y)       = {cam0_o[1]:+.3f} m  (expect ~0, on centreline)")


if __name__ == "__main__":
    main()
