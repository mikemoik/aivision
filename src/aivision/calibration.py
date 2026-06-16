"""Single-source KITTI calibration loader.

Reads ``data/kitti/calibration_fveh.json`` — the canonical calibration document
(see ``scripts/build_kitti_calibration.py``) in which every sensor extrinsic is
expressed relative to ``f_veh`` (the centreline vehicle frame) and all camera
intrinsics (raw + rectified) are stored. Any code that needs KITTI extrinsics or
intrinsics should go through here rather than re-deriving from the raw calib
files, so there is one source of truth.

Frames: ``f_veh`` (x-fwd, y-left, z-up, on the lateral centreline), ``cam0..cam3``
(x-right, y-down, z-fwd, rectified), ``velodyne`` / ``imu`` (x-fwd, y-left, z-up).

Usage::

    cal = KittiCalibration.load()
    T = cal.transform("velodyne", "cam0")      # maps a velodyne point into cam0
    p_cam0 = cal.origin_in("fveh", "cam0")     # f_veh origin in cam0 coords
    K = cal.K("cam0")                          # rectified intrinsics
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path("data/kitti/calibration_fveh.json")

# accept a few friendly aliases for sensor names
_ALIASES = {"fveh": "f_veh", "vehicle": "f_veh", "gps": "imu", "velo": "velodyne"}


@dataclass
class KittiCalibration:
    """In-memory view over the single-source calibration JSON."""

    data: dict

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "KittiCalibration":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — generate it with "
                "`uv run python scripts/build_kitti_calibration.py`")
        return cls(json.loads(path.read_text()))

    # -- extrinsics -----------------------------------------------------
    def _key(self, name: str) -> str:
        return _ALIASES.get(name, name)

    def T_fveh(self, sensor: str) -> np.ndarray:
        """``T_fveh_<sensor>`` (4x4): maps a point in ``sensor`` coords to f_veh."""
        return np.array(self.data["extrinsics"][self._key(sensor)], float)

    def transform(self, src: str, dst: str) -> np.ndarray:
        """``T_dst_src`` (4x4): maps a point in ``src`` coords to ``dst`` coords."""
        return np.linalg.inv(self.T_fveh(dst)) @ self.T_fveh(src)

    def origin_in(self, sensor: str, frame: str) -> np.ndarray:
        """Origin of the ``sensor`` frame expressed in ``frame`` coords (3,)."""
        return self.transform(sensor, frame)[:3, 3]

    @property
    def sensors(self) -> list[str]:
        return list(self.data["extrinsics"])

    # -- f_veh_t0 world-frame reframing ---------------------------------
    # The map frame M is cam0 at frame 0 (x-right, y-down, z-fwd). f_veh_t0 is the
    # vehicle frame f_veh at frame 0 (x-fwd, y-left, z-up). Because both are "at
    # frame 0", the static cam0->f_veh extrinsic IS the M->f_veh_t0 reframe.
    def T_fveh0_M(self) -> np.ndarray:
        """``T_fveh0_M`` (4x4): maps a point in M (cam0@frame0) into f_veh_t0."""
        return self.T_fveh("cam0")          # static cam0 -> f_veh, evaluated at t0

    def cam0_pose_to_fveh0(self, T_M_cam0: np.ndarray) -> np.ndarray:
        """ORB camera pose ``T_M_cam0(t)`` -> camera pose in f_veh_t0 (T_fveh0_cam0)."""
        return self.T_fveh0_M() @ np.asarray(T_M_cam0, float)

    def cam0_pose_to_vehicle(self, T_M_cam0: np.ndarray) -> np.ndarray:
        """ORB camera pose ``T_M_cam0(t)`` -> VEHICLE (f_veh) pose in f_veh_t0.

        ``T_fveh0_fveh(t) = C · T_M_cam0(t) · C⁻¹`` with ``C = T_fveh_cam0``; the
        f_veh body pose in the f_veh_t0 world frame (identity at t=0).
        """
        C = self.T_fveh("cam0")
        return C @ np.asarray(T_M_cam0, float) @ np.linalg.inv(C)

    def vehicle_pose_to_cam0_M(self, T_fveh0_fveh: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`cam0_pose_to_vehicle`: f_veh pose in f_veh_t0 ->
        ORB camera pose ``T_M_cam0(t)`` (what the lift steps consume internally)."""
        C = self.T_fveh("cam0")
        return np.linalg.inv(C) @ np.asarray(T_fveh0_fveh, float) @ C

    # -- intrinsics -----------------------------------------------------
    def K(self, cam: str) -> np.ndarray:
        """Rectified 3x3 intrinsics of a camera."""
        return np.array(self.data["intrinsics"][self._key(cam)]["K"], float)

    def image_size(self, cam: str) -> tuple[int, int]:
        return tuple(self.data["intrinsics"][self._key(cam)]["image_size"])

    def baseline(self, cam: str) -> float:
        """Rectified baseline ``b_x`` of a camera wrt cam0 (m)."""
        return float(self.data["intrinsics"][self._key(cam)]["baseline_m"])

    @property
    def recording_date(self) -> str:
        return self.data.get("recording_date", "")

    def __repr__(self) -> str:
        return (f"KittiCalibration(date={self.recording_date!r}, "
                f"root={self.data.get('root_frame')!r}, sensors={self.sensors})")
