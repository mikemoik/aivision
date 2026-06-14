"""Map frame definition and georeferencing.

The ADAS map lives in a constant Euclidean SE(3) frame ``M`` whose origin is
arbitrary but fixed at map-creation time (for seq04: the camera pose of the
first frame, i.e. the ORB-SLAM3 world frame). :class:`FrameInfo` records the
provenance; the frame never moves once set.

Georeferencing is a stored rigid transform ``T_M_to_ENU`` plus a datum anchor
and a measured ``enu_offset`` (the along-track bias correction added to every
static-scene coordinate — see [[reference_seq04_gnss_offset]]). Both ORB-stereo
and GNSS are metric, so a rigid SE(3) is correct. ENU uses a spherical
equirectangular projection about the datum (matching the OXTS loader), which is
exact enough over a 1-2 km map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray

_R_EARTH = 6_378_137.0  # spherical Earth radius [m] (matches OXTS->ENU forward)


@dataclass
class FrameInfo:
    """Metadata for the constant map frame ``M`` (units always metres)."""

    name: str = "M"
    provenance: str = ""  # e.g. "ORB-SLAM3 seq04 Twc(frame0)=I"
    units: str = "m"
    created: str = ""  # ISO timestamp, stamped by the caller

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provenance": self.provenance,
            "units": self.units,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameInfo":
        return cls(**d)


@dataclass
class GeoReference:
    """Rigid map->ENU transform with a WGS84 datum anchor.

    ``T_M_to_ENU`` maps a point in map frame ``M`` to local ENU metres about
    ``datum = (lat0, lon0, alt0)``.
    """

    T_M_to_ENU: Array  # (4, 4)
    datum: tuple[float, float, float]  # (lat0_deg, lon0_deg, alt0_m)
    enu_offset: Array = field(default_factory=lambda: np.zeros(3))  # (3,) E,N,U bias

    def __post_init__(self) -> None:
        self.T_M_to_ENU = np.asarray(self.T_M_to_ENU, dtype=float).reshape(4, 4)
        self.enu_offset = np.asarray(self.enu_offset, dtype=float).reshape(3)

    @classmethod
    def from_correspondence(
        cls,
        pose_M: Array,
        pose_ENU: Array,
        datum: tuple[float, float, float],
    ) -> "GeoReference":
        """Solve ``T_M_to_ENU`` from one full pose correspondence.

        ``T_M_to_ENU = pose_ENU @ inv(pose_M)`` — one rigid correspondence is
        enough because both frames are metric.
        """
        pose_M = np.asarray(pose_M, dtype=float).reshape(4, 4)
        pose_ENU = np.asarray(pose_ENU, dtype=float).reshape(4, 4)
        return cls(T_M_to_ENU=pose_ENU @ np.linalg.inv(pose_M), datum=datum)

    @classmethod
    def from_point_correspondences(
        cls,
        src_M: Array,
        dst_ENU: Array,
        datum: tuple[float, float, float],
        with_scale: bool = False,
        enu_offset: Array | None = None,
    ) -> "GeoReference":
        """Rigid (Umeyama) fit of ``T_M_to_ENU`` from matched point sets.

        More robust than a single pose pair: uses the whole trajectory so the
        camera<->GNSS lever arm averages out. ``with_scale=False`` enforces a
        true rigid transform (both frames are metric); set True only to audit
        the residual scale.
        """
        src = np.asarray(src_M, dtype=float)
        dst = np.asarray(dst_ENU, dtype=float)
        mu_s, mu_d = src.mean(0), dst.mean(0)
        S, D = src - mu_s, dst - mu_d
        H = S.T @ D / len(src)
        U, sv, Vt = np.linalg.svd(H)
        W = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            W[2, 2] = -1
        R = Vt.T @ W @ U.T
        s = (sv * np.diag(W)).sum() / (S**2).sum() * len(src) if with_scale else 1.0
        t = mu_d - s * R @ mu_s
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = t
        return cls(T_M_to_ENU=T, datum=datum,
                   enu_offset=np.zeros(3) if enu_offset is None else enu_offset)

    def to_enu(self, p_M: Array) -> Array:
        """Map a point (or Nx3 array) from frame ``M`` to local ENU metres.

        Includes the stored ``enu_offset`` bias correction.
        """
        p = np.atleast_2d(np.asarray(p_M, dtype=float))
        out = (self.T_M_to_ENU[:3, :3] @ p.T).T + self.T_M_to_ENU[:3, 3] + self.enu_offset
        return out.reshape(np.shape(p_M))

    def to_wgs84(self, p_M: Array) -> Array:
        """Map a point from frame ``M`` to (lat_deg, lon_deg, alt_m).

        Spherical equirectangular inverse about the datum — the exact inverse
        of the OXTS->ENU forward projection used to fit the transform.
        """
        lat0, lon0, alt0 = self.datum
        enu = np.atleast_2d(self.to_enu(p_M))
        lat = lat0 + np.degrees(enu[:, 1] / _R_EARTH)
        lon = lon0 + np.degrees(enu[:, 0] / (_R_EARTH * np.cos(np.radians(lat0))))
        alt = alt0 + enu[:, 2]
        out = np.stack([lat, lon, alt], axis=1)
        return out.reshape(np.shape(p_M))

    def to_dict(self) -> dict:
        return {
            "T_M_to_ENU": self.T_M_to_ENU.tolist(),
            "datum": list(self.datum),
            "enu_offset": self.enu_offset.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GeoReference":
        return cls(T_M_to_ENU=d["T_M_to_ENU"], datum=tuple(d["datum"]),
                   enu_offset=d.get("enu_offset", np.zeros(3)))
