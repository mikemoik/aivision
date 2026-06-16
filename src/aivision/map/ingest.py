"""Format adapters: external detector artifacts -> ADAS map objects.

This module is the ETL layer. It knows about the messy, per-detector file
schemas (``poles_signs_world.json``, ``arrows_world.json``,
``lane_polylines.json``, ...) so :class:`~aivision.map.map_adas.MapAdas`
does not have to. Each adapter parses one source and returns a list of map
objects (or builds a whole map); the caller feeds them in via ``map.add()``.

All seq04 sources are already in the constant map frame ``M`` (the ORB-SLAM3
world frame == cam0 of frame 0: X-right, Y-down, Z-forward, metres), so no
reprojection is needed here. "Up" in this frame is ``-Y``.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np

from .frame import FrameInfo, GeoReference
from .geometry import GaussianPoint, GaussianPose, LineSegment, OrientedBox, Polyline
from .map_adas import MapAdas
from .objects import GuidePost, LaneMarking, Pole, RoadArrow, TrafficLight, TrafficSign

_R_EARTH = 6_378_137.0  # WGS84 equatorial radius [m]

UP = np.array([0.0, -1.0, 0.0])  # world up in the map (cam) frame

# Defaults when a source gives no per-object uncertainty.
_DEFAULT_SIGMA_M = 0.10  # positional stddev [m]
_UNKNOWN_ROT_VAR = float(np.pi**2)  # rad^2, for unknown orientation


def _iso_cov(sigma: float) -> np.ndarray:
    return np.eye(3) * float(sigma) ** 2


def _pose_cov(sigma_t: float, var_rot: float) -> np.ndarray:
    cov = np.eye(6)
    cov[:3, :3] = _iso_cov(sigma_t)
    cov[3:, 3:] = np.eye(3) * var_rot
    return cov


def _rot_from_forward(forward: np.ndarray) -> np.ndarray:
    """Right-handed R with columns [right, down, forward], up fixed to -Y."""
    f = np.asarray(forward, dtype=float)
    f = f / np.linalg.norm(f)
    right = np.cross(UP, f)
    n = np.linalg.norm(right)
    if n < 1e-6:  # forward ~parallel to up; fall back to identity
        return np.eye(3)
    right /= n
    down = np.cross(f, right)
    return np.column_stack([right, down, f])


# -- shared object builders ---------------------------------------------
# One place that fixes each class's geometry SLOTS, so the AGGREGATED map
# (load_poles_signs, frame M) and the per-frame VIRTUAL CAMERA SENSOR
# (steps_objects, camera frame) build byte-identical structures from the same
# code. The frame is whatever ``center`` is expressed in; "up" is -Y in both the
# map frame M and the camera frame (X-right, Y-down, Z-forward).

def make_pole(obj_id, center, height, sigma=_DEFAULT_SIGMA_M, *,
              n_obs=1, created_frame=None, attributes=None) -> Pole:
    c = np.asarray(center, dtype=float)
    half = 0.5 * float(height or 0.0)
    cov = _iso_cov(sigma)
    shaft = LineSegment(GaussianPoint(c + half * UP, cov), GaussianPoint(c - half * UP, cov))
    return Pole(obj_id, {"shaft": shaft}, attributes={"height": height, **(attributes or {})},
                n_obs=n_obs, created_frame=created_frame, last_update_frame=created_frame)


def make_sign(obj_id, center, size, sigma=_DEFAULT_SIGMA_M, *,
              n_obs=1, created_frame=None, attributes=None) -> TrafficSign:
    c = np.asarray(center, dtype=float)
    half = 0.5 * float(size or 0.0)
    mount = GaussianPose.from_Rt(np.eye(3), c, _pose_cov(sigma, _UNKNOWN_ROT_VAR))
    face = OrientedBox(GaussianPose.from_Rt(np.eye(3), c, _pose_cov(sigma, _UNKNOWN_ROT_VAR)),
                       extents=[half, half, 0.01])
    return TrafficSign(obj_id, {"mount": mount, "face": face},
                       attributes=attributes or {}, n_obs=n_obs,
                       created_frame=created_frame, last_update_frame=created_frame)


def make_traffic_light(obj_id, center, height, sigma=_DEFAULT_SIGMA_M, *,
                       n_obs=1, created_frame=None, attributes=None) -> TrafficLight:
    c = np.asarray(center, dtype=float)
    h = float(height if height is not None else 0.3)
    mount = GaussianPose.from_Rt(np.eye(3), c, _pose_cov(sigma, _UNKNOWN_ROT_VAR))
    box = OrientedBox(GaussianPose.from_Rt(np.eye(3), c, _pose_cov(sigma, _UNKNOWN_ROT_VAR)),
                      extents=[0.15, 0.5 * h, 0.15])
    return TrafficLight(obj_id, {"mount": mount, "box": box},
                        attributes={"height": height, **(attributes or {})}, n_obs=n_obs,
                        created_frame=created_frame, last_update_frame=created_frame)


def make_guidepost(obj_id, center, height, sigma=_DEFAULT_SIGMA_M, *,
                   n_obs=1, created_frame=None, attributes=None) -> GuidePost:
    c = np.asarray(center, dtype=float)
    return GuidePost(obj_id, {"point": GaussianPoint(c, _iso_cov(sigma))},
                     attributes={"height": height, **(attributes or {})}, n_obs=n_obs,
                     created_frame=created_frame, last_update_frame=created_frame)


def make_arrow(obj_id, centroid, forward, glyph, sigma=_DEFAULT_SIGMA_M, *,
               yaw_var_deg: float = 5.0, n_obs=1, created_frame=None,
               attributes=None) -> RoadArrow:
    c = np.asarray(centroid, dtype=float)
    R = _rot_from_forward(forward)
    pose = GaussianPose.from_Rt(R, c, _pose_cov(sigma, np.radians(yaw_var_deg) ** 2))
    gl = Polyline([GaussianPoint(np.asarray(v, dtype=float), _iso_cov(sigma)) for v in glyph],
                  closed=True)
    return RoadArrow(obj_id, {"pose": pose, "glyph": gl}, attributes=attributes or {},
                     n_obs=n_obs, created_frame=created_frame, last_update_frame=created_frame)


def make_lane_marking(obj_id, side, verts, sigma=_DEFAULT_SIGMA_M, *,
                      n_obs=1, created_frame=None, method=None, attributes=None) -> LaneMarking:
    line = Polyline([GaussianPoint(np.asarray(p, dtype=float), _iso_cov(sigma)) for p in verts])
    attrs = {"side": side}
    if method is not None:
        attrs["method"] = method
    if attributes:
        attrs.update(attributes)
    return LaneMarking(obj_id, {"line": line}, attributes=attrs, n_obs=n_obs,
                       created_frame=created_frame, last_update_frame=created_frame)


# -- individual adapters ------------------------------------------------
def load_poles_signs(path: str | Path, default_sigma: float = _DEFAULT_SIGMA_M) -> list:
    """Parse ``poles_signs_world.json`` -> Pole/TrafficSign/TrafficLight/GuidePost."""
    d = json.loads(Path(path).read_text())
    objs: list = []

    for i, p in enumerate(d.get("poles", [])):
        objs.append(make_pole(f"pole_{i}", p["center"], p.get("height"),
                              p.get("res_m", default_sigma), n_obs=p.get("n_frames", 1)))

    for i, s in enumerate(d.get("signs", [])):
        objs.append(make_sign(f"sign_{i}", s["center"], s.get("size"),
                              s.get("res_m", default_sigma), n_obs=s.get("n_frames", 1),
                              attributes={"name": s.get("name"), "agree": s.get("agree")}))

    for i, t in enumerate(d.get("traffic_lights", [])):
        objs.append(make_traffic_light(f"tl_{i}", t["center"], t.get("height"),
                                       t.get("res_m", default_sigma),
                                       n_obs=t.get("n_frames", 1)))

    for i, g in enumerate(d.get("guideposts", [])):
        objs.append(make_guidepost(f"gp_{i}", g["center"], g.get("height"),
                                   g.get("res_m", default_sigma), n_obs=g.get("n_frames", 1)))
    return objs


def load_arrows(path: str | Path, default_sigma: float = _DEFAULT_SIGMA_M) -> list[RoadArrow]:
    """Parse ``arrows_world.json`` -> RoadArrow (pose from forward, glyph polygon)."""
    arrows = json.loads(Path(path).read_text())
    objs: list[RoadArrow] = []
    for i, a in enumerate(arrows):
        objs.append(make_arrow(f"arrow_{i}", a["centroid"], a["forward"],
                               a.get("glyph_world", []), default_sigma,
                               attributes={"cls": a.get("cls"), "length": a.get("length"),
                                           "width": a.get("width"),
                                           "yaw_to_lane_deg": a.get("yaw_to_lane_deg")},
                               n_obs=a.get("n_frames", 1)))
    return objs


def load_lane_polylines(path: str | Path, default_sigma: float = _DEFAULT_SIGMA_M) -> list[LaneMarking]:
    """Parse ``lane_polylines.json`` -> left/right LaneMarking polylines."""
    d = json.loads(Path(path).read_text())
    objs: list[LaneMarking] = []
    for side in ("left", "right"):
        pts = d.get(side)
        if not pts:
            continue
        objs.append(make_lane_marking(f"lane_{side}", side, pts, default_sigma,
                                      method=d.get("method")))
    return objs


# -- georeferencing -----------------------------------------------------
def load_oxts_enu(oxts_dir: str | Path, n: int) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load KITTI OXTS samples 0..n-1 -> (ENU metres (n,3), datum lat/lon/alt).

    Each ``{i:010d}.txt`` starts with lat[deg] lon[deg] alt[m]. ENU is an
    equirectangular projection about sample 0 (datum) — exact enough at
    map scale.
    """
    oxts_dir = Path(oxts_dir)
    ll = np.array([np.loadtxt(oxts_dir / f"{i:010d}.txt")[:3] for i in range(n)])
    lat0, lon0, alt0 = ll[0]
    east = np.radians(ll[:, 1] - lon0) * _R_EARTH * math.cos(math.radians(lat0))
    north = np.radians(ll[:, 0] - lat0) * _R_EARTH
    enu = np.stack([east, north, ll[:, 2] - alt0], axis=1)
    return enu, (float(lat0), float(lon0), float(alt0))


# f_veh sits FVEH_LATERAL_LEFT (m, +left in the vehicle x-fwd/y-left/z-up frame)
# from the OXTS/IMU antenna — the IMU is 0.32 m left of the centreline, so f_veh
# is 0.32 m to its right. See [[reference_kitti_calibration]].
FVEH_LATERAL_LEFT = -0.32


def load_oxts_fveh_enu(oxts_dir: str | Path, n: int,
                       lateral_left: float = FVEH_LATERAL_LEFT
                       ) -> tuple[np.ndarray, tuple[float, float, float]]:
    """OXTS samples 0..n-1 -> the f_veh (vehicle-centreline) ENU track + datum.

    The OXTS position is the IMU/GPS antenna; f_veh is that origin shifted
    ``(0, lateral_left, 0)`` in the vehicle frame (x-fwd, y-left, z-up) — i.e.
    laterally onto the centreline. We rotate that body-frame lateral shift into
    ENU by the per-frame OXTS yaw (field 5: 0=east, CCW+), so each frame's f_veh
    ENU is the GNSS pose of f_veh for that frame.
    """
    oxts_dir = Path(oxts_dir)
    rows = [np.loadtxt(oxts_dir / f"{i:010d}.txt") for i in range(n)]
    ll = np.array([r[:3] for r in rows])
    yaw = np.array([r[5] for r in rows])
    lat0, lon0, alt0 = ll[0]
    east = np.radians(ll[:, 1] - lon0) * _R_EARTH * math.cos(math.radians(lat0))
    north = np.radians(ll[:, 0] - lat0) * _R_EARTH
    imu_enu = np.stack([east, north, ll[:, 2] - alt0], axis=1)
    left = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], axis=1)  # +left in ENU
    fveh_enu = imu_enu + lateral_left * left
    return fveh_enu, (float(lat0), float(lon0), float(alt0))


# Measured along-track GNSS bias (truth - ours) in ENU metres, carried over
# from the seq04 KML export (export_static_scene_kml.py); see
# [[reference_seq04_gnss_offset]]. Added to every static-scene coordinate.
SEQ04_ENU_OFFSET = np.array([0.20, 1.37, 0.0])


def georef_from_orb_gnss(
    trajectory: str | Path = "outputs/pocs/orb_slam3/CameraTrajectory.txt",
    oxts_dir: str | Path = "data/kitti/raw/2011_09_30/2011_09_30_drive_0016_sync/oxts/data",
    enu_offset: np.ndarray = SEQ04_ENU_OFFSET,
) -> GeoReference:
    """Fit map(ORB-world) -> ENU from camera centres vs GNSS, as in the KML export.

    Umeyama (scale+R+t, as yesterday) over camera centres vs OXTS ENU, plus the
    stored ``enu_offset`` bias correction.
    """
    T = np.loadtxt(trajectory)
    cam_centres = T[:, [3, 7, 11]]  # translation of each Twc row
    enu, datum = load_oxts_enu(oxts_dir, len(cam_centres))
    return GeoReference.from_point_correspondences(
        cam_centres, enu, datum, with_scale=True, enu_offset=enu_offset
    )


# -- convenience assembler ----------------------------------------------
def build_seq04_map(
    poles_signs: str | Path = "outputs/static_env/poles_signs/poles_signs_world.json",
    arrows: str | Path = "outputs/static_env/arrows_lanemodel/arrows_world.json",
    lanes: str | Path = "outputs/ego_lanes/lane_map/lane_polylines.json",
    created: str = "",
    georef: bool = True,
) -> MapAdas:
    """Assemble the seq04 ADAS map from existing artifacts (georef from GNSS)."""
    m = MapAdas(frame=FrameInfo(
        name="M",
        provenance="ORB-SLAM3 seq04 world (cam0 frame0, X-right Y-down Z-fwd, metric)",
        created=created,
    ))
    for obj in load_poles_signs(poles_signs):
        m.add(obj)
    for obj in load_arrows(arrows):
        m.add(obj)
    for obj in load_lane_polylines(lanes):
        m.add(obj)
    if georef:
        m.georef = georef_from_orb_gnss()
    return m
