"""Factor-graph localization in the georeferenced ADAS map.

Localize the ego camera *inside* the finished, georeferenced ``map_adas.json``
using only two sensors:

  1. the :class:`~aivision.map.virtual_camera.VirtualCameraSensor` — per-frame
     static-object detections in CAMERA coordinates (bearings + ranges to
     poles / signs / lights / guideposts), the raw single-frame perception that
     the map was *built from* (so the localizer may re-observe those landmarks
     but never sees the aggregated map as a sensor);
  2. a NOISY GNSS fix per frame — loose 2D position (sigma ~5 m), no heading,
     height left free.

ORB odometry is deliberately excluded (it is near-ground-truth and would
trivialize the test). Frame coupling is instead a *smoothed trajectory*: weak
constant-velocity + non-holonomic motion priors between consecutive poses, no
odometry sensor.

Observability intuition: GNSS pins E/N loosely but gives no heading and no
height; the camera-to-map associations give orientation and fine position but
no global gauge. Fused in one factor graph the problem is well posed.

The map is treated as FIXED — landmarks are baked into the factors (no landmark
variables), so the only unknowns are the per-frame poses ``X_t = T_M_cam`` in
the map frame ``M``. We solve a batch Levenberg-Marquardt problem, re-doing the
data association between rounds (EM-style) because the bootstrap heading from
the noisy GNSS track is too coarse for one-shot matching.

Frames: the map, the virtual camera and the camera bundle all share frame ``M``
(cam0, X-right / Y-down / Z-forward, metric). The georef ``T_M_to_ENU`` (with a
WGS84 datum) lives in the DB; ENU is the scoring frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Rot3
from gtsam.symbol_shorthand import X  # pose keys X(t)

from .map_adas import MapAdas, _representative_point
from .objects import MapClass
from .virtual_camera import VirtualCameraSensor

# Classes localized via a single representative 3D point (bearing+range factor).
# Poles / signs / lights / guideposts have a natural point; a road arrow's pose
# centroid is one too (its along-track location pins the trajectory along-track,
# which the lanes — free to slide along themselves — cannot). Lane *borders* are
# extended curves and get the dedicated point-to-curve factor below.
POINT_CLASSES = (
    MapClass.POLE,
    MapClass.SIGN,
    MapClass.TRAFFIC_LIGHT,
    MapClass.GUIDE_POST,
    MapClass.ROAD_ARROW,
)

# sensor lane-border side attribute -> map lane-marking side attribute
LANE_SIDE = {"ego_left": "left", "ego_right": "right"}

SIGMA_BEARING = 1.4e-3  # rad, ~1 px / focal length (707) — tight
HUBER_K = 1.345         # robust kernel knee in whitened units


# --------------------------------------------------------------------------- #
# georef helpers: M <-> ENU
# --------------------------------------------------------------------------- #
def enu_to_world(georef, enu: np.ndarray) -> np.ndarray:
    """Inverse of :meth:`GeoReference.to_enu` — map ENU metres into frame ``M``.

    ``to_enu`` computes ``R p + t + offset``; this undoes it exactly (``R``
    carries the georef scale, so we invert the full 3x3).
    """
    R = np.asarray(georef.T_M_to_ENU)[:3, :3]
    t = np.asarray(georef.T_M_to_ENU)[:3, 3]
    off = np.asarray(georef.enu_offset)
    enu = np.asarray(enu, float)
    out = (np.linalg.inv(R) @ (np.atleast_2d(enu) - t - off).T).T
    return out.reshape(enu.shape)


def R_world_from_ENU(georef) -> np.ndarray:
    """Orthonormal rotation ENU -> M (drops the georef scale)."""
    R_M_to_ENU = np.asarray(georef.T_M_to_ENU)[:3, :3]
    A = np.linalg.inv(R_M_to_ENU)            # ENU -> M, with 1/scale
    U, _, Vt = np.linalg.svd(A)              # nearest rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# --------------------------------------------------------------------------- #
# map landmarks (fixed) and the bootstrap trajectory
# --------------------------------------------------------------------------- #
@dataclass
class Landmark:
    obj_id: str
    cls: MapClass
    p_M: np.ndarray  # representative point in frame M


def map_landmarks(amap: MapAdas, classes=POINT_CLASSES) -> list[Landmark]:
    """Fixed point landmarks (one representative point per map object)."""
    out = []
    for cls in classes:
        for o in amap.of_class(cls):
            p = _representative_point(o, None)
            if p is not None:
                out.append(Landmark(o.id, cls, np.asarray(p, float)))
    return out


def gnss_track_headings(enu: np.ndarray, window: int = 15) -> np.ndarray:
    """Per-frame ground-track heading (rad, ENU atan2(N,E)) from a noisy track.

    Uses a wide forward/backward window so the ~5 m fix noise (>> the ~1.5 m
    per-frame step) averages out into a usable ~±15 deg bearing.
    """
    n = len(enu)
    head = np.zeros(n)
    for i in range(n):
        a = max(0, i - window)
        b = min(n - 1, i + window)
        d = enu[b, :2] - enu[a, :2]
        head[i] = np.arctan2(d[1], d[0]) if np.hypot(*d) > 1e-6 else (head[i - 1] if i else 0.0)
    return head


def bootstrap_poses(gnss_enu: np.ndarray, georef, window: int = 15) -> list[Pose3]:
    """Initial VEHICLE pose ``T_fveh0_fveh`` per frame from the noisy GNSS track.

    Position = GNSS fix in the world frame (height left free in the graph).
    Orientation = f_veh +X (forward) along the smoothed ground track, +Z up,
    +Y left, built in ENU then rotated into the world frame via the georef.
    """
    R_world_ENU = R_world_from_ENU(georef)
    head = gnss_track_headings(gnss_enu, window)
    poses = []
    for i in range(len(gnss_enu)):
        f = np.array([np.cos(head[i]), np.sin(head[i]), 0.0])   # forward (ENU)
        up = np.array([0.0, 0.0, 1.0])                          # f_veh +Z up
        left = np.cross(up, f)                                  # f_veh +Y left
        R_ENU_fveh = np.column_stack([f, left, up])            # cols [x-fwd, y-left, z-up]
        R_world_fveh = R_world_ENU @ R_ENU_fveh
        t = enu_to_world(georef, np.array([gnss_enu[i, 0], gnss_enu[i, 1], 0.0]))
        poses.append(Pose3(Rot3(R_world_fveh), Point3(*t)))
    return poses


# --------------------------------------------------------------------------- #
# data association
# --------------------------------------------------------------------------- #
@dataclass
class Assoc:
    """One sensor->map correspondence at a frame."""
    frame: int
    landmark: Landmark
    pos_cam: np.ndarray   # measured detection centre, f_veh(t) coords
    sigma_range: float
    sigma_perp: float


def _det_point(sensor: VirtualCameraSensor, obj):
    """f_veh(t) centre + (range, forward-depth) of a sensor detection.

    Detections are now in the moving vehicle frame f_veh(t) (x-fwd, y-left, z-up),
    so forward depth is the +X component (the stored ``depth_m`` attribute is the
    raw camera-frame depth and is only used as a fallback)."""
    c = np.asarray(sensor.center(obj), float)
    depth = float(c[0]) if c[0] > 0 else float(obj.attributes.get("depth_m", c[0]))
    rng = float(np.linalg.norm(c))
    return c, rng, depth


def associate(
    poses: list[Pose3],
    sensor: VirtualCameraSensor,
    landmarks: list[Landmark],
    *,
    gate_m: float = 6.0,
    z_min: float = 3.0,
    z_max: float = 45.0,
    sigma_bearing: float = SIGMA_BEARING,
) -> list[Assoc]:
    """Match per-frame detections to fixed map landmarks under the current poses.

    For each frame, every map landmark is transformed into the current camera
    frame; class-gated, in-front and in-range. Each detection is matched to the
    nearest same-class predicted landmark within ``gate_m`` (nearest-neighbour,
    one landmark per detection). Robust kernels + EM re-association absorb the
    coarse bootstrap.
    """
    by_cls: dict[MapClass, list[Landmark]] = {}
    for lm in landmarks:
        by_cls.setdefault(lm.cls, []).append(lm)

    out: list[Assoc] = []
    for t, dets in sensor:
        if t >= len(poses):
            break
        pose = poses[t]
        # predict landmark positions in this frame's vehicle frame f_veh(t), per
        # class; in-front = +X (forward) in f_veh, in-range gate on it
        pred: dict[MapClass, list[tuple[Landmark, np.ndarray]]] = {}
        for cls, lms in by_cls.items():
            for lm in lms:
                pc = np.asarray(pose.transformTo(Point3(*lm.p_M)), float)
                if z_min < pc[0] < z_max:
                    pred.setdefault(cls, []).append((lm, pc))

        for obj in dets:
            cls = obj.cls
            if cls not in pred:
                continue
            c, rng, depth = _det_point(sensor, obj)
            best, bestd = None, gate_m
            for lm, pc in pred[cls]:
                d = float(np.linalg.norm(pc - c))
                if d < bestd:
                    best, bestd = lm, d
            if best is not None:
                out.append(Assoc(
                    frame=t, landmark=best, pos_cam=c,
                    sigma_range=0.15 + 0.03 * depth,
                    sigma_perp=max(rng * sigma_bearing, 0.05),
                ))
    return out


# --------------------------------------------------------------------------- #
# lane data association (point-to-curve)
# --------------------------------------------------------------------------- #
@dataclass
class LaneAssoc:
    """One sensor lane-border vertex matched to a point on a map lane curve."""
    frame: int
    v_cam: np.ndarray     # sensor border vertex, camera coords
    q_M: np.ndarray       # nearest point on the map lane polyline (fixed), M
    tangent: np.ndarray   # unit map-lane tangent at q (M), the free direction
    sigma_perp: float


def map_lane_curves(amap: MapAdas) -> dict[str, np.ndarray]:
    """Map lane-marking polylines keyed by side ('left'/'right'), as (n,3) in M."""
    out: dict[str, np.ndarray] = {}
    for o in amap.of_class(MapClass.LANE_MARKING):
        side = o.attributes.get("side")
        out[side] = np.array([v.mean for v in o.geometry["line"].verts], float)
    return out


def _nearest_on_polyline(p: np.ndarray, V: np.ndarray):
    """Nearest point + unit tangent on a polyline ``V`` (n,3) to point ``p``."""
    best_q, best_t, best_d = None, None, np.inf
    for i in range(len(V) - 1):
        a, ab = V[i], V[i + 1] - V[i]
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        s = np.clip(float((p - a) @ ab) / L2, 0.0, 1.0)
        q = a + s * ab
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_q, best_t, best_d = q, ab / np.sqrt(L2), d
    return best_q, best_t, best_d


def associate_lanes(
    poses: list[Pose3],
    sensor: VirtualCameraSensor,
    lanes_M: dict[str, np.ndarray],
    *,
    gate_m: float = 1.5,
    z_min: float = 3.0,
    z_max: float = 45.0,
    sigma_perp: float = 0.20,
) -> list[LaneAssoc]:
    """Match per-frame sensor lane-border vertices to the map lane curves.

    Each vertex is lifted to ``M`` under the current pose and snapped to the
    nearest point of the *same-side* map lane (gated). The match stores that
    point + tangent so the factor penalizes only the perpendicular offset.
    """
    out: list[LaneAssoc] = []
    for t, dets in sensor:
        if t >= len(poses):
            break
        pose = poses[t]
        for o in dets:
            if o.cls != MapClass.LANE_MARKING:
                continue
            side = LANE_SIDE.get(o.attributes.get("side"))
            V = lanes_M.get(side)
            if V is None or len(V) < 2:
                continue
            for vert in o.geometry["line"].verts:
                v_cam = np.asarray(vert.mean, float)
                if not (z_min < v_cam[0] < z_max):     # forward = +X in f_veh(t)
                    continue
                p_M = np.asarray(pose.transformFrom(Point3(*v_cam)))
                q, tan, d = _nearest_on_polyline(p_M, V)
                if q is None or d > gate_m:
                    continue
                out.append(LaneAssoc(t, v_cam, q, tan, sigma_perp))
    return out


# --------------------------------------------------------------------------- #
# factors
# --------------------------------------------------------------------------- #
def _ray_cov(pos_cam: np.ndarray, sigma_range: float, sigma_perp: float) -> np.ndarray:
    """Anisotropic 3x3 covariance of a bearing+range point in camera coords:
    sigma_range along the line of sight, sigma_perp across it."""
    u = pos_cam / max(np.linalg.norm(pos_cam), 1e-9)
    P = np.outer(u, u)
    return sigma_range ** 2 * P + sigma_perp ** 2 * (np.eye(3) - P)


def point_factor(a: Assoc) -> gtsam.CustomFactor:
    """Bearing+range factor: pose X_t vs the FIXED landmark.

    Residual = ``X_t.transformTo(L_M) - measured_pos_cam`` (3D, camera frame),
    whitened by the anisotropic ray covariance (tight across the ray = bearing,
    loose along it = range). Map point baked in; analytic pose Jacobian from
    ``transformTo``. Huber-robust.
    """
    L = Point3(*a.landmark.p_M)
    meas = np.asarray(a.pos_cam, float)
    cov = _ray_cov(meas, a.sigma_range, a.sigma_perp)
    base = gtsam.noiseModel.Gaussian.Covariance(cov)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K), base)

    def err(this, v, H):
        pose = v.atPose3(this.keys()[0])
        if H is not None:
            Hp = np.zeros((3, 6), order="F")
            pc = pose.transformTo(L, Hp, np.zeros((3, 3), order="F"))
            H[0] = Hp
        else:
            pc = pose.transformTo(L)
        return np.asarray(pc) - meas

    return gtsam.CustomFactor(noise, [X(a.frame)], err)


def lane_factor(a: LaneAssoc, sigma_along: float = 5.0) -> gtsam.CustomFactor:
    """Point-to-curve lane factor: a sensor border vertex vs the FIXED map curve.

    Residual = ``X_t.transformFrom(v_cam) - q`` (3D, frame M) whitened by an
    anisotropic covariance that is LOOSE along the lane tangent (free to slide —
    a lane gives no along-track information) and TIGHT in the perpendicular plane.
    That perpendicular plane spans both the lateral (cross-track + yaw) and the
    vertical direction, so lane factors also make HEIGHT observable. Analytic
    pose Jacobian from ``transformFrom``; Huber-robust.
    """
    v = Point3(*a.v_cam)
    q = np.asarray(a.q_M, float)
    t = np.asarray(a.tangent, float)
    P = np.outer(t, t)
    cov = sigma_along ** 2 * P + a.sigma_perp ** 2 * (np.eye(3) - P)
    base = gtsam.noiseModel.Gaussian.Covariance(cov)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K), base)

    def err(this, vv, H):
        pose = vv.atPose3(this.keys()[0])
        if H is not None:
            Hp = np.zeros((3, 6), order="F")
            p = pose.transformFrom(v, Hp, np.zeros((3, 3), order="F"))
            H[0] = Hp
        else:
            p = pose.transformFrom(v)
        return np.asarray(p) - q

    return gtsam.CustomFactor(noise, [X(a.frame)], err)


def gnss_factor(t: int, p_M: np.ndarray, cov_M: np.ndarray) -> gtsam.GPSFactor:
    """Loose 2D position factor on X_t (anisotropic cov: tight E/N, free up)."""
    return gtsam.GPSFactor(X(t), Point3(*p_M),
                           gtsam.noiseModel.Gaussian.Covariance(cov_M))


def _numeric_H(poses: list[Pose3], idx: int, fn, e0: np.ndarray, eps: float = 1e-5):
    H = np.zeros((e0.shape[0], 6), order="F")
    for d in range(6):
        dv = np.zeros(6)
        dv[d] = eps
        ep = list(poses); ep[idx] = poses[idx].retract(dv)
        em = list(poses); em[idx] = poses[idx].retract(-dv)
        H[:, d] = (fn(ep) - fn(em)) / (2 * eps)
    return H


def nonholo_factor(t: int, sigma: float = 0.05) -> gtsam.CustomFactor:
    """Non-holonomic no-side-slip prior between X_t and X_{t+1}.

    The inter-frame displacement, expressed in frame t's f_veh body axes, must
    have ~zero lateral (Y-left) and vertical (Z-up) component — the car moves
    along its heading (+X). 2D residual, numeric Jacobians.
    """
    def resid(poses):
        rel = poses[0].transformTo(poses[1].translation())
        return np.array([rel[1], rel[2]])     # lateral (Y) + vertical (Z) in f_veh

    def err(this, v, H):
        ps = [v.atPose3(k) for k in this.keys()]
        e = resid(ps)
        if H is not None:
            for i in range(2):
                H[i] = _numeric_H(ps, i, resid, e)
        return e

    noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
    return gtsam.CustomFactor(noise, [X(t), X(t + 1)], err)


def const_vel_factor(t: int, sigma: float = 0.30) -> gtsam.CustomFactor:
    """Constant-velocity (zero-jerk) smoothing prior over X_{t-1}, X_t, X_{t+1}.

    Penalizes the change in the body-frame relative motion between the two
    consecutive steps: ``(X_{t-1}^-1 X_t)^-1 (X_t^-1 X_{t+1}) ominus I`` (6D in
    the SE(3) tangent). Numeric Jacobians.
    """
    def resid(poses):
        a = poses[0].between(poses[1])      # X_{t-1}^-1 X_t
        b = poses[1].between(poses[2])      # X_t^-1   X_{t+1}
        return Pose3.Logmap(a.between(b))   # a^-1 b ominus I

    def err(this, v, H):
        ps = [v.atPose3(k) for k in this.keys()]
        e = resid(ps)
        if H is not None:
            for i in range(3):
                H[i] = _numeric_H(ps, i, resid, e)
        return e

    noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
    return gtsam.CustomFactor(noise, [X(t - 1), X(t), X(t + 1)], err)


# --------------------------------------------------------------------------- #
# the localizer
# --------------------------------------------------------------------------- #
@dataclass
class LocalizeConfig:
    gnss_sigma: float = 5.0       # m, horizontal
    gnss_up_sigma: float = 1.0e3  # m, height effectively free
    cv_sigma: float = 0.008       # stiff zero-jerk smoothing (mostly-straight drive)
    nonholo_sigma: float = 0.02   # no side-slip / no vertical motion
    gate_m: float = 3.0           # point-object association gate
    lane_gate_m: float = 1.5      # lane vertex -> curve association gate
    lane_sigma_perp: float = 0.20 # lateral/vertical lane fit noise
    sigma_bearing: float = SIGMA_BEARING  # rad; bearing (perpendicular) measurement noise
    em_rounds: int = 4
    head_window: int = 15


@dataclass
class LocalizeResult:
    poses_M: list[Pose3]                 # estimated T_M_cam per frame
    centers_M: np.ndarray                # (n,3) camera centres in M
    n_assoc: list[int] = field(default_factory=list)        # point assoc / round
    n_lane_assoc: list[int] = field(default_factory=list)   # lane vertex assoc / round


def _build_graph(poses, gnss_M, gnss_cov, assoc, lane_assoc, cfg) -> gtsam.NonlinearFactorGraph:
    g = gtsam.NonlinearFactorGraph()
    for t in range(len(poses)):
        g.add(gnss_factor(t, gnss_M[t], gnss_cov))
    for t in range(len(poses) - 1):
        g.add(nonholo_factor(t, cfg.nonholo_sigma))
    for t in range(1, len(poses) - 1):
        g.add(const_vel_factor(t, cfg.cv_sigma))
    for a in assoc:
        g.add(point_factor(a))
    for a in lane_assoc:
        g.add(lane_factor(a))
    return g


def localize(amap: MapAdas, sensor: VirtualCameraSensor, gnss_enu: np.ndarray,
             cfg: LocalizeConfig | None = None) -> LocalizeResult:
    """Batch-localize the trajectory in ``amap`` from the virtual camera + GNSS.

    ``gnss_enu`` is the per-frame noisy GNSS fix in ENU metres (same datum as
    the map georef). Returns the estimated poses in frame ``M``.
    """
    cfg = cfg or LocalizeConfig()
    georef = amap.georef
    n = len(gnss_enu)

    landmarks = map_landmarks(amap)
    lanes_M = map_lane_curves(amap)
    poses = bootstrap_poses(gnss_enu, georef, cfg.head_window)

    # GNSS measurements in M + anisotropic covariance (tight E/N, free up)
    gnss_M = np.array([enu_to_world(georef, np.array([e[0], e[1], 0.0])) for e in gnss_enu])
    R_M_ENU = R_world_from_ENU(georef)
    Sigma_enu = np.diag([cfg.gnss_sigma ** 2, cfg.gnss_sigma ** 2, cfg.gnss_up_sigma ** 2])
    gnss_cov = R_M_ENU @ Sigma_enu @ R_M_ENU.T

    n_assoc, n_lane = [], []
    for _ in range(cfg.em_rounds):
        assoc = associate(poses, sensor, landmarks, gate_m=cfg.gate_m,
                          sigma_bearing=cfg.sigma_bearing)
        lane_assoc = associate_lanes(poses, sensor, lanes_M, gate_m=cfg.lane_gate_m,
                                     sigma_perp=cfg.lane_sigma_perp)
        n_assoc.append(len(assoc))
        n_lane.append(len(lane_assoc))
        graph = _build_graph(poses, gnss_M, gnss_cov, assoc, lane_assoc, cfg)

        init = gtsam.Values()
        for t in range(n):
            init.insert(X(t), poses[t])
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
        result = opt.optimize()
        poses = [result.atPose3(X(t)) for t in range(n)]

    centers = np.array([p.translation() for p in poses])
    return LocalizeResult(poses_M=poses, centers_M=centers,
                          n_assoc=n_assoc, n_lane_assoc=n_lane)


# --------------------------------------------------------------------------- #
# scoring (in ENU, against the clean OXTS track)
# --------------------------------------------------------------------------- #
def _heading_enu(centers_enu: np.ndarray, window: int = 5) -> np.ndarray:
    return gnss_track_headings(centers_enu, window)


def score(est_centers_M: np.ndarray, georef, gt_enu: np.ndarray,
          est_poses_M: list[Pose3] | None = None, rpe_delta: int = 10) -> dict:
    """ATE / RPE / heading error of an estimate (in M) vs clean OXTS (ENU).

    The georef already aligns ``M`` to the OXTS ENU, so no extra Umeyama step is
    applied; positions are compared directly.
    """
    est_enu = georef.to_enu(est_centers_M)
    gt = np.asarray(gt_enu, float)
    d = est_enu - gt

    ate_3d = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
    ate_2d = float(np.sqrt(np.mean(np.sum(d[:, :2] ** 2, axis=1))))

    # RPE: translation error of relative motion over rpe_delta frames
    rpe = []
    for i in range(len(gt) - rpe_delta):
        de = est_enu[i + rpe_delta] - est_enu[i]
        dg = gt[i + rpe_delta] - gt[i]
        rpe.append(np.linalg.norm(de - dg))
    rpe_t = float(np.sqrt(np.mean(np.square(rpe)))) if rpe else float("nan")

    out = {
        "n": int(len(gt)),
        "ate_2d": ate_2d,
        "ate_3d": ate_3d,
        "ate_2d_max": float(np.max(np.linalg.norm(d[:, :2], axis=1))),
        "rpe_t": rpe_t,
        "rpe_delta": rpe_delta,
    }

    # heading error: estimated vehicle +X forward (in ENU) vs GT track bearing
    if est_poses_M is not None:
        R_ENU_M = np.linalg.inv(R_world_from_ENU(georef))
        gt_head = gnss_track_headings(gt, window=5)
        errs = []
        for i, p in enumerate(est_poses_M):
            fwd_M = p.rotation().matrix()[:, 0]          # f_veh +X (forward) in world
            fwd_enu = R_ENU_M @ fwd_M
            h = np.arctan2(fwd_enu[1], fwd_enu[0])
            e = np.degrees(np.abs(np.arctan2(np.sin(h - gt_head[i]),
                                             np.cos(h - gt_head[i]))))
            errs.append(e)
        out["heading_deg_mean"] = float(np.mean(errs))
        out["heading_deg_med"] = float(np.median(errs))
    return out
