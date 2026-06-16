"""Sliding-window factor-graph localization (Wilbers et al., ICRA 2019).

A faithful re-implementation of the sliding-window graph-based localizer from
"Localization with Sliding Window Factor Graphs on Third-Party Maps for Automated
Driving" (Wilbers, Merfels, Stachniss), adapted to this project's virtual-camera
sensor + ADAS map.

Differences from the batch smoother in :mod:`aivision.map.localize`:

  * ONLINE / CAUSAL — frames are processed one at a time; the estimate of pose
    ``X_t`` uses only measurements up to ``t`` (no future frames).
  * SLIDING WINDOW by TRUNCATION — only the last ``window`` poses live in the
    active graph; older poses are dropped (not marginalized). The paper rejects
    marginalization (Sec III-D): the high-information map-landmark priors anchor
    the window, and truncation avoids Hessian fill-in and linearization lock-in.
    The single retained boundary belief is a weak prior on the oldest window pose.
  * LANDMARKS ARE VARIABLES — each observed map landmark gets a 3D variable that
    is (a) pulled toward its map position by a tight isotropic *map prior* factor
    (Eq. 3-6; the third-party map as a prior, not a hard point) and (b) tied to the
    poses by *landmark-observation* factors. With a tight map prior this reduces to
    pinning, but keeping it a variable lets associations be revised (the paper's
    point) and is what makes it a sliding-window SLAM-style estimator.
  * ODOMETRY between-factors couple consecutive poses (the paper's ego-motion).
    We have no wheel encoder, so odometry is derived from the reconstructed vehicle
    poses with added drift noise (a stand-in for real odometry).
  * GNSS is used for INITIALIZATION only by default (the paper's best "GBL"
    variant: GNSS bootstraps, then a complete GNSS outage); a continuous mode is
    available.

Frames match :mod:`aivision.map.localize`: poses ``X_t = T_fveh0_fveh(t)`` in the
map/world frame, landmarks in the same frame; ENU is the scoring frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Rot3
from gtsam.symbol_shorthand import L, X  # pose keys X(t), landmark keys L(k)

from .localize import (
    HUBER_K, SIGMA_BEARING, R_world_from_ENU, _ray_cov,
    bootstrap_poses, enu_to_world, map_landmarks, map_lane_curves,
    _nearest_on_polyline, LANE_SIDE,
)
from .map_adas import MapAdas
from .objects import MapClass
from .virtual_camera import VirtualCameraSensor

POINT_CLASSES = (MapClass.POLE, MapClass.SIGN, MapClass.TRAFFIC_LIGHT,
                 MapClass.GUIDE_POST, MapClass.ROAD_ARROW)


@dataclass
class SWConfig:
    window: int = 30              # sliding-window length (poses kept in the graph)
    gnss_sigma: float = 5.0       # m, horizontal GNSS noise
    gnss_up_sigma: float = 1.0e3  # m, height ~free
    gnss_mode: str = "init"       # 'init' (GNSS bootstrap then outage) | 'continuous'
    init_frames: int = 12         # frames that receive a GNSS factor in 'init' mode
    map_sigma: float = 0.05       # m, isotropic landmark map-prior sigma (paper ~0.008)
    odom_sigma_t: float = 0.05    # m, per-step odometry translation noise (1-sigma)
    odom_sigma_r: float = 0.3     # deg, per-step odometry rotation noise (1-sigma)
    boundary_sigma_t: float = 0.10  # m, weak prior on the oldest window pose (truncation belief)
    boundary_sigma_r: float = 0.5   # deg
    gate_m: float = 3.0           # landmark association gate
    lane_gate_m: float = 1.5
    lane_sigma_perp: float = 0.20
    use_lanes: bool = True
    sigma_bearing: float = SIGMA_BEARING
    head_window: int = 40         # GNSS-track window for the INIT heading (GNSS avail. at init)
    lm_iters: int = 2             # inner re-association iterations per frame


@dataclass
class SWResult:
    online_poses: list[Pose3]            # causal estimate of X_t at time t
    online_centers: np.ndarray           # (n,3) causal centres in world frame
    n_assoc: list[int] = field(default_factory=list)   # point assoc used per frame
    n_lane: list[int] = field(default_factory=list)
    states: list = field(default_factory=list)         # per-step graph snapshot (if record)


# --------------------------------------------------------------------------- #
# odometry stand-in (no wheel encoder): noised relative vehicle motion
# --------------------------------------------------------------------------- #
def odometry_from_poses(poses_world: np.ndarray, sigma_t: float, sigma_r_deg: float,
                        seed: int = 0) -> list[Pose3]:
    """Per-step body-frame odometry ``T_{t-1}^{t}`` from world poses + drift noise.

    ``poses_world`` is (n,4,4) ``T_world_fveh(t)``. Returns ``n-1`` relative Pose3
    with seeded Gaussian noise on the SE(3) tangent (paper uses real ego odometry;
    this simulates it from the reconstructed trajectory)."""
    rng = np.random.default_rng(seed)
    sr = np.radians(sigma_r_deg)
    out = []
    for t in range(1, len(poses_world)):
        a = Pose3(poses_world[t - 1]); b = Pose3(poses_world[t])
        rel = a.between(b)
        xi = np.concatenate([rng.normal(0, sr, 3), rng.normal(0, sigma_t, 3)])
        out.append(rel.compose(Pose3.Expmap(xi)))
    return out


# --------------------------------------------------------------------------- #
# factors
# --------------------------------------------------------------------------- #
def map_prior_factor(lm_key: int, p_M: np.ndarray, sigma: float) -> gtsam.PriorFactorPoint3:
    """Third-party-map prior over a landmark variable (Eq. 3-4): x^l_k ~ N(m_k, sigma^2 I)."""
    return gtsam.PriorFactorPoint3(lm_key, Point3(*p_M),
                                   gtsam.noiseModel.Isotropic.Sigma(3, sigma))


def landmark_obs_factor(pose_key: int, lm_key: int, meas_cam: np.ndarray,
                        cov: np.ndarray) -> gtsam.CustomFactor:
    """Landmark-observation factor between a pose and a landmark VARIABLE.

    Residual = ``X_t.transformTo(L_k) - measured_pos_cam`` (camera frame), whitened
    by the anisotropic ray covariance. Analytic Jacobians wrt both pose and point
    from ``transformTo``. Huber-robust."""
    meas = np.asarray(meas_cam, float)
    base = gtsam.noiseModel.Gaussian.Covariance(cov)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K), base)

    def err(this, v, H):
        pose = v.atPose3(this.keys()[0]); pt = v.atPoint3(this.keys()[1])
        if H is not None:
            Hp = np.zeros((3, 6), order="F"); Hl = np.zeros((3, 3), order="F")
            pc = pose.transformTo(pt, Hp, Hl)
            H[0] = Hp; H[1] = Hl
        else:
            pc = pose.transformTo(pt)
        return np.asarray(pc) - meas

    return gtsam.CustomFactor(noise, [pose_key, lm_key], err)


def lane_obs_factor(pose_key: int, v_cam: np.ndarray, q_M: np.ndarray,
                    tangent: np.ndarray, sigma_perp: float,
                    sigma_along: float = 5.0) -> gtsam.CustomFactor:
    """Point-to-curve lane factor (fixed map curve); same form as the batch version."""
    v = Point3(*v_cam); q = np.asarray(q_M, float); t = np.asarray(tangent, float)
    P = np.outer(t, t)
    cov = sigma_along ** 2 * P + sigma_perp ** 2 * (np.eye(3) - P)
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

    return gtsam.CustomFactor(noise, [pose_key], err)


# --------------------------------------------------------------------------- #
# per-frame association (map matching, against the current pose estimate)
# --------------------------------------------------------------------------- #
def _match_frame(pose: Pose3, dets, by_cls, gate_m, sigma_bearing,
                 z_min=3.0, z_max=45.0):
    """Match a single frame's detections to map landmarks under ``pose``.

    Returns list of (landmark, meas_cam, cov) for the landmark-observation factors."""
    pred = {}
    for cls, lms in by_cls.items():
        for lm in lms:
            pc = np.asarray(pose.transformTo(Point3(*lm.p_M)), float)
            if z_min < pc[0] < z_max:
                pred.setdefault(cls, []).append((lm, pc))
    out = []
    for obj in dets:
        cls = obj.cls
        if cls not in pred:
            continue
        c = np.asarray(VirtualCameraSensor.center(obj), float)
        depth = float(c[0]) if c[0] > 0 else float(obj.attributes.get("depth_m", c[0]))
        rng = float(np.linalg.norm(c))
        best, bestd = None, gate_m
        for lm, pc in pred[cls]:
            d = float(np.linalg.norm(pc - c))
            if d < bestd:
                best, bestd = lm, d
        if best is not None:
            sigma_range = 0.15 + 0.03 * depth
            sigma_perp = max(rng * sigma_bearing, 0.05)
            out.append((best, c, _ray_cov(c, sigma_range, sigma_perp)))
    return out


def _match_lanes(pose: Pose3, dets, lanes_M, gate_m, sigma_perp, z_min=3.0, z_max=45.0):
    out = []
    for o in dets:
        if o.cls != MapClass.LANE_MARKING:
            continue
        side = LANE_SIDE.get(o.attributes.get("side"))
        V = lanes_M.get(side)
        if V is None or len(V) < 2:
            continue
        for vert in o.geometry["line"].verts:
            v_cam = np.asarray(vert.mean, float)
            if not (z_min < v_cam[0] < z_max):
                continue
            p_M = np.asarray(pose.transformFrom(Point3(*v_cam)))
            q, tan, d = _nearest_on_polyline(p_M, V)
            if q is None or d > gate_m:
                continue
            out.append((v_cam, q, tan, sigma_perp))
    return out


# --------------------------------------------------------------------------- #
# the sliding-window localizer
# --------------------------------------------------------------------------- #
def localize_sliding_window(amap: MapAdas, sensor: VirtualCameraSensor,
                            gnss_enu: np.ndarray, odom_rel: list[Pose3],
                            cfg: SWConfig | None = None,
                            record: bool = False) -> SWResult:
    """Online sliding-window localization (truncation, map-prior landmarks).

    ``odom_rel`` are the ``n-1`` per-step body-frame odometry measurements
    ``T_{t-1}^{t}`` (e.g. from :func:`odometry_from_poses`)."""
    cfg = cfg or SWConfig()
    georef = amap.georef
    n = len(gnss_enu)

    landmarks = map_landmarks(amap, POINT_CLASSES)
    lm_key_of = {lm.obj_id: i for i, lm in enumerate(landmarks)}   # stable landmark keys
    by_cls: dict = {}
    for lm in landmarks:
        by_cls.setdefault(lm.cls, []).append(lm)
    lanes_M = map_lane_curves(amap) if cfg.use_lanes else {}

    # GNSS in the world frame + anisotropic covariance (tight E/N, free up)
    gnss_M = np.array([enu_to_world(georef, np.array([e[0], e[1], 0.0])) for e in gnss_enu])
    R_M_ENU = R_world_from_ENU(georef)
    Sig = np.diag([cfg.gnss_sigma ** 2, cfg.gnss_sigma ** 2, cfg.gnss_up_sigma ** 2])
    gnss_cov = R_M_ENU @ Sig @ R_M_ENU.T

    # initialise pose 0 from the bootstrap (GNSS position + smoothed track heading)
    boot = bootstrap_poses(gnss_enu, georef, cfg.head_window)
    est: dict[int, Pose3] = {0: boot[0]}
    lm_est: dict[str, np.ndarray] = {}            # persistent landmark estimates
    online = [boot[0]]
    n_assoc, n_lane = [0], [0]
    states: list = []

    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[np.full(3, np.radians(cfg.odom_sigma_r)), np.full(3, cfg.odom_sigma_t)])
    bnd_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[np.full(3, np.radians(cfg.boundary_sigma_r)), np.full(3, cfg.boundary_sigma_t)])

    for t in range(1, n):
        # predict the new pose from odometry, then refine over the window
        est[t] = est[t - 1].compose(odom_rel[t - 1])
        a = max(0, t - cfg.window + 1)

        na = nl = 0
        for _ in range(cfg.lm_iters):
            graph = gtsam.NonlinearFactorGraph()
            vals = gtsam.Values()
            for i in range(a, t + 1):
                vals.insert(X(i), est[i])
            # boundary belief on the oldest retained pose (truncation, not marginalization)
            graph.add(gtsam.PriorFactorPose3(X(a), est[a], bnd_noise))
            # odometry between-factors within the window
            for i in range(a + 1, t + 1):
                graph.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), odom_rel[i - 1], odom_noise))
            # GNSS: init-only (bootstrap) or continuous
            for i in range(a, t + 1):
                if cfg.gnss_mode == "continuous" or i < cfg.init_frames:
                    graph.add(gtsam.GPSFactor(X(i), Point3(*gnss_M[i]),
                                              gtsam.noiseModel.Gaussian.Covariance(gnss_cov)))
            # landmark + lane observation factors over the window
            seen_lm: set[str] = set()
            na = nl = 0
            for i in range(a, t + 1):
                dets = sensor.detections(i)
                for lm, meas_cam, cov in _match_frame(est[i], dets, by_cls,
                                                       cfg.gate_m, cfg.sigma_bearing):
                    key = L(lm_key_of[lm.obj_id])
                    if lm.obj_id not in seen_lm:
                        seen_lm.add(lm.obj_id)
                        if lm.obj_id not in lm_est:
                            lm_est[lm.obj_id] = lm.p_M.copy()
                        if not vals.exists(key):
                            vals.insert(key, Point3(*lm_est[lm.obj_id]))
                        graph.add(map_prior_factor(key, lm.p_M, cfg.map_sigma))
                    graph.add(landmark_obs_factor(X(i), key, meas_cam, cov))
                    na += 1
                if cfg.use_lanes:
                    for v_cam, q, tan, sp in _match_lanes(est[i], dets, lanes_M,
                                                          cfg.lane_gate_m, cfg.lane_sigma_perp):
                        graph.add(lane_obs_factor(X(i), v_cam, q, tan, sp))
                        nl += 1

            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(20)
            res = gtsam.LevenbergMarquardtOptimizer(graph, vals, params).optimize()
            for i in range(a, t + 1):
                est[i] = res.atPose3(X(i))
            for oid in seen_lm:
                key = L(lm_key_of[oid])
                if res.exists(key):
                    lm_est[oid] = np.asarray(res.atPoint3(key))
            last_graph, last_res = graph, res

        online.append(est[t])     # causal estimate of X_t (using data up to t)
        n_assoc.append(na); n_lane.append(nl)

        if record:
            # per-pose marginal uncertainty (1-sigma worst-axis of the translation block)
            poses_snap = {}
            try:
                marg = gtsam.Marginals(last_graph, last_res)
            except Exception:
                marg = None
            for i in range(a, t + 1):
                r = 0.5
                if marg is not None:
                    try:
                        tc = np.asarray(marg.marginalCovariance(X(i)))[3:6, 3:6]
                        r = float(np.sqrt(max(np.linalg.eigvalsh(tc)[-1], 1e-6)))
                    except Exception:
                        pass
                poses_snap[i] = [est[i].translation().tolist(), r]
            # active factor edges (world frame M; export converts to ENU)
            elm, elane = [], []
            for i in range(a, t + 1):
                dets = sensor.detections(i)
                pi = est[i].translation().tolist()
                for lm, _mc, _cov in _match_frame(est[i], dets, by_cls, cfg.gate_m, cfg.sigma_bearing):
                    elm.append([pi, lm.p_M.tolist()])
                if cfg.use_lanes:
                    for _vc, q, _tan, _sp in _match_lanes(est[i], dets, lanes_M,
                                                          cfg.lane_gate_m, cfg.lane_sigma_perp):
                        elane.append([pi, np.asarray(q).tolist()])
            eodom = [[est[i - 1].translation().tolist(), est[i].translation().tolist()]
                     for i in range(a + 1, t + 1)]
            use_g = lambda i: cfg.gnss_mode == "continuous" or i < cfg.init_frames
            egnss = [[est[i].translation().tolist(), gnss_M[i].tolist()]
                     for i in range(a, t + 1) if use_g(i)]
            gnode = gnss_M[t].tolist() if use_g(t) else None
            states.append({"a": a, "t": t, "poses": poses_snap, "edges_lm": elm,
                           "edges_lane": elane, "edges_odom": eodom,
                           "edges_gnss": egnss, "gnss": gnode})

    centers = np.array([p.translation() for p in online])
    return SWResult(online_poses=online, online_centers=centers,
                    n_assoc=n_assoc, n_lane=n_lane, states=states)
