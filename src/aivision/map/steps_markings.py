"""Markings step (RLMD detector + lane-model ground plane) — T6 of the map-gen workflow.

Lives outside ``mapgen.py`` so importing the framework never drags in cv2 / the
RLMD road-marking detector; the heavy work happens inside :func:`lift_arrows`,
called from :meth:`MarkingsStep.process`. Registered as ``(markings,
rlmd+lanemodel)`` — the canonical config string (the backfill ``step.json``
recorded ``RLMD+lanemodel-plane``; the live config is the source of truth for
dispatch, see the T0 audit consistency nits).

Refactored core of ``scripts/lift_arrows_lanemodel.py`` (the rendering byproducts
— BEV png + reprojection mp4 — stay in the script; the step emits only the
contract). Straight-ahead road arrows are recovered WITHOUT per-frame stereo:

  1. per-frame detection — straight / left- / right-straight arrows from the RLMD
     road-marking detector (``aivision.road_marking``),
  2. ground plane from the LANE MODEL — the fitted ego-lane borders already sit on
     the road surface, so a LOCAL least-squares plane through the border points in
     this frame's forward window is the drop target (no noisy single-frame stereo),
  3. lift every arrow-mask pixel by ray-plane intersection, then an ego-lane gate
     keeps only arrows whose lifted centroid falls between the interpolated borders,
  4. cluster footprints across all frames (union-find) → one physical arrow; fit a
     3D pose (centroid + heading, PCA clamped to the lane tangent) + a sized glyph.

Out contract (``map-gen/markings/out/arrows_world.json``): a list of arrows in the
SLAM world (cam0) frame, each ``{cls, centroid, forward, length, width,
yaw_to_lane_deg, n_frames, glyph_world}`` (consumed by ``ingest.load_arrows``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mapgen import MapGenContext, MapGenStep, register_step

ARROW_CLASSES = {"straight_arrow", "left_straight_arrow", "right_straight_arrow"}


# --- geometry / clustering helpers (refactored from the lift_arrows_lanemodel producer) ---

def fit_plane(P: np.ndarray):
    """Least-squares plane n·X = p through world points P (≥3). Returns (n unit, p)."""
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    return n, float(n @ c)


def union_find_clusters(centroids: np.ndarray, eps: float) -> list[list[int]]:
    """Group centroids whose pairwise distance < eps (single-linkage)."""
    n = len(centroids)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(centroids[i] - centroids[j]) < eps:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def arrow_glyph(L: float, W: float) -> np.ndarray:
    """Canonical straight-arrow outline in (along, lateral) metres; tip at +L/2 (forward)."""
    sh = L / 2 - 0.42 * L                          # shaft/head boundary
    shw, hw = 0.22 * W, 0.50 * W                   # shaft half-width, head half-width
    return np.array([
        (-L / 2, -shw), (sh, -shw), (sh, -hw), (L / 2, 0.0),
        (sh, hw), (sh, shw), (-L / 2, shw),
    ])


def fit_arrow(pts: np.ndarray, fwd0: np.ndarray, n: np.ndarray):
    """Fit a 3D pose + glyph to a cluster's pooled world points lying on plane normal n.
    fwd0 = lane tangent (forward) to disambiguate / clamp the PCA heading. Returns dict."""
    c = pts.mean(0)
    f = fwd0 - (fwd0 @ n) * n                       # forward projected into the plane
    f /= np.linalg.norm(f)
    r = np.cross(n, f)
    r /= np.linalg.norm(r)
    rel = pts - c
    a, b = rel @ f, rel @ r                          # in-plane coords (along, lateral)

    # PCA heading, but clamp to the lane tangent if it disagrees (noisy thin blobs)
    cov = np.cov(np.stack([a, b]))
    w, V = np.linalg.eigh(cov)
    mj = V[:, np.argmax(w)]                          # major axis in (along, lateral)
    if mj[0] < 0:
        mj = -mj
    yaw = float(np.degrees(np.arctan2(mj[1], mj[0])))
    if abs(yaw) > 30.0:                              # PCA went sideways → trust lane
        mj = np.array([1.0, 0.0])
        yaw = 0.0
    mn = np.array([-mj[1], mj[0]])

    pa, pb = rel @ (f * mj[0] + r * mj[1]), rel @ (f * mn[0] + r * mn[1])
    L = float(np.clip(np.percentile(pa, 95) - np.percentile(pa, 5), 1.5, 9.0))
    Wd = float(np.clip(np.percentile(pb, 90) - np.percentile(pb, 10), 0.25, 1.4))

    maj3 = f * mj[0] + r * mj[1]
    min3 = f * mn[0] + r * mn[1]
    g = arrow_glyph(L, Wd)
    glyph_w = c + g[:, 0:1] * maj3 + g[:, 1:2] * min3
    return {"centroid": c, "forward": maj3, "length": L, "width": Wd,
            "yaw_to_lane_deg": yaw, "glyph_world": glyph_w}


def arrows_to_json(arrows: list[dict]) -> list[dict]:
    """Serialize fitted arrows (numpy fields) to the JSON contract list."""
    return [{"cls": a["cls"], "centroid": a["centroid"].tolist(),
             "forward": a["forward"].tolist(), "length": a["length"],
             "width": a["width"], "yaw_to_lane_deg": a["yaw_to_lane_deg"],
             "n_frames": a["n_frames"], "glyph_world": a["glyph_world"].tolist()}
            for a in arrows]


def lift_arrows(
    seq_dir: Path,
    poses: np.ndarray,
    lane_model: dict,
    *,
    conf: float = 0.25,
    z_max: float = 45.0,
    subsample: int = 2,
    lane_margin: float = 0.4,
    cluster_eps: float = 2.0,
    min_obs: int = 2,
    start: int = 0,
    end: int = -1,
    log=None,
) -> dict:
    """Detect → lane-model-plane lift → ego-lane gate → cluster → fit straight arrows.

    ``poses`` is (N,4,4) cam0→world (Twc). ``lane_model`` is the parsed
    ``lane_polylines_model.json`` (``{left, right}`` world-frame border polylines,
    sorted by Z). Returns ``{arrows, footprints, contours_by_frame}``: ``arrows`` is
    the fitted list (numpy fields — feed through :func:`arrows_to_json` for the
    contract); ``footprints`` + ``contours_by_frame`` are kept for rendering.
    """
    import cv2

    from .. import road_marking as rm
    from ..kitti import _read_calib, load_sequence
    from .ingest import make_arrow
    from .virtual_camera import make_artifact, sensor_T_fveh_cam

    T_sensor = sensor_T_fveh_cam()       # cam0 -> f_veh(t): reframe each detection

    def _say(msg: str) -> None:
        (log.info if log is not None else print)(msg)

    color = load_sequence(seq_dir, camera="image_2")
    N = color.n_frames
    end = N if end < 0 else min(end, N)

    K = _read_calib(seq_dir / "calib.txt")["P2"][:3, :3]
    Kinv = np.linalg.inv(K)
    if poses.shape[0] < N:
        raise ValueError(f"poses ({poses.shape[0]}) fewer than frames ({N})")

    Lb = np.array(lane_model["left"], np.float64)        # (M,3) world, sorted by Z
    Rb = np.array(lane_model["right"], np.float64)
    border_pts = np.vstack([Lb, Rb])
    Zc = Lb[:, 2]                                        # forward axis samples (monotone)
    centerline = 0.5 * (Lb + Rb)
    leftX = lambda z: np.interp(z, Zc, Lb[:, 0])         # noqa: E731 — forward-Z interpolators
    rightX = lambda z: np.interp(z, Zc, Rb[:, 0])        # noqa: E731
    z_lo, z_hi = float(Zc.min()), float(Zc.max())

    def lane_tangent(z: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(Zc, z), 1, len(Zc) - 1))
        t = centerline[i] - centerline[i - 1]
        return t / (np.linalg.norm(t) + 1e-9)

    # ---- Pass 1: detect, fit per-frame ground plane from the lane model, lift, ego-lane gate
    footprints = []                                  # {frame, cls, pts(world), centroid}
    contours_by_frame: dict[int, list] = {}          # raw detection outlines for verification
    arrow_dets_by_frame: dict[int, list] = {}        # per-frame RoadArrow (camera frame) = sensor
    n_arrow_frames = 0

    def _sensor_artifact() -> dict:
        return make_artifact(
            {str(t): arrow_dets_by_frame.get(t, []) for t in range(N)},
            n_frames=N, image_size=(color.width, color.height), K=K,
            note="per-frame ego-lane road arrows (RoadArrow) in f_veh(t)")
    for t in range(start, end):
        imC = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        dets = [d for d in rm.detect(imC, conf=conf) if d["name"] in ARROW_CLASSES]
        if not dets:
            continue
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        # local ground plane: lane-model border points inside this frame's forward window
        Pc = (border_pts - C) @ R.T
        win = (Pc[:, 2] > 3.0) & (Pc[:, 2] < z_max)
        if win.sum() < 6:
            continue
        n, p = fit_plane(border_pts[win])
        nC = n @ C
        n_arrow_frames += 1
        for d in dets:
            ys, xs = np.where(d["mask"])
            if subsample > 1:
                ys, xs = ys[::subsample], xs[::subsample]
            if len(xs) == 0:
                continue
            dcam = (Kinv @ np.stack([xs, ys, np.ones_like(xs)], 0)).T   # (M,3) cam rays
            dworld = dcam @ R.T                                          # world ray dirs
            denom = dworld @ n
            ok = np.abs(denom) > 1e-6
            s = np.where(ok, (p - nC) / np.where(ok, denom, 1.0), -1.0)
            ok &= s > 0
            Xw = C + s[:, None] * dworld
            Zcam = (Xw - C) @ R[:, 2]                                    # forward depth
            ok &= (Zcam > 3.0) & (Zcam < z_max)
            Xw = Xw[ok]
            if len(Xw) < 8:
                continue
            ctr = Xw.mean(0)
            cz = ctr[2]
            if not (z_lo <= cz <= z_hi):
                continue
            lx, rx = float(leftX(cz)), float(rightX(cz))
            lo, hi = min(lx, rx) - lane_margin, max(lx, rx) + lane_margin
            if not (lo <= ctr[0] <= hi):                                # ego-lane gate
                continue
            footprints.append({"frame": t, "cls": d["name"], "pts": Xw, "centroid": ctr})
            m = (d["mask"] * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_by_frame.setdefault(t, []).extend(cnts)

            # per-frame sensor: fit a RoadArrow to THIS frame's footprint in CAMERA coords
            # (the single-frame arrow the camera saw, before pose-transform to the map)
            Xc = (Xw - C) @ R
            n_cam = n @ R
            if n_cam[1] < 0:                                # orient consistently (Y-down)
                n_cam = -n_cam
            fwd_cam = lane_tangent(cz) @ R
            try:
                a = fit_arrow(Xc, fwd_cam, n_cam)
            except Exception:
                a = None
            if a is not None:
                aid = f"arrow_f{t:06d}_{len(arrow_dets_by_frame.get(t, []))}"
                obj = make_arrow(
                    aid, a["centroid"], a["forward"], a["glyph_world"], created_frame=t,
                    attributes={"cls": d["name"], "length": a["length"], "width": a["width"],
                                "yaw_to_lane_deg": a["yaw_to_lane_deg"],
                                "range_m": float(np.linalg.norm(a["centroid"])),
                                "depth_m": float(a["centroid"][2])})
                arrow_dets_by_frame.setdefault(t, []).append(
                    obj.transformed(T_sensor).to_dict())   # camera -> f_veh(t)

    _say(f"[arrows] {n_arrow_frames} arrow frames, {len(footprints)} ego-lane footprints")
    if not footprints:
        _say("[arrows] nothing in the ego lane")
        return {"arrows": [], "footprints": [], "contours_by_frame": {},
                "detections_by_frame": _sensor_artifact()}

    # ---- Cluster footprints across frames → physical arrows, fit pose + shape
    cents = np.array([f["centroid"] for f in footprints])
    clusters = union_find_clusters(cents, cluster_eps)
    arrows = []
    for idx in clusters:
        if len(idx) < min_obs:
            continue
        pts = np.vstack([footprints[i]["pts"] for i in idx])
        c = pts.mean(0)
        n, _ = fit_plane(pts)
        if n[1] < 0:                                   # orient roughly "up" (world Y-down)
            n = -n
        a = fit_arrow(pts, lane_tangent(c[2]), n)
        names = [footprints[i]["cls"] for i in idx]
        a["cls"] = max(set(names), key=names.count)
        a["n_frames"] = len(set(footprints[i]["frame"] for i in idx))
        arrows.append(a)
    arrows.sort(key=lambda a: a["centroid"][2])
    _say(f"[arrows] {len(arrows)} physical arrows after clustering "
         f"(≥ {min_obs} obs, eps {cluster_eps} m)")
    for k, a in enumerate(arrows):
        c = a["centroid"]
        _say(f"   #{k} {a['cls']:<20} Z={c[2]:6.1f}m  L={a['length']:.2f} W={a['width']:.2f} "
             f"yaw={a['yaw_to_lane_deg']:+.1f}deg  obs={a['n_frames']}")

    return {"arrows": arrows, "footprints": footprints, "contours_by_frame": contours_by_frame,
            "detections_by_frame": _sensor_artifact()}


def _poses_from_trajectory(traj_json: Path) -> np.ndarray:
    """trajectory.json (Nx12 Twc row-major) → (N, 4, 4) homogeneous Twc."""
    rows = np.asarray(json.loads(traj_json.read_text())["poses"], dtype=np.float64)
    n = rows.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = rows.reshape(n, 3, 4)
    return out


@register_step
class MarkingsStep(MapGenStep):
    """Ego-lane straight-arrow road markings lifted onto the lane-model ground plane.

    Consumes the ``slam`` trajectory (camera poses) and ``lanes`` (the coupled lane
    model = road plane + ego-lane gate borders). Emits the fitted arrows in the SLAM
    world frame; the per-frame stereo plane is deliberately NOT used (see module doc).
    """

    role = "markings"
    type = "rlmd+lanemodel"
    inputs = ["slam", "lanes"]

    def process(self, ctx: MapGenContext) -> list[str]:
        seq_dir = ctx.input("odometry")
        poses = _poses_from_trajectory(ctx.upstream_out("slam") / "trajectory.json")
        lane_model = json.loads(
            (ctx.upstream_out("lanes") / "lane_polylines_model.json").read_text())
        ctx.log.info("markings: seq=%s lane-model borders L=%d R=%d", seq_dir,
                     len(lane_model["left"]), len(lane_model["right"]))

        result = lift_arrows(
            seq_dir, poses, lane_model,
            conf=float(self.params.get("conf", 0.25)),
            cluster_eps=float(self.params.get("cluster_eps", 2.0)),
            log=ctx.log,
        )
        payload = arrows_to_json(result["arrows"])

        out_dir = ctx.out("markings")
        (out_dir / "arrows_world.json").write_text(json.dumps(payload, indent=1))
        det = result["detections_by_frame"]
        (out_dir / "arrow_detections_by_frame.json").write_text(json.dumps(det))
        n_arrow_det = sum(len(v) for v in det["frames"].values())
        ctx.log.info("markings: %d arrows → arrows_world.json; per-frame arrows %d over %d frames",
                     len(payload), n_arrow_det, det["n_frames"])
        return ["map-gen/markings/out/arrows_world.json",
                "map-gen/markings/out/arrow_detections_by_frame.json"]
