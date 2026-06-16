"""Lanes step (EgoLanes + gray-stereo) — T4 of the map-gen workflow.

Lives outside ``mapgen.py`` so importing the framework never drags in onnxruntime
/ open3d / cv2 (the heavy work happens inside the builders, called from
:meth:`LanesStep.process`). Registered as ``(lanes, egolanes+gray-stereo)``.

This is the TWO-ARTIFACT role (closes T0 gap #3): it runs both producers and
writes both contracts.

  1. ``build_lane_cloud`` — per-frame EgoLanes ego-left/ego-right border masks on
     colour-left (image_2), each lane pixel lifted to metric 3D via gray-stereo
     SGBM depth (image_0/1, baseline ~0.537 m) and transformed into the SLAM world
     (cam0) frame by the trajectory. Refactored core of ``scripts/build_lane_map.py``.
  2. ``fit_border_polylines`` → ``out/lane_polylines.json`` — INDEPENDENT left/right
     border polylines (median X,Y per 3 m Z-bin, min 15 pts, 1-2-1 smoothed). This
     reconstructs a producer that was lost from the tree (it survived only as the
     output artifact + its embedded ``method`` string). Consumed by ``assemble``.
  3. ``fit_lane_model_from_pts`` → ``out/lane_polylines_model.json`` — the COUPLED
     lane model (RANSAC border reject + Whittaker smooth; shared centerline ± half
     width). Refactored core of ``scripts/fit_lane_model.py``. Consumed by
     ``objects`` and ``markings`` as the road ground plane.

Both JSONs share the frame ``cam0 (X-right, Y-down, Z-forward), metric meters``;
``left``/``right`` are lists of ``[x, y, z]`` verts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mapgen import MapGenContext, MapGenStep, register_step

FRAME = "cam0 (X-right, Y-down, Z-forward), metric meters"
# ego-left blue, ego-right pink (matches build_lane_map's PLY colours / consumers).
CLASS_RGB = {0: (0.10, 0.30, 1.00), 1: (1.00, 0.30, 0.75)}
CLASS_NAME = {0: "ego_left", 1: "ego_right"}


# --- 1. lane point cloud (refactored build_lane_map core) -------------------

def _make_sgbm(num_disp: int = 128, block: int = 7):
    import cv2
    return cv2.StereoSGBM.create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def build_lane_cloud(
    seq_dir: Path,
    poses_twc: np.ndarray,
    *,
    thr: float | None = None,
    z_min: float = 3.0,
    z_max: float = 40.0,
    subsample: int = 3,
    log=None,
) -> dict:
    """EgoLanes borders → metric 3D world points per class.

    ``poses_twc`` is (N, 4, 4) cam0→world (first frame = identity ⇒ world IS the
    first-camera frame). Returns ``{"pts": {0: (P,3), 1: (Q,3)}, "cam_track":
    (N,3), "intrinsics": (fx,fy,cx,cy,baseline), "validity": (valid_px, total_px)}``.
    """
    import cv2

    from .. import ego_lanes as el
    from ..kitti import _read_calib, load_sequence

    thr = el.DEFAULT_THR if thr is None else thr

    def _say(msg: str) -> None:
        (log.info if log is not None else print)(msg)

    color = load_sequence(seq_dir, camera="image_2")   # colour-left (EgoLanes masks)
    gleft = load_sequence(seq_dir, camera="image_0")    # gray-left  (stereo)
    gright = load_sequence(seq_dir, camera="image_1")   # gray-right (stereo)
    N = color.n_frames
    if len(poses_twc) < N:
        raise ValueError(f"poses ({len(poses_twc)}) fewer than frames ({N})")

    calib = _read_calib(seq_dir / "calib.txt")
    K = calib["P2"][:3, :3]                              # == P0 K (shared rectified intrinsics)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    baseline = float((calib["P0"][0, 3] - calib["P1"][0, 3]) / fx)  # cam0-cam1 (~0.537 m)
    _say(f"[lanemap] fx={fx:.1f} gray-stereo baseline={baseline:.4f} m")

    sgbm = _make_sgbm()
    pts: dict[int, list] = {0: [], 1: []}
    cam_by_frame: dict[int, dict[int, np.ndarray]] = {}   # per-frame camera-coord lane pts
    cam_track = []
    valid_px = total_px = 0

    for t in range(N):
        imC = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        gL = cv2.imread(str(gleft.image_paths[t]), cv2.IMREAD_GRAYSCALE)
        gR = cv2.imread(str(gright.image_paths[t]), cv2.IMREAD_GRAYSCALE)
        disp = sgbm.compute(gL, gR).astype(np.float32) / 16.0

        Rt, Ct = poses_twc[t, :3, :3], poses_twc[t, :3, 3]
        cam_track.append(Ct.copy())

        dets = {d["cls"]: d["mask"] for d in el.detect(imC, thr=thr)}
        for cls in (0, 1):
            mask = dets.get(cls)
            if mask is None:
                continue
            ys, xs = np.where(mask)
            if subsample > 1:
                ys, xs = ys[::subsample], xs[::subsample]
            total_px += len(xs)
            d = disp[ys, xs]
            ok = d > 0
            Z = np.full(len(xs), np.nan)
            Z[ok] = fx * baseline / d[ok]
            good = ok & (Z > z_min) & (Z < z_max)
            if not good.any():
                continue
            valid_px += int(good.sum())
            u, v, z = xs[good], ys[good], Z[good]
            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Xc = np.stack([X, Y, z], 1)                   # camera coords (the per-frame sensor)
            cam_by_frame.setdefault(t, {})[cls] = Xc
            Xw = Xc @ Rt.T + Ct                           # cam0 → world (aggregated map)
            pts[cls].append(Xw)

    out_pts = {cls: (np.concatenate(v, 0) if v else np.empty((0, 3))) for cls, v in pts.items()}
    _say(f"[lanemap] stereo validity at lane pixels: {valid_px}/{total_px} "
         f"({100 * valid_px / max(total_px, 1):.1f}%)  "
         f"ego_left={len(out_pts[0])} ego_right={len(out_pts[1])} pts")
    return {"pts": out_pts, "cam_track": np.asarray(cam_track),
            "cam_by_frame": cam_by_frame, "n_frames": N, "image_size": (color.width, color.height),
            "intrinsics": (fx, fy, cx, cy, baseline),
            "validity": (valid_px, total_px)}


# --- 2. independent border polylines (RECONSTRUCTED lost producer) ----------

def _smooth_121(a: np.ndarray) -> np.ndarray:
    """Binomial 1-2-1 smoothing; endpoints unchanged."""
    if len(a) < 3:
        return a.astype(float)
    s = a.astype(float).copy()
    s[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
    return s


def fit_border_polylines(pts: dict, *, dz: float = 3.0, minpts: int = 15) -> dict:
    """Independent left/right border polylines: median X,Y per Z-bin, 1-2-1 smoothed.

    Reconstructs the lost producer from its recorded method string
    ``"median X,Y per 3m Z-bin (min15 pts), 1-2-1 smoothed"``.
    """
    out = {"frame": FRAME, "method": "median X,Y per 3m Z-bin (min15 pts), 1-2-1 smoothed"}
    for side, cls in (("left", 0), ("right", 1)):
        P = np.asarray(pts[cls], dtype=float)
        if len(P) == 0:
            out[side] = []
            continue
        z, x, y = P[:, 2], P[:, 0], P[:, 1]
        edges = np.arange(z.min(), z.max() + dz, dz)
        zc, xb, yb = [], [], []
        for i in range(len(edges) - 1):
            sel = (z >= edges[i]) & (z < edges[i + 1])
            if int(sel.sum()) < minpts:
                continue
            zc.append(0.5 * (edges[i] + edges[i + 1]))
            xb.append(float(np.median(x[sel])))
            yb.append(float(np.median(y[sel])))
        xs, ys, zs = _smooth_121(np.array(xb)), _smooth_121(np.array(yb)), np.array(zc)
        out[side] = [[round(float(xs[i]), 4), round(float(ys[i]), 4), round(float(zs[i]), 4)]
                     for i in range(len(zs))]
    return out


# --- 2b. per-frame ego-lane borders = the VIRTUAL CAMERA SENSOR (camera frame) ---

def _bin_border_cam(P: np.ndarray, dz: float, minpts: int) -> list:
    """Single border polyline (camera frame): median X,Y per Z-bin, 1-2-1 smoothed."""
    P = np.asarray(P, dtype=float)
    if len(P) == 0:
        return []
    z, x, y = P[:, 2], P[:, 0], P[:, 1]
    edges = np.arange(z.min(), z.max() + dz, dz)
    zc, xb, yb = [], [], []
    for i in range(len(edges) - 1):
        sel = (z >= edges[i]) & (z < edges[i + 1])
        if int(sel.sum()) < minpts:
            continue
        zc.append(0.5 * (edges[i] + edges[i + 1]))
        xb.append(float(np.median(x[sel])))
        yb.append(float(np.median(y[sel])))
    if len(zc) < 2:
        return []
    xs, ys = _smooth_121(np.array(xb)), _smooth_121(np.array(yb))
    return [[round(float(xs[i]), 4), round(float(ys[i]), 4), round(float(zc[i]), 4)]
            for i in range(len(zc))]


def build_lane_detections(cloud: dict, *, dz: float = 3.0, minpts: int = 6) -> dict:
    """Per-frame ego-left/ego-right borders (camera frame) as LaneMarking objects.

    Reads the ``cam_by_frame`` the cloud builder kept (the camera-frame lane points
    that, transformed by each pose, fed the aggregated map). Each frame's border is a
    :class:`LaneMarking` (same structure as the map), only in CAMERA coords — the
    single-frame ego-lane the camera actually saw, independent of the map.
    """
    from .ingest import make_lane_marking
    from .virtual_camera import make_artifact, sensor_T_fveh_cam

    T_sensor = sensor_T_fveh_cam()       # cam0 -> f_veh(t): reframe each detection
    n_frames = int(cloud["n_frames"])
    fx, fy, cx, cy, _ = cloud["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    frames: dict[str, list] = {str(t): [] for t in range(n_frames)}
    for t, by_cls in cloud["cam_by_frame"].items():
        for cls, P in by_cls.items():
            verts = _bin_border_cam(P, dz, minpts)
            if len(verts) < 2:
                continue
            side = CLASS_NAME[cls]
            obj = make_lane_marking(
                f"{side}_f{t:06d}", side, verts, created_frame=t,
                method="per-frame median X,Y per Z-bin, 1-2-1 smoothed (camera frame)",
                attributes={"n_pts": int(len(P))})
            frames[str(t)].append(obj.transformed(T_sensor).to_dict())   # camera -> f_veh(t)
    return make_artifact(frames, n_frames=n_frames, image_size=cloud["image_size"], K=K,
                         note="per-frame ego-lane borders (LaneMarking) in f_veh(t)")


# --- 3. coupled lane model (refactored fit_lane_model core) -----------------

def _robust_bin(z, x, y, edges, minpts, mad_k=3.0):
    zc, xb, yb = [], [], []
    for i in range(len(edges) - 1):
        sel = (z >= edges[i]) & (z < edges[i + 1])
        if sel.sum() < minpts:
            continue
        xs, ys = x[sel], y[sel]
        mx = np.median(xs)
        mad = np.median(np.abs(xs - mx)) + 1e-6
        keep = np.abs(xs - mx) < mad_k * 1.4826 * mad
        if keep.sum() < minpts:
            keep = np.ones_like(xs, bool)
        zc.append(0.5 * (edges[i] + edges[i + 1]))
        xb.append(np.median(xs[keep]))
        yb.append(np.median(ys[keep]))
    return np.array(zc), np.array(xb), np.array(yb)


def _ransac_poly(z, x, deg, thresh, iters=300, seed=0):
    n = len(z)
    if n <= deg + 1:
        return np.ones(n, bool)
    rng = np.random.default_rng(seed)
    zn = (z - z.mean()) / (z.std() + 1e-9)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, deg + 1, replace=False)
        try:
            c = np.polyfit(zn[idx], x[idx], deg)
        except Exception:
            continue
        inl = np.abs(np.polyval(c, zn) - x) < thresh
        if inl.sum() > best.sum():
            best = inl
    if best.sum() > deg + 1:
        c = np.polyfit(zn[best], x[best], deg)
        best = np.abs(np.polyval(c, zn) - x) < thresh
    return best


def _whittaker(y, w, lam, dz):
    n = len(y)
    D = np.diff(np.eye(n), 2, axis=0) / (dz ** 2)
    W = np.diag(w)
    A = W + lam * (D.T @ D)
    return np.linalg.solve(A, W @ np.nan_to_num(y))


def fit_lane_model_from_pts(
    pts: dict, *, dz: float = 2.0, minpts: int = 12, ransac_deg: int = 3,
    ransac_thresh: float = 0.35, lam_center: float = 20.0, lam_width: float = 400.0,
    lam_height: float = 80.0, log=None,
) -> dict:
    """Coupled lane model from per-class border points (refactor of fit_lane_model)."""
    def _say(msg: str) -> None:
        (log.info if log is not None else print)(msg)

    Pl, Pr = np.asarray(pts[0], dtype=float), np.asarray(pts[1], dtype=float)
    zmin = max(Pl[:, 2].min(), Pr[:, 2].min())
    zmax = min(Pl[:, 2].max(), Pr[:, 2].max())
    edges = np.arange(zmin, zmax + dz, dz)

    raw = {}
    for name, P in (("left", Pl), ("right", Pr)):
        zc, xb, yb = _robust_bin(P[:, 2], P[:, 0], P[:, 1], edges, minpts)
        inl = _ransac_poly(zc, xb, ransac_deg, ransac_thresh)
        _say(f"[lanemodel] {name}: {len(zc)} bins, {int(inl.sum())} RANSAC inliers")
        raw[name] = dict(z=zc, x=xb, y=yb, inl=inl)

    grid = 0.5 * (edges[:-1] + edges[1:])

    def to_grid(d):
        x = np.full(len(grid), np.nan)
        y = np.full(len(grid), np.nan)
        w = np.zeros(len(grid))
        gi = np.round((d["z"] - grid[0]) / dz).astype(int)
        ok = (gi >= 0) & (gi < len(grid)) & d["inl"]
        x[gi[ok]] = d["x"][ok]
        y[gi[ok]] = d["y"][ok]
        w[gi[ok]] = 1.0
        return x, y, w

    xl, yl, wl = to_grid(raw["left"])
    xr, yr, wr = to_grid(raw["right"])
    wboth = wl * wr

    xc = 0.5 * (np.nan_to_num(xl) + np.nan_to_num(xr))
    hw = 0.5 * (np.nan_to_num(xr) - np.nan_to_num(xl))
    xc_s = _whittaker(xc, wboth, lam_center, dz)
    hw_s = _whittaker(hw, wboth, lam_width, dz)
    yl_s = _whittaker(yl, wl, lam_height, dz)
    yr_s = _whittaker(yr, wr, lam_height, dz)
    hw_s = np.clip(hw_s, 0.3, None)
    xl_s, xr_s = xc_s - hw_s, xc_s + hw_s
    full_w = 2.0 * hw_s

    keep = wboth > 0
    z = grid[keep]
    _say(f"[lanemodel] lane width: mean {full_w[keep].mean():.2f} m  "
         f"range {full_w[keep].min():.2f}-{full_w[keep].max():.2f} m")
    return {
        "frame": FRAME,
        "method": (f"coupled lane model: RANSAC(deg{ransac_deg}) border reject + "
                   f"Whittaker smooth (lam_center={lam_center}, lam_width={lam_width}); "
                   f"left=center-halfwidth, right=center+halfwidth"),
        "lane_width_m": {"mean": round(float(full_w[keep].mean()), 3),
                         "min": round(float(full_w[keep].min()), 3),
                         "max": round(float(full_w[keep].max()), 3)},
        "left": [[round(float(xl_s[i]), 4), round(float(yl_s[i]), 4), round(float(z[j]), 4)]
                 for j, i in enumerate(np.where(keep)[0])],
        "right": [[round(float(xr_s[i]), 4), round(float(yr_s[i]), 4), round(float(z[j]), 4)]
                  for j, i in enumerate(np.where(keep)[0])],
    }


def _poses_from_trajectory(traj_json: Path) -> np.ndarray:
    rows = np.asarray(json.loads(traj_json.read_text())["poses"], dtype=np.float64)
    n = rows.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = rows.reshape(n, 3, 4)
    return out


@register_step
class LanesStep(MapGenStep):
    """EgoLanes borders + gray-stereo depth → border polylines + coupled lane model."""

    role = "lanes"
    type = "egolanes+gray-stereo"
    inputs = ["slam", "input:odometry"]

    def process(self, ctx: MapGenContext) -> list[str]:
        seq_dir = ctx.input("odometry")
        poses = _poses_from_trajectory(ctx.upstream_out("slam") / "trajectory.json")
        z_max = float(self.params.get("z_max", 40.0))
        dz = float(self.params.get("dz", 2.0))
        ctx.log.info("lanes: seq=%s z_max=%.1f dz=%.1f", seq_dir, z_max, dz)

        cloud = build_lane_cloud(seq_dir, poses, z_max=z_max, log=ctx.log)
        pts = cloud["pts"]

        out_dir = ctx.out("lanes")
        poly = fit_border_polylines(pts)
        (out_dir / "lane_polylines.json").write_text(json.dumps(poly))
        model = fit_lane_model_from_pts(pts, dz=dz, log=ctx.log)
        (out_dir / "lane_polylines_model.json").write_text(json.dumps(model))
        det = build_lane_detections(cloud)
        (out_dir / "lane_detections_by_frame.json").write_text(json.dumps(det))
        n_lane_det = sum(len(v) for v in det["frames"].values())
        ctx.log.info("lanes: polylines %d/%d verts, model %d/%d verts, "
                     "per-frame borders %d over %d frames",
                     len(poly["left"]), len(poly["right"]),
                     len(model["left"]), len(model["right"]), n_lane_det, det["n_frames"])
        return [
            "map-gen/lanes/out/lane_polylines.json",
            "map-gen/lanes/out/lane_polylines_model.json",
            "map-gen/lanes/out/lane_detections_by_frame.json",
        ]
