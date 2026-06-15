"""Objects step (Vistas seg + GTSRB ViT + delineator) — T5 of the map-gen workflow.

Lives outside ``mapgen.py`` so importing the framework never drags in cv2 / torch
/ the GTSRB ViT + YOLOv12 delineator toolchains; the heavy work happens inside
:func:`lift_objects`, called from :meth:`ObjectsStep.process`. Registered as
``(objects, vistas+gtsrb+delineator)`` — the canonical config string (the backfill
``step.json`` recorded a differently-cased variant; the live config is the source
of truth for dispatch, see the T0 audit consistency nits).

Refactored core of ``scripts/lift_poles_signs.py`` (the rendering byproducts —
BEV png + reprojection mp4 — are dropped; the step emits only the contract). Off-
road vertical structures and signs have no lane-plane constraint, so each physical
object is recovered by:

  1. per-frame detection — poles / traffic-lights as upright connected components of
     the Mapillary-Vistas semantic seg (read from the ``segmentation`` cache);
     sign regions classified by the 43-class GTSRB ViT (every recognised class, not
     just speed limits); guideposts (Leitpfosten) by the YOLOv12 delineator,
  2. COARSE 3D for association only — median fused-rho depth (the ``depth`` mkv) at
     the detection → noisy world position,
  3. cluster across all frames (union-find) → one physical object,
  4. FINE 3D — robust least-squares triangulation of the bearing rays from every
     member frame (wide along-track baseline beats single-frame stereo, and honours
     [[feedback_no_anchor_to_ego_path]] — the estimate comes from observations).

Out contract (``map-gen/objects/out/poles_signs_world.json``): ``{poles, signs,
traffic_lights, guideposts}`` in the SLAM world (cam0) frame, metric meters; each
object carries ``center`` + a size (``height``/``size``), ``n_frames`` and the
triangulation residual ``res_m`` (consumed by ``ingest.load_poles_signs``).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from .mapgen import MapGenContext, MapGenStep, register_step

POLE_IDS = {44, 45, 47}        # Vistas: Street Light, Pole, Utility Pole
SIGN_FRONT_ID = 50             # Vistas: Traffic Sign (Front)
TLIGHT_ID = 48                 # Vistas: Traffic Light


def sign_label(name: str) -> str:
    """Compact tag for logging: speed limits → the number; else uppercase initials."""
    m = re.search(r"(\d+)\s*km/h", name)
    if m:
        return ("E" if name.lower().startswith("end") else "") + m.group(1)
    return "".join(w[0] for w in re.findall(r"[A-Za-z0-9]+", name))[:4].upper()


# --- geometry / clustering helpers (copied from the lift_poles_signs producer) ---

def union_find_clusters(cents: np.ndarray, eps: float) -> list[list[int]]:
    n = len(cents)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(cents[i] - cents[j]) < eps:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    g: dict[int, list[int]] = {}
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    return list(g.values())


def _triangulate_once(origins, dirs):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, r in zip(origins, dirs):
        P = np.eye(3) - np.outer(r, r)
        A += P
        b += P @ o
    if np.linalg.cond(A) > 1e6:
        return None, None
    p = np.linalg.solve(A, b)
    res = np.array([np.linalg.norm((np.eye(3) - np.outer(r, r)) @ (p - o))
                    for o, r in zip(origins, dirs)])
    return p, res


def triangulate(origins: np.ndarray, dirs: np.ndarray, trim: float = 1.0):
    """Robust least-squares point closest to the world rays (o_i, r_i unit). Fits, then
    drops rays whose perpendicular residual exceeds max(trim, 2.5*median) and refits once
    — so a single mis-associated bearing can't drag the estimate. Returns
    (point, mean_residual_m) or (None, inf) if degenerate / too few rays survive."""
    p, res = _triangulate_once(origins, dirs)
    if p is None:
        return None, np.inf
    keep = res <= max(trim, 2.5 * np.median(res))
    if keep.sum() >= 3 and keep.sum() < len(res):
        p2, res2 = _triangulate_once(origins[keep], dirs[keep])
        if p2 is not None:
            p, res = p2, res2
    return p, float(np.mean(res))


def decode_rho(video: Path, W: int, H: int) -> np.ndarray:
    """Decode the fused inverse-depth mkv → (N,H,W) uint16 (rho=(pix-1)/65534, Z=1/rho)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE, check=True)
    return np.frombuffer(proc.stdout, np.uint16).reshape(-1, H, W)


def sample_rho_depth(rho_pix: np.ndarray, ys, xs):
    """Median metric depth over pixels (ys,xs) of a gray16 fused-rho frame, or None.
    rho = (pix-1)/65534, Z = 1/rho (matches the depth step encoding)."""
    r = rho_pix[ys, xs]
    r = r[r > 0]
    if len(r) < 8:
        return None
    rho = (r.astype(np.float32) - 1.0) / 65534.0
    rho = rho[rho > 1e-4]
    if len(rho) < 8:
        return None
    return float(1.0 / np.median(rho))


def vertical_ccs(sem, ids, min_area=120, min_h=16, max_aspect=1.4):
    """Connected components of a class-id mask that look like upright structures
    (poles / traffic lights). Returns list of (label_img, label_id, h_px, (u,v) centroid)."""
    import cv2

    mask = np.isin(sem, list(ids)).astype(np.uint8)
    if not mask.any():
        return []
    ncc, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, ncc):
        _, _, bw, bh, area = stats[i]
        if area < min_area or bh < min_h or bh < max_aspect * bw:   # taller than wide
            continue
        out.append((lab, i, float(bh), (float(cent[i][0]), float(cent[i][1]))))
    return out


def lift_objects(
    seq_dir: Path,
    poses: np.ndarray,
    cache_dir: Path,
    rho_video: Path,
    *,
    sign_conf: float = 0.40,
    gp_conf: float = 0.25,
    pole_eps: float = 2.5,
    sign_eps: float = 3.5,
    gp_eps: float = 2.0,
    pole_min_obs: int = 4,
    sign_min_obs: int = 3,
    tl_min_obs: int = 4,
    gp_min_obs: int = 3,
    sign_stable: float = 0.5,
    pole_zmax: float = 55.0,
    sign_zmax: float = 80.0,
    gp_zmax: float = 45.0,
    log=None,
) -> dict:
    """Detect → coarse-3D → cluster → triangulate poles/signs/lights/guideposts.

    ``poses`` is (N,4,4) cam0→world (Twc). Returns the ``poles_signs_world.json``
    payload ``{poles, signs, traffic_lights, guideposts}`` in the SLAM world frame.
    """
    import cv2

    from .. import delineator as dl
    from .. import sign_detector as sd
    from ..kitti import _read_calib, load_sequence

    def _say(msg: str) -> None:
        (log.info if log is not None else print)(msg)

    color = load_sequence(seq_dir, camera="image_2")
    W, H, N = color.width, color.height, color.n_frames

    K = _read_calib(seq_dir / "calib.txt")["P2"][:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    if poses.shape[0] < N:
        raise ValueError(f"poses ({poses.shape[0]}) fewer than frames ({N})")

    rho = decode_rho(rho_video, W, H)     # fused stereo+mono inverse depth
    _say(f"[objects] fused-rho stream: {len(rho)} frames; {N} colour frames")

    def ray_world(u, v, R):
        d = Kinv @ np.array([u, v, 1.0])
        d = R @ d
        return d / np.linalg.norm(d)

    def lift_vertical(ccs, rho_t, R, C, zmax, sink, t):
        for lab, i, h_px, (u, v) in ccs:
            ys, xs = np.where(lab == i)
            Z = sample_rho_depth(rho_t, ys, xs)
            if Z is None or not (4.0 < Z < zmax):
                continue
            d0 = Kinv @ np.array([u, v, 1.0])
            sink.append({"frame": t, "ctr_w": C + (d0 * Z) @ R.T,
                         "ray": ray_world(u, v, R), "origin": C.copy(), "h_px": h_px})

    def lift_box(box, rho_t, R, C, zmax, t):
        x0, y0, x1, y1 = box
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        yi0, yi1 = int(max(0, y0)), int(min(H - 1, y1))
        xi0, xi1 = int(max(0, x0)), int(min(W - 1, x1))
        ys, xs = np.mgrid[yi0:yi1 + 1, xi0:xi1 + 1]
        Z = sample_rho_depth(rho_t, ys.ravel(), xs.ravel())
        if Z is None or not (4.0 < Z < zmax):
            return None
        d0 = Kinv @ np.array([u, v, 1.0])
        return {"frame": t, "ctr_w": C + (d0 * Z) @ R.T, "ray": ray_world(u, v, R),
                "origin": C.copy(), "h_px": float(y1 - y0)}

    # ---- Pass 1: detect, coarse 3D from fused rho, store bearing rays
    pole_obs, sign_obs, tl_obs, gp_obs = [], [], [], []
    for t in range(N):
        npz = cache_dir / f"frame_{t:06d}.npz"
        if not npz.exists() or t >= len(rho):
            continue
        sem = np.load(npz)["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        R, C = poses[t][:3, :3], poses[t][:3, 3]

        pole_ccs = vertical_ccs(sem, POLE_IDS, max_aspect=0.9)        # very tall-thin
        tl_ccs = vertical_ccs(sem, {TLIGHT_ID}, min_area=80, max_aspect=1.2)

        imC = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        gp_dets = dl.detect(imC, conf=gp_conf)                        # YOLOv12 Leitpfosten

        sign_dets = []
        sign_mask = (sem == SIGN_FRONT_ID).astype(np.uint8)
        if sign_mask.sum() > 50:
            nsc, _, sstats, _ = cv2.connectedComponentsWithStats(sign_mask, 8)
            regions = [(int(sstats[i, 0]), int(sstats[i, 1]),
                        int(sstats[i, 0] + sstats[i, 2]), int(sstats[i, 1] + sstats[i, 3]))
                       for i in range(1, nsc) if sstats[i, 4] > 50]
            if regions:
                sign_dets = sd.detect_in_regions(imC, regions, conf=sign_conf)

        if not pole_ccs and not tl_ccs and not sign_dets and not gp_dets:
            continue
        rho_t = rho[t]

        lift_vertical(pole_ccs, rho_t, R, C, pole_zmax, pole_obs, t)
        lift_vertical(tl_ccs, rho_t, R, C, pole_zmax, tl_obs, t)
        for d in sign_dets:
            o = lift_box(d["box"], rho_t, R, C, sign_zmax, t)
            if o is not None:
                o["name"] = d["name"]
                sign_obs.append(o)
        for d in gp_dets:
            o = lift_box(d["box"], rho_t, R, C, gp_zmax, t)
            if o is not None:
                gp_obs.append(o)

    _say(f"[objects] coarse obs: {len(pole_obs)} pole, {len(sign_obs)} sign, "
         f"{len(tl_obs)} traffic-light, {len(gp_obs)} guidepost")

    # ---- cluster + triangulate → one physical object per cluster
    def build(obs, eps, min_obs, kind, stable_frac=0.0):
        if not obs:
            return []
        cl = union_find_clusters(np.array([o["ctr_w"] for o in obs]), eps)
        out = []
        for idx in cl:
            if len(idx) < min_obs:
                continue
            O = np.array([obs[i]["origin"] for i in idx])
            D = np.array([obs[i]["ray"] for i in idx])
            p, res = triangulate(O, D)
            if p is None or res > 4.0:
                continue
            sizes = []
            for i in idx:
                Zi = float((p - obs[i]["origin"]) @ poses[obs[i]["frame"]][:3, 2])
                if Zi > 1.0:
                    sizes.append(obs[i]["h_px"] * Zi / fy)
            size = float(np.median(sizes)) if sizes else 1.0
            n_frames = len(set(obs[i]["frame"] for i in idx))
            rec = {"kind": kind, "center": p, "size": size,
                   "n_frames": n_frames, "res_m": res}
            if kind == "sign":
                names = [obs[i]["name"] for i in idx]
                top = max(set(names), key=names.count)
                if names.count(top) < stable_frac * len(names):
                    continue
                rec["name"] = top
                rec["agree"] = names.count(top) / len(names)
            out.append(rec)
        out.sort(key=lambda r: r["center"][2])
        return out

    poles = build(pole_obs, pole_eps, pole_min_obs, "pole")
    signs = build(sign_obs, sign_eps, sign_min_obs, "sign", sign_stable)
    tlights = build(tl_obs, pole_eps, tl_min_obs, "light")
    guideposts = build(gp_obs, gp_eps, gp_min_obs, "guidepost")
    poles = [p for p in poles if 1.0 < p["size"] < 14.0]            # plausible pole height
    tlights = [t for t in tlights if 0.3 < t["size"] < 6.0]        # plausible light-head height
    guideposts = [g for g in guideposts if 0.4 < g["size"] < 2.0]  # Leitpfosten ~1-1.5 m
    _say(f"[objects] {len(poles)} poles, {len(signs)} signs, {len(tlights)} traffic lights, "
         f"{len(guideposts)} guideposts after cluster+triangulate")
    for k, s in enumerate(signs):
        c = s["center"]
        _say(f"   sign #{k} {s['name']:<24} X={c[0]:6.1f} Z={c[2]:6.1f}m  "
             f"~{s['size']:.2f}m obs={s['n_frames']} agree={s['agree']:.2f} res={s['res_m']:.2f}m")

    return {
        "poles": [{"center": p["center"].tolist(), "height": p["size"],
                   "n_frames": p["n_frames"], "res_m": p["res_m"]} for p in poles],
        "signs": [{"name": s["name"], "center": s["center"].tolist(),
                   "size": s["size"], "n_frames": s["n_frames"],
                   "agree": s["agree"], "res_m": s["res_m"]} for s in signs],
        "traffic_lights": [{"center": t["center"].tolist(), "height": t["size"],
                            "n_frames": t["n_frames"], "res_m": t["res_m"]} for t in tlights],
        "guideposts": [{"center": g["center"].tolist(), "height": g["size"],
                        "n_frames": g["n_frames"], "res_m": g["res_m"]} for g in guideposts],
    }


def _poses_from_trajectory(traj_json: Path) -> np.ndarray:
    """trajectory.json (Nx12 Twc row-major) → (N, 4, 4) homogeneous Twc."""
    rows = np.asarray(json.loads(traj_json.read_text())["poses"], dtype=np.float64)
    n = rows.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = rows.reshape(n, 3, 4)
    return out


@register_step
class ObjectsStep(MapGenStep):
    """Off-road static objects: poles, traffic signs, traffic lights, guideposts.

    Consumes the ``slam`` trajectory (poses + bearing origins), the ``depth`` rho
    mkv (coarse association depth), the ``segmentation`` cache (Vistas class masks)
    and ``lanes`` (declared for DAG order; the model is the optional viz ground
    plane, not needed for the JSON contract).
    """

    role = "objects"
    type = "vistas+gtsrb+delineator"
    inputs = ["slam", "depth", "segmentation", "lanes"]

    def process(self, ctx: MapGenContext) -> list[str]:
        seq_dir = ctx.input("odometry")
        poses = _poses_from_trajectory(ctx.upstream_out("slam") / "trajectory.json")
        cache_dir = ctx.upstream_out("segmentation")
        rho_video = ctx.upstream_out("depth") / "inv_depth_rho16.mkv"
        ctx.log.info("objects: seq=%s cache=%s rho=%s", seq_dir, cache_dir, rho_video.name)

        payload = lift_objects(
            seq_dir, poses, cache_dir, rho_video,
            sign_conf=float(self.params.get("sign_conf", 0.40)),
            gp_conf=float(self.params.get("gp_conf", 0.25)),
            pole_eps=float(self.params.get("pole_eps", 2.5)),
            log=ctx.log,
        )

        out_dir = ctx.out("objects")
        (out_dir / "poles_signs_world.json").write_text(json.dumps(payload, indent=1))
        ctx.log.info("objects: %d poles, %d signs, %d lights, %d guideposts",
                     len(payload["poles"]), len(payload["signs"]),
                     len(payload["traffic_lights"]), len(payload["guideposts"]))
        return ["map-gen/objects/out/poles_signs_world.json"]
