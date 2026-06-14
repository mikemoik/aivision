"""Step 1c of the 3D static-environment thread: lift POLES, TRAFFIC SIGNS (every recognised
class) and TRAFFIC LIGHTS into the world frame and estimate one best-estimate 3D pose per
physical object.

Unlike the arrows (lift_arrows_lanemodel.py), these objects are NOT on the road plane and have
NO ego-lane constraint -- they stand vertically off to the side at unknown height, so the
lane-model plane lift does not apply. Instead:

  * detect per frame from the Mapillary-Vistas semantic seg: poles (Pole/Utility Pole/Street
    Light), traffic lights (Traffic Light), and sign regions (Traffic Sign (Front)) which are
    classified by the full 43-class GTSRB ViT (aivision.sign_detector) -- EVERY recognised
    sign class is registered, not just speed limits; PLUS guideposts (Leitpfosten) from the
    dedicated YOLOv12 delineator detector (aivision.delineator),
  * COARSE per-frame 3D for association only: stereo-SGBM (image_0/image_1) depth at the
    detection -> world position (noisy, esp. on thin poles),
  * cluster those coarse positions across all frames (union-find) -> one physical object; a
    sign cluster is "stable" only if one class dominates >= --sign-stable of its observations,
  * FINE 3D estimate: least-squares TRIANGULATION of the bearing rays from every member frame
    (bearings + the wide along-track baseline beat single-frame stereo, and respect
    [[feedback_no_anchor_to_ego_path]] -- the estimate comes from the observations),
  * render: world BEV + an RGB video with each object's estimated 3D glyph reprojected into
    every frame -- orange pole sticks, red sign billboards with the class tag, green traffic-
    light housings. Render on colour image_2.

Frame/projection conventions match render_lane_reprojection.py / lift_arrows_lanemodel.py.
Segmentation is read from the prebuilt perception cache (no model reload).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence
from aivision import sign_detector as sd
from aivision import delineator as dl

POLE_IDS = {44, 45, 47}        # Vistas: Street Light, Pole, Utility Pole
SIGN_FRONT_ID = 50             # Vistas: Traffic Sign (Front)
TLIGHT_ID = 48                 # Vistas: Traffic Light
ORANGE = (0, 165, 255)         # BGR -- poles
RED = (40, 40, 235)            # BGR -- traffic signs (any class)
GREEN = (60, 220, 60)          # BGR -- traffic lights
WHITE = (245, 245, 245)        # BGR -- guideposts (Leitpfosten)
BORDER = {"left": (255, 130, 30), "right": (190, 60, 255)}
ZNEAR = 3.0


def sign_label(name: str) -> str:
    """Compact on-image tag: speed limits -> the number; else uppercase word-initials."""
    m = re.search(r"(\d+)\s*km/h", name)
    if m:
        return ("E" if name.lower().startswith("end") else "") + m.group(1)
    return "".join(w[0] for w in re.findall(r"[A-Za-z0-9]+", name))[:4].upper()


def read_calib(p: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in p.read_text().strip().splitlines():
        k, *vals = line.replace(":", "").split()
        out[k] = np.array(vals, np.float64).reshape(3, 4)
    return out


def read_poses(p: Path) -> np.ndarray:
    raw = np.loadtxt(p, dtype=np.float64)
    T = np.tile(np.eye(4), (raw.shape[0], 1, 1))
    T[:, :3, :4] = raw.reshape(-1, 3, 4)
    return T


def project_pts(Pw, Rinv, C, fx, fy, cx, cy):
    """World points -> int pixels, or None if any vertex is behind the near plane."""
    Pc = (Pw - C) @ Rinv.T
    if np.any(Pc[:, 2] <= ZNEAR):
        return None
    u = fx * Pc[:, 0] / Pc[:, 2] + cx
    v = fy * Pc[:, 1] / Pc[:, 2] + cy
    return np.round(np.stack([u, v], 1)).astype(np.int32)


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
    -- so a single mis-associated bearing can't drag the estimate. Returns
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
    """Decode the fused-mono inverse-depth mkv -> (N,H,W) uint16 (Z = 1/rho, see dynamic.py)."""
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE, check=True)
    return np.frombuffer(proc.stdout, np.uint16).reshape(-1, H, W)


def sample_rho_depth(rho_pix: np.ndarray, ys, xs):
    """Median metric depth over pixels (ys,xs) of a gray16 fused-rho frame, or None.
    rho = (pix-1)/65534, Z = 1/rho (matches aivision.dynamic decode)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/04"))
    ap.add_argument("--poses", type=Path,
                    default=Path("outputs/pocs/orb_slam3/CameraTrajectory.txt"))
    ap.add_argument("--cache", type=Path,
                    default=Path("outputs/pocs/orb_slam3/perception"))
    ap.add_argument("--rho", type=Path,
                    default=Path("outputs/pocs/orb_slam3/inv_depth_fused_mono_rho_gf_16bit.mkv"),
                    help="fused stereo+mono inverse-depth stream for per-frame coarse depth")
    ap.add_argument("--lane-model", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_polylines_model.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/static_env/poles_signs"))
    ap.add_argument("--sign-conf", type=float, default=0.40,
                    help="per-crop GTSRB confidence to count a sign observation")
    ap.add_argument("--pole-zmax", type=float, default=55.0)
    ap.add_argument("--sign-zmax", type=float, default=80.0)
    ap.add_argument("--pole-eps", type=float, default=2.5, help="pole clustering radius (m)")
    ap.add_argument("--sign-eps", type=float, default=3.5, help="sign clustering radius (m)")
    ap.add_argument("--pole-min-obs", type=int, default=4)
    ap.add_argument("--sign-min-obs", type=int, default=3)
    ap.add_argument("--tl-min-obs", type=int, default=4)
    ap.add_argument("--sign-stable", type=float, default=0.5,
                    help="fraction of a sign cluster's obs that must share the top class")
    ap.add_argument("--gp-conf", type=float, default=0.25, help="delineator detector confidence")
    ap.add_argument("--gp-zmax", type=float, default=45.0)
    ap.add_argument("--gp-eps", type=float, default=2.0, help="guidepost clustering radius (m)")
    ap.add_argument("--gp-min-obs", type=int, default=3)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    color = load_sequence(args.seq_dir, camera="image_2")
    W, H, N = color.width, color.height, color.n_frames
    end = N if args.end < 0 else min(args.end, N)

    calib = read_calib(args.seq_dir / "calib.txt")
    K = calib["P2"][:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    poses = read_poses(args.poses)
    # world "up" = mean camera up over the drive (cam Y is down) -> averages out pitch
    up = -np.mean(poses[:end, :3, 1], axis=0)
    UP = up / np.linalg.norm(up)

    model = json.loads(args.lane_model.read_text()) if args.lane_model.exists() else None
    rho = decode_rho(args.rho, W, H)     # fused stereo+mono inverse depth, far steadier than SGBM
    print(f"[static] fused-rho stream: {len(rho)} frames from {args.rho.name}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def ray_world(u, v, R):
        d = Kinv @ np.array([u, v, 1.0])
        d = R @ d
        return d / np.linalg.norm(d)

    def lift_vertical(ccs, rho_t, R, C, zmax, sink):
        """Coarse-3D each upright CC (pole / light) from fused rho + store its bearing ray."""
        for lab, i, h_px, (u, v) in ccs:
            ys, xs = np.where(lab == i)
            Z = sample_rho_depth(rho_t, ys, xs)
            if Z is None or not (4.0 < Z < zmax):
                continue
            d0 = Kinv @ np.array([u, v, 1.0])
            sink.append({"frame": t, "ctr_w": C + (d0 * Z) @ R.T,
                         "ray": ray_world(u, v, R), "origin": C.copy(), "h_px": h_px})

    def lift_box(box, rho_t, R, C, zmax):
        """Coarse-3D a detection box (sign / guidepost) from fused rho over its extent."""
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
    pole_obs, sign_obs, tl_obs, gp_obs = [], [], [], []   # {frame, ctr_w, ray, origin, h_px, name?}
    for t in tqdm(range(args.start, end), desc="detect+coarse-3D"):
        npz = args.cache / f"frame_{t:06d}.npz"
        if not npz.exists() or t >= len(rho):
            continue
        z = np.load(npz)
        sem = z["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        R, C = poses[t][:3, :3], poses[t][:3, 3]

        pole_ccs = vertical_ccs(sem, POLE_IDS, max_aspect=0.9)        # very tall-thin
        tl_ccs = vertical_ccs(sem, {TLIGHT_ID}, min_area=80, max_aspect=1.2)

        imC = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        gp_dets = dl.detect(imC, conf=args.gp_conf)                   # YOLOv12 Leitpfosten

        # ---- sign regions -> GTSRB classifier (EVERY recognised class, not just speed)
        sign_dets = []
        sign_mask = (sem == SIGN_FRONT_ID).astype(np.uint8)
        if sign_mask.sum() > 50:
            nsc, _, sstats, _ = cv2.connectedComponentsWithStats(sign_mask, 8)
            regions = [(int(sstats[i, 0]), int(sstats[i, 1]),
                        int(sstats[i, 0] + sstats[i, 2]), int(sstats[i, 1] + sstats[i, 3]))
                       for i in range(1, nsc) if sstats[i, 4] > 50]
            if regions:
                sign_dets = sd.detect_in_regions(imC, regions, conf=args.sign_conf)

        if not pole_ccs and not tl_ccs and not sign_dets and not gp_dets:
            continue
        rho_t = rho[t]

        lift_vertical(pole_ccs, rho_t, R, C, args.pole_zmax, pole_obs)
        lift_vertical(tl_ccs, rho_t, R, C, args.pole_zmax, tl_obs)

        for d in sign_dets:
            o = lift_box(d["box"], rho_t, R, C, args.sign_zmax)
            if o is not None:
                o["name"] = d["name"]
                sign_obs.append(o)

        for d in gp_dets:
            o = lift_box(d["box"], rho_t, R, C, args.gp_zmax)
            if o is not None:
                gp_obs.append(o)

    print(f"[static] coarse obs: {len(pole_obs)} pole, {len(sign_obs)} sign, "
          f"{len(tl_obs)} traffic-light, {len(gp_obs)} guidepost")

    # ---- cluster + triangulate
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
            # metric size from per-view pixel height at the triangulated depth
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
                # "stable": the same class must dominate the cluster's observations
                if names.count(top) < stable_frac * len(names):
                    continue
                rec["name"] = top
                rec["agree"] = names.count(top) / len(names)
            out.append(rec)
        out.sort(key=lambda r: r["center"][2])
        return out

    poles = build(pole_obs, args.pole_eps, args.pole_min_obs, "pole")
    signs = build(sign_obs, args.sign_eps, args.sign_min_obs, "sign", args.sign_stable)
    tlights = build(tl_obs, args.pole_eps, args.tl_min_obs, "light")
    guideposts = build(gp_obs, args.gp_eps, args.gp_min_obs, "guidepost")
    poles = [p for p in poles if 1.0 < p["size"] < 14.0]     # plausible pole height
    tlights = [t for t in tlights if 0.3 < t["size"] < 6.0]  # plausible light-head height
    guideposts = [g for g in guideposts if 0.4 < g["size"] < 2.0]   # Leitpfosten ~1-1.5 m
    print(f"[static] {len(poles)} poles, {len(signs)} signs, {len(tlights)} traffic lights, "
          f"{len(guideposts)} guideposts after cluster+triangulate")
    for k, s in enumerate(signs):
        c = s["center"]
        print(f"   sign #{k} {s['name']:<24} X={c[0]:6.1f} Z={c[2]:6.1f}m  "
              f"~{s['size']:.2f}m  obs={s['n_frames']} agree={s['agree']:.2f} res={s['res_m']:.2f}m")

    # ---- BEV (aspect auto: lateral spread is tiny vs the 400 m drive)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 13))
    track = poses[:end, :3, 3]
    ax.plot(track[:, 0], track[:, 2], "0.6", lw=1.2, label="camera track")
    if model is not None:
        for nm, key in (("left", "left"), ("right", "right")):
            B = np.array(model[key])
            ax.plot(B[:, 0], B[:, 2], color=np.array(BORDER[nm][::-1]) / 255, lw=1.0)
    if poles:
        P = np.array([p["center"] for p in poles])
        ax.scatter(P[:, 0], P[:, 2], s=22, c="darkorange", marker="|", label=f"poles ({len(poles)})")
    if tlights:
        T = np.array([t["center"] for t in tlights])
        ax.scatter(T[:, 0], T[:, 2], s=55, c="green", marker="P",
                   label=f"traffic lights ({len(tlights)})")
    if guideposts:
        G = np.array([g["center"] for g in guideposts])
        ax.scatter(G[:, 0], G[:, 2], s=28, c="black", marker="d",
                   label=f"guideposts ({len(guideposts)})")
    for s in signs:
        c = s["center"]
        ax.scatter([c[0]], [c[2]], s=70, c="red", marker="s")
        ax.annotate(sign_label(s["name"]), (c[0], c[2]),
                    color="red", fontsize=8, xytext=(4, 0), textcoords="offset points")
    ax.scatter([], [], c="red", marker="s", label=f"traffic signs ({len(signs)})")
    ax.set_xlabel("X world (m)"); ax.set_ylabel("Z world (m)")
    ax.set_title("Static map: poles, signs, traffic lights, guideposts (triangulated)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.savefig(args.out_dir / "poles_signs_bev.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- RGB video: reproject every object into every frame
    writer = cv2.VideoWriter(str(args.out_dir / "poles_signs.mp4"),
                             cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (W, H))
    for t in tqdm(range(args.start, end), desc="render"):
        im = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        Rinv = R.T
        # poles: vertical stick of estimated height through the triangulated centre
        for p in poles:
            c, h = p["center"], p["size"]
            seg = project_pts(np.array([c - 0.5 * h * UP, c + 0.5 * h * UP]),
                              Rinv, C, fx, fy, cx, cy)
            if seg is not None:
                cv2.line(im, tuple(seg[0]), tuple(seg[1]), ORANGE, 3, cv2.LINE_AA)
                cv2.circle(im, tuple(seg[1]), 3, ORANGE, -1, cv2.LINE_AA)   # base
        # traffic lights: green upright housing (taller than wide) through the centre
        for tl in tlights:
            c, h = tl["center"], tl["size"]
            box = project_pts(np.array([c - 0.5 * h * UP, c + 0.5 * h * UP]),
                              Rinv, C, fx, fy, cx, cy)
            if box is not None:
                Zc = float((c - C) @ R[:, 2])
                wpx = max(2, int(0.25 * h / Zc * fy))
                (u0, v0), (u1, v1) = box[0], box[1]
                cv2.rectangle(im, (u1 - wpx, v1), (u1 + wpx, v0), GREEN, 2, cv2.LINE_AA)
        # guideposts (Leitpfosten): short white vertical stick + base dot
        for g in guideposts:
            c, h = g["center"], g["size"]
            seg = project_pts(np.array([c - 0.5 * h * UP, c + 0.5 * h * UP]),
                              Rinv, C, fx, fy, cx, cy)
            if seg is not None:
                cv2.line(im, tuple(seg[0]), tuple(seg[1]), WHITE, 3, cv2.LINE_AA)
                cv2.circle(im, tuple(seg[1]), 3, (0, 0, 0), -1, cv2.LINE_AA)   # base
        # signs: red billboard square (facing camera) + recognised class tag
        for s in signs:
            c, sz = s["center"], max(s["size"], 0.6)
            px = project_pts(c[None], Rinv, C, fx, fy, cx, cy)
            if px is None:
                continue
            Zc = float((c - C) @ R[:, 2])
            rpx = int(0.5 * sz / Zc * fy)
            u, v = int(px[0, 0]), int(px[0, 1])
            cv2.rectangle(im, (u - rpx, v - rpx), (u + rpx, v + rpx), RED, 2, cv2.LINE_AA)
            cv2.putText(im, sign_label(s["name"]), (u - rpx, v - rpx - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
        writer.write(im)
    writer.release()

    json.dump({"poles": [{"center": p["center"].tolist(), "height": p["size"],
                          "n_frames": p["n_frames"], "res_m": p["res_m"]} for p in poles],
               "signs": [{"name": s["name"], "center": s["center"].tolist(),
                          "size": s["size"], "n_frames": s["n_frames"],
                          "agree": s["agree"], "res_m": s["res_m"]} for s in signs],
               "traffic_lights": [{"center": t["center"].tolist(), "height": t["size"],
                                   "n_frames": t["n_frames"], "res_m": t["res_m"]}
                                  for t in tlights],
               "guideposts": [{"center": g["center"].tolist(), "height": g["size"],
                               "n_frames": g["n_frames"], "res_m": g["res_m"]}
                              for g in guideposts]},
              open(args.out_dir / "poles_signs_world.json", "w"), indent=1)
    print(f"[static] wrote {args.out_dir}/ : poles_signs_bev.png, poles_signs.mp4, "
          f"poles_signs_world.json")


if __name__ == "__main__":
    main()
