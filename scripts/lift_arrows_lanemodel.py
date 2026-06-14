"""Step 1b of the 3D static-environment thread: lift straight-arrow markings onto the
ground plane GIVEN BY THE LANE MODEL (not per-frame stereo), then estimate one 3D
pose + shape per physical arrow and render it.

Why the lane model and not stereo: the stereo-SGBM road plane (lift_arrows.py) is fit
per frame from low-texture asphalt -> noisy and recomputed every frame. The fitted ego-lane
model (lane_polylines_model.json: left/right borders as world-frame [X,Y,Z], time-integrated,
RANSAC-rejected, Whittaker-smoothed) already IS the road surface: the borders sit on it and
carry its height + tilt + grade. So we take the ground plane from the lane model and drop
stereo from this script entirely.

Pipeline:
  1. Detect straight-ahead arrows (straight_arrow + left/right_straight_arrow) per frame.
  2. Ground plane: select lane-model border points inside the forward window of THIS frame
     (camera-frame Z in [3, z_max]) and least-squares fit a LOCAL plane in the world frame.
     Local, not global -- over 400 m of grade/curve the road is not one plane.
  3. Lift each arrow-mask pixel: world ray from the camera centre intersected with that plane.
  4. Ego-lane gate: keep only arrows whose lifted centroid falls between the left/right
     borders (interpolated by forward Z) -> "straight arrows within the ego lane".
  5. Cluster footprints across all frames (union-find on world centroids) -> physical arrows.
  6. Per arrow: fit a 3D pose (centroid + heading, PCA on the plane, disambiguated/clamped to
     the lane tangent) and a straight-arrow glyph (shaft + head) sized to the measured extents.
  7. Render: world BEV (yellow glyphs over faint footprints + lane borders + track) and an RGB
     mp4 with the estimated arrow glyphs (yellow) reprojected into every frame + a blue
     ego-lane corridor fill out to the 50 m line of sight.

Frame conventions match render_lane_reprojection.py / build_lane_map.py (cam0->world poses,
rectified P2 K, render on colour-left image_2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence
from aivision import road_marking as rm

ARROW_CLASSES = {"straight_arrow", "left_straight_arrow", "right_straight_arrow"}
YELLOW = (0, 255, 255)        # BGR -- the arrows
CORRIDOR = (255, 150, 0)      # BGR -- the ego-lane blue corridor fill
BORDER = {"left": (255, 130, 30), "right": (190, 60, 255)}
ZNEAR = 5.0                   # m, near-plane clip for projection


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


def fit_plane(P: np.ndarray):
    """Least-squares plane n.X = p through world points P (>=3). Returns (n unit, p)."""
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    return n, float(n @ c)


def project_poly(Pw: np.ndarray, Rinv: np.ndarray, C: np.ndarray, fx, fy, cx, cy):
    """World polygon -> int pixel polygon, or None if any vertex is behind the near plane."""
    Pc = (Pw - C) @ Rinv.T
    if np.any(Pc[:, 2] <= ZNEAR):
        return None
    u = fx * Pc[:, 0] / Pc[:, 2] + cx
    v = fy * Pc[:, 1] / Pc[:, 2] + cy
    return np.round(np.stack([u, v], 1)).astype(np.int32)


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
    if abs(yaw) > 30.0:                              # PCA went sideways -> trust lane
        mj = np.array([1.0, 0.0]); yaw = 0.0
    mn = np.array([-mj[1], mj[0]])

    pa, pb = rel @ (f * mj[0] + r * mj[1]), rel @ (f * mn[0] + r * mn[1])
    L = float(np.percentile(pa, 95) - np.percentile(pa, 5))
    Wd = float(np.percentile(pb, 90) - np.percentile(pb, 10))
    L = float(np.clip(L, 1.5, 9.0)); Wd = float(np.clip(Wd, 0.25, 1.4))

    maj3 = f * mj[0] + r * mj[1]
    min3 = f * mn[0] + r * mn[1]
    g = arrow_glyph(L, Wd)
    glyph_w = c + g[:, 0:1] * maj3 + g[:, 1:2] * min3
    return {"centroid": c, "forward": maj3, "length": L, "width": Wd,
            "yaw_to_lane_deg": yaw, "glyph_world": glyph_w}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/04"))
    ap.add_argument("--poses", type=Path,
                    default=Path("outputs/pocs/orb_slam3/CameraTrajectory.txt"))
    ap.add_argument("--lane-model", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_polylines_model.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/static_env/arrows_lanemodel"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--z-max", type=float, default=45.0, help="lift / trust plane to here (m)")
    ap.add_argument("--los", type=float, default=50.0, help="ego-lane corridor line of sight (m)")
    ap.add_argument("--subsample", type=int, default=2, help="keep every Nth arrow-mask pixel")
    ap.add_argument("--lane-margin", type=float, default=0.4, help="ego-lane gate slack (m)")
    ap.add_argument("--cluster-eps", type=float, default=2.0, help="footprint clustering radius (m)")
    ap.add_argument("--min-obs", type=int, default=2, help="min frames per accepted arrow")
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

    model = json.loads(args.lane_model.read_text())
    Lb = np.array(model["left"], np.float64)        # (M,3) world, sorted by Z
    Rb = np.array(model["right"], np.float64)
    border_pts = np.vstack([Lb, Rb])
    Zc = Lb[:, 2]                                    # forward axis samples (monotone)
    centerline = 0.5 * (Lb + Rb)
    # forward-Z interpolators for the ego-lane gate (drive is ~straight: X is lateral)
    leftX = lambda z: np.interp(z, Zc, Lb[:, 0])
    rightX = lambda z: np.interp(z, Zc, Rb[:, 0])
    z_lo, z_hi = float(Zc.min()), float(Zc.max())

    def lane_tangent(z: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(Zc, z), 1, len(Zc) - 1))
        t = centerline[i] - centerline[i - 1]
        return t / (np.linalg.norm(t) + 1e-9)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: detect, fit per-frame ground plane from the lane model, lift, ego-lane gate
    footprints = []                                  # {frame, cls, pts(world), centroid}
    contours_by_frame: dict[int, list] = {}          # raw detection outlines for verification
    n_arrow_frames = 0
    for t in tqdm(range(args.start, end), desc="lift (lane-model plane)"):
        imC = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        dets = [d for d in rm.detect(imC, conf=args.conf) if d["name"] in ARROW_CLASSES]
        if not dets:
            continue
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        # local ground plane: lane-model border points inside this frame's forward window
        Pc = (border_pts - C) @ R.T
        win = (Pc[:, 2] > 3.0) & (Pc[:, 2] < args.z_max)
        if win.sum() < 6:
            continue
        n, p = fit_plane(border_pts[win])
        nC = n @ C
        n_arrow_frames += 1
        for d in dets:
            ys, xs = np.where(d["mask"])
            if args.subsample > 1:
                ys, xs = ys[::args.subsample], xs[::args.subsample]
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
            ok &= (Zcam > 3.0) & (Zcam < args.z_max)
            Xw = Xw[ok]
            if len(Xw) < 8:
                continue
            ctr = Xw.mean(0)
            cz = ctr[2]
            if not (z_lo <= cz <= z_hi):
                continue
            lx, rx = float(leftX(cz)), float(rightX(cz))
            lo, hi = min(lx, rx) - args.lane_margin, max(lx, rx) + args.lane_margin
            if not (lo <= ctr[0] <= hi):                                # ego-lane gate
                continue
            footprints.append({"frame": t, "cls": d["name"], "pts": Xw, "centroid": ctr})
            m = (d["mask"] * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_by_frame.setdefault(t, []).extend(cnts)

    print(f"[arrows] {n_arrow_frames} arrow frames, {len(footprints)} ego-lane footprints")
    if not footprints:
        print("[arrows] nothing in the ego lane -- aborting render")
        return

    # ---- Cluster footprints across frames -> physical arrows, fit pose + shape
    cents = np.array([f["centroid"] for f in footprints])
    clusters = union_find_clusters(cents, args.cluster_eps)
    arrows = []
    for idx in clusters:
        if len(idx) < args.min_obs:
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
    print(f"[arrows] {len(arrows)} physical arrows after clustering "
          f"(>= {args.min_obs} obs, eps {args.cluster_eps} m)")
    for k, a in enumerate(arrows):
        c = a["centroid"]
        print(f"   #{k} {a['cls']:<20} Z={c[2]:6.1f}m  L={a['length']:.2f} W={a['width']:.2f} "
              f"yaw={a['yaw_to_lane_deg']:+.1f}deg  obs={a['n_frames']}")

    # ---- BEV world scatter + glyphs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 14))
    track = poses[:end, :3, 3]
    ax.plot(track[:, 0], track[:, 2], "0.6", lw=1.0, label="camera track")
    ax.plot(Lb[:, 0], Lb[:, 2], color=np.array(BORDER["left"][::-1]) / 255, lw=1.2)
    ax.plot(Rb[:, 0], Rb[:, 2], color=np.array(BORDER["right"][::-1]) / 255, lw=1.2)
    allf = np.vstack([f["pts"] for f in footprints])
    ax.scatter(allf[:, 0], allf[:, 2], s=1, c="0.75", label="lifted footprints")
    for a in arrows:
        g = a["glyph_world"]
        poly = plt.Polygon(g[:, [0, 2]], closed=True, facecolor="gold",
                           edgecolor="darkgoldenrod", lw=1.3, alpha=0.85)
        ax.add_patch(poly)
    ax.scatter([], [], c="gold", marker="^", label="estimated arrows")
    ax.set_aspect("equal"); ax.set_xlabel("X world (m)"); ax.set_ylabel("Z world (m)")
    ax.set_title(f"Ego-lane straight arrows ({len(arrows)}) lifted via lane-model plane")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.savefig(args.out_dir / "arrows_bev.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- RGB video: blue ego-lane corridor + reprojected yellow arrow glyphs
    writer = cv2.VideoWriter(str(args.out_dir / "arrows_lift.mp4"),
                             cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (W, H))
    for t in tqdm(range(args.start, end), desc="render"):
        im = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        Rinv = R.T
        overlay = im.copy()

        # blue ego-lane corridor out to the line-of-sight horizon
        Pcl = (Lb - C) @ Rinv.T
        win = (Pcl[:, 2] > ZNEAR) & (Pcl[:, 2] < args.los)
        if win.sum() >= 2:
            order = np.argsort(Pcl[win, 2])
            li = np.where(win)[0][order]
            corridor = np.vstack([Lb[li], Rb[li][::-1]])
            poly = project_poly(corridor, Rinv, C, fx, fy, cx, cy)
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(overlay, [poly], CORRIDOR)
        im = cv2.addWeighted(overlay, 0.28, im, 0.72, 0)
        # border polylines (in-window portion) for a crisp corridor edge
        for name, B in (("left", Lb), ("right", Rb)):
            Pc = (B - C) @ Rinv.T
            wb = (Pc[:, 2] > ZNEAR) & (Pc[:, 2] < args.los)
            if wb.sum() >= 2:
                seg = B[np.where(wb)[0][np.argsort(Pc[wb, 2])]]
                poly = project_poly(seg, Rinv, C, fx, fy, cx, cy)
                if poly is not None and len(poly) >= 2:
                    cv2.polylines(im, [poly], False, BORDER[name], 2, cv2.LINE_AA)

        # estimated arrow glyphs (yellow), reprojected -> they stick to the road across frames
        ov2 = im.copy()
        drew = False
        for a in arrows:
            poly = project_poly(a["glyph_world"], Rinv, C, fx, fy, cx, cy)
            if poly is None or len(poly) < 3:
                continue
            cv2.fillPoly(ov2, [poly], YELLOW)
            drew = True
        if drew:
            im = cv2.addWeighted(ov2, 0.45, im, 0.55, 0)
        for a in arrows:
            poly = project_poly(a["glyph_world"], Rinv, C, fx, fy, cx, cy)
            if poly is not None and len(poly) >= 3:
                cv2.polylines(im, [poly], True, YELLOW, 2, cv2.LINE_AA)
        # faint raw detection outline (estimate-vs-detection check)
        for c in contours_by_frame.get(t, []):
            cv2.drawContours(im, [c], -1, (120, 120, 120), 1, cv2.LINE_AA)
        writer.write(im)
    writer.release()

    # ---- persist the estimated static-arrow map
    np.save(args.out_dir / "footprints_world.npy",
            np.array([{"frame": f["frame"], "cls": f["cls"], "pts": f["pts"]}
                      for f in footprints], dtype=object), allow_pickle=True)
    json.dump([{"cls": a["cls"], "centroid": a["centroid"].tolist(),
                "forward": a["forward"].tolist(), "length": a["length"],
                "width": a["width"], "yaw_to_lane_deg": a["yaw_to_lane_deg"],
                "n_frames": a["n_frames"], "glyph_world": a["glyph_world"].tolist()}
               for a in arrows],
              open(args.out_dir / "arrows_world.json", "w"), indent=1)
    print(f"[arrows] wrote {args.out_dir}/ : arrows_bev.png, arrows_lift.mp4, "
          f"arrows_world.json, footprints_world.npy")


if __name__ == "__main__":
    main()
