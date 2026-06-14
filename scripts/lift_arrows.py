"""Step 1 of the 3D static-environment thread: detect + lift + render straight-arrow footprints.

For every frame:
  1. Detect straight-ahead arrows (RLMD road_marking: straight_arrow + left/right_straight_arrow)
     -> per-instance pixel masks.
  2. Fit the LOCAL road plane in the cam0 frame by RANSAC on gray-stereo (SGBM image_0/image_1)
     3D points inside a road ROI. Stereo is used ONLY to find the plane -- NOT to get depth at
     the low-texture arrow pixels (which is the noisy part we want to avoid).
  3. Lift each arrow-mask pixel onto that plane by ray-plane intersection -> metric 3D in cam0,
     then -> world (first-camera) frame via the estimated pose. This is the "road-plane projection".
  4. Render: a world-frame BEV scatter of all footprints (same physical arrow from different
     frames should land on the same spot if lift+pose are sound) + per-frame RGB overlays.

No cross-frame clustering or template fitting yet -- that's the next step. This milestone just
verifies the lift is metric and consistent. Frame conventions match build_lane_map.py.
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
COLORS = {"straight_arrow": (0, 230, 0), "left_straight_arrow": (0, 200, 255),
          "right_straight_arrow": (255, 160, 0)}


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


def make_sgbm(num_disp=128, block=7):
    return cv2.StereoSGBM.create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def fit_plane_ransac(P, thresh=0.06, iters=250, seed=0):
    """Fit n.X = p to 3D points P (N,3). Returns (n unit, p, inlier_mask) or None."""
    n_pts = len(P)
    if n_pts < 50:
        return None
    rng = np.random.default_rng(seed)
    best = np.zeros(n_pts, bool)
    for _ in range(iters):
        i = rng.choice(n_pts, 3, replace=False)
        a, b, c = P[i]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            continue
        nrm /= ln
        d = nrm @ a
        inl = np.abs(P @ nrm - d) < thresh
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 50:
        return None
    # SVD refit on inliers
    Q = P[best]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    nrm = Vt[-1]
    nrm /= np.linalg.norm(nrm)
    if nrm[1] < 0:                     # orient so normal points "down" (cam0 Y-down -> +Y is road)
        nrm = -nrm
    p = nrm @ c
    return nrm, float(p), best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/04"))
    ap.add_argument("--poses", type=Path,
                    default=Path("outputs/pocs/orb_slam3/CameraTrajectory.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/static_env/arrows"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--z-max", type=float, default=45.0, help="trust road plane / lift to here (m)")
    ap.add_argument("--subsample", type=int, default=2, help="keep every Nth arrow-mask pixel")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    color = load_sequence(args.seq_dir, camera="image_2")
    gleft = load_sequence(args.seq_dir, camera="image_0")
    gright = load_sequence(args.seq_dir, camera="image_1")
    W, H, N = color.width, color.height, color.n_frames
    end = N if args.end < 0 else min(args.end, N)

    calib = read_calib(args.seq_dir / "calib.txt")
    K = calib["P2"][:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    baseline = float((calib["P0"][0, 3] - calib["P1"][0, 3]) / fx)
    poses = read_poses(args.poses)

    # road ROI for plane fit: lower-central trapezoid (avoid sky/buildings)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    roi = (vv > 0.55 * H) & (np.abs(uu - cx) < 0.40 * W)

    sgbm = make_sgbm()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out_dir / "arrows_lift.mp4"),
                             cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (W, H))
    footprints = []          # list of dicts: frame, cls, world points (M,3)
    n_arrow_frames = 0

    for t in tqdm(range(args.start, end), desc="lift arrows"):
        imC = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        dets = [d for d in rm.detect(imC, conf=args.conf) if d["name"] in ARROW_CLASSES]
        if dets:
            n_arrow_frames += 1
            gL = cv2.imread(str(gleft.image_paths[t]), cv2.IMREAD_GRAYSCALE)
            gR = cv2.imread(str(gright.image_paths[t]), cv2.IMREAD_GRAYSCALE)
            disp = sgbm.compute(gL, gR).astype(np.float32) / 16.0
            ys, xs = np.where(roi & (disp > 0))
            Z = fx * baseline / disp[ys, xs]
            keep = (Z > 3) & (Z < args.z_max)
            ys, xs, Z = ys[keep], xs[keep], Z[keep]
            Proad = np.stack([(xs - cx) * Z / fx, (ys - cy) * Z / fy, Z], 1)
            plane = fit_plane_ransac(Proad)
        else:
            plane = None

        Tt = poses[t]; R, C = Tt[:3, :3], Tt[:3, 3]
        if plane is not None:
            nrm, p, _ = plane
            for d in dets:
                ys, xs = np.where(d["mask"])
                if args.subsample > 1:
                    ys, xs = ys[::args.subsample], xs[::args.subsample]
                if len(xs) == 0:
                    continue
                dirs = (Kinv @ np.stack([xs, ys, np.ones_like(xs)], 0)).T   # (M,3) cam rays
                denom = dirs @ nrm
                ok = np.abs(denom) > 1e-6
                tt = np.where(ok, p / denom, np.nan)
                ok &= (tt > 0) & (tt * dirs[:, 2] < args.z_max)
                Xc = dirs[ok] * tt[ok, None]                                # cam0 metric
                if len(Xc) == 0:
                    continue
                Xw = Xc @ R.T + C                                           # world
                footprints.append({"frame": t, "cls": d["name"], "pts": Xw})
                # draw mask outline on the RGB overlay (verify what we lifted)
                col = COLORS[d["name"]]
                m = (d["mask"] * 255).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(imC, cnts, -1, col, 2, cv2.LINE_AA)
        writer.write(imC)
    writer.release()

    print(f"[arrows] {n_arrow_frames} frames with arrows, {len(footprints)} lifted footprints")
    if footprints:
        allpts = np.concatenate([f["pts"] for f in footprints], 0)
        np.save(args.out_dir / "footprints_world.npy",
                np.array([{"frame": f["frame"], "cls": f["cls"], "pts": f["pts"]}
                          for f in footprints], dtype=object), allow_pickle=True)
        # BEV world X-Z scatter, coloured by class, with camera track
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 14))
        track = poses[:end, :3, 3]
        ax.plot(track[:, 0], track[:, 2], "0.6", lw=1.0, label="camera track")
        for cls in ARROW_CLASSES:
            P = [f["pts"] for f in footprints if f["cls"] == cls]
            if P:
                P = np.concatenate(P, 0)
                ax.scatter(P[:, 0], P[:, 2], s=2, c=[np.array(COLORS[cls][::-1]) / 255],
                           label=cls)
        ax.set_aspect("equal"); ax.set_xlabel("X world (m)"); ax.set_ylabel("Z world (m)")
        ax.set_title("Lifted straight-arrow footprints (BEV, world frame)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(args.out_dir / "arrows_bev.png", dpi=130, bbox_inches="tight")
        print(f"[arrows] wrote {args.out_dir/'arrows_bev.png'} and arrows_lift.mp4")


if __name__ == "__main__":
    main()
