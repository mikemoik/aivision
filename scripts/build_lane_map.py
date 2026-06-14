"""Time-integrated 3D ego-lane map in the first-camera world frame.

Takes the per-frame EgoLanes output (ego-left + ego-right lane borders), lifts each
lane pixel to a metric 3D point via COLOUR STEREO depth (image_2 vs image_3), then
transforms it by the estimated camera pose into a single fixed world frame (the first
camera's frame), accumulating all frames into one lane point cloud.

seq04 has no colour-right camera, so depth comes from the GRAY stereo pair
(image_0 / image_1, baseline ~0.537 m). EgoLanes masks live on colour-left image_2,
but image_2 and gray-left image_0 are both rectified left cameras with IDENTICAL K
(P0 & P2 share fx,fy,cx,cy) and differ only by a ~6.6 cm baseline, so an image_2 lane
pixel indexes the image_0-derived disparity map to within a few px (a few cm of depth
error at most) -- negligible vs the stereo noise and the ~0.19 m pose ATE. We therefore
work directly in the cam0 frame the gray stereo + the poses already live in.

Geometry chain for a lane pixel (u,v) in frame t:
  1. stereo: disparity d from SGBM(image_0, image_1) -> Z = fx*B / d   (B = cam0-cam1 baseline)
  2. back-project in the cam0 frame: X_cam0 = ((u-cx)Z/fx, (v-cy)Z/fy, Z)
  3. cam0 -> world via the estimated pose Tt (cam0->world, first frame = identity):
     X_world = Rt @ X_cam0 + Ct
Poses default to the ORB-SLAM3 stereo trajectory (metric, ATE ~0.19 m on seq04); the
first pose is identity so "world" IS the first camera frame, as requested.

Outputs (under --out-dir): lane_map.ply (open3d; ego-left blue, ego-right pink) and
lane_map_bev.png (bird's-eye X-Z scatter + camera track).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence
from aivision import ego_lanes as el

# ego-left / ego-right only (the borders the user asked for first); RGB for PLY/plot
CLASS_RGB = {0: (0.10, 0.30, 1.00), 1: (1.00, 0.30, 0.75)}   # ego-left blue, ego-right pink
CLASS_NAME = {0: "ego_left", 1: "ego_right"}


def read_calib(calib_path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in calib_path.read_text().strip().splitlines():
        k, *vals = line.replace(":", "").split()
        out[k] = np.array(vals, np.float64).reshape(3, 4)
    return out


def read_poses(path: Path) -> np.ndarray:
    """KITTI 3x4 cam->world rows -> (N,4,4)."""
    raw = np.loadtxt(path, dtype=np.float64)
    n = raw.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :4] = raw.reshape(n, 3, 4)
    return T


def make_sgbm(num_disp: int = 128, block: int = 7) -> cv2.StereoSGBM:
    return cv2.StereoSGBM.create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/04"))
    ap.add_argument("--poses", type=Path,
                    default=Path("outputs/pocs/orb_slam3/CameraTrajectory.txt"),
                    help="estimated cam0->world poses (KITTI 3x4 per row)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/ego_lanes/lane_map"))
    ap.add_argument("--thr", type=float, default=el.DEFAULT_THR)
    ap.add_argument("--z-min", type=float, default=3.0)
    ap.add_argument("--z-max", type=float, default=40.0, help="trust stereo depth to here (m)")
    ap.add_argument("--subsample", type=int, default=3, help="keep every Nth lane pixel")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    color = load_sequence(args.seq_dir, camera="image_2")  # colour left (EgoLanes masks)
    gleft = load_sequence(args.seq_dir, camera="image_0")  # gray left  (stereo)
    gright = load_sequence(args.seq_dir, camera="image_1")  # gray right (stereo)
    W, H, N = color.width, color.height, color.n_frames
    end = N if args.end < 0 else min(args.end, N)

    calib = read_calib(args.seq_dir / "calib.txt")
    K = calib["P2"][:3, :3]                                # == P0 K (shared rectified intrinsics)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    baseline = float((calib["P0"][0, 3] - calib["P1"][0, 3]) / fx)  # cam0-cam1 (~0.537 m)
    print(f"[lanemap] fx={fx:.1f} gray-stereo baseline={baseline:.4f} m")

    poses = read_poses(args.poses)
    if len(poses) < end:
        raise SystemExit(f"poses ({len(poses)}) fewer than frames to process ({end})")

    sgbm = make_sgbm()
    pts = {0: [], 1: []}          # accumulated world points per class
    valid_px = total_px = 0
    cam_track = []

    for t in tqdm(range(args.start, end), desc="lane map"):
        imC = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)   # for EgoLanes
        gL = cv2.imread(str(gleft.image_paths[t]), cv2.IMREAD_GRAYSCALE)
        gR = cv2.imread(str(gright.image_paths[t]), cv2.IMREAD_GRAYSCALE)
        # disparity (px) in the gray-left (cam0) frame
        disp = sgbm.compute(gL, gR).astype(np.float32) / 16.0

        Tt = poses[t]
        Rt, Ct = Tt[:3, :3], Tt[:3, 3]
        cam_track.append(Ct.copy())

        dets = {d["cls"]: d["mask"] for d in el.detect(imC, thr=args.thr)}
        for cls in (0, 1):
            mask = dets.get(cls)
            if mask is None:
                continue
            ys, xs = np.where(mask)
            if args.subsample > 1:
                ys, xs = ys[::args.subsample], xs[::args.subsample]
            total_px += len(xs)
            d = disp[ys, xs]
            ok = d > 0
            Z = np.full(len(xs), np.nan)
            Z[ok] = fx * baseline / d[ok]
            good = ok & (Z > args.z_min) & (Z < args.z_max)
            if not good.any():
                continue
            valid_px += int(good.sum())
            u, v, z = xs[good], ys[good], Z[good]
            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Xc0 = np.stack([X, Y, z], 1)             # cam0 frame
            Xw = Xc0 @ Rt.T + Ct                     # world (first-camera) frame
            pts[cls].append(Xw)

    cam_track = np.array(cam_track)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lanemap] stereo validity at lane pixels: {valid_px}/{total_px} "
          f"({100*valid_px/max(total_px,1):.1f}%)")

    # --- write PLY (open3d) ---
    import open3d as o3d
    all_xyz, all_rgb = [], []
    for cls in (0, 1):
        if not pts[cls]:
            continue
        P = np.concatenate(pts[cls], 0)
        all_xyz.append(P)
        all_rgb.append(np.tile(CLASS_RGB[cls], (len(P), 1)))
        print(f"    {CLASS_NAME[cls]:9s} {len(P)} points")
    if all_xyz:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.concatenate(all_xyz, 0))
        pc.colors = o3d.utility.Vector3dVector(np.concatenate(all_rgb, 0))
        ply = args.out_dir / "lane_map.ply"
        o3d.io.write_point_cloud(str(ply), pc)
        print(f"[lanemap] wrote {ply}")

    # --- BEV top-down (X right, Z forward) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axp = plt.subplots(figsize=(9, 11))
    for cls in (0, 1):
        if pts[cls]:
            P = np.concatenate(pts[cls], 0)
            axp.scatter(P[:, 0], P[:, 2], s=1.0, c=[CLASS_RGB[cls]],
                        label=CLASS_NAME[cls])
    axp.plot(cam_track[:, 0], cam_track[:, 2], "k-", lw=1.2, label="camera track")
    axp.set_aspect("equal"); axp.set_xlabel("X world (m)"); axp.set_ylabel("Z world (m)")
    axp.set_title("Time-integrated ego-lane borders (BEV, first-camera frame)")
    axp.legend(); axp.grid(alpha=0.3)
    bev = args.out_dir / "lane_map_bev.png"
    fig.savefig(bev, dpi=130, bbox_inches="tight")
    print(f"[lanemap] wrote {bev}")


if __name__ == "__main__":
    main()
