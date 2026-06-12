"""Render the dense-cloud triptych video: source / source+cloud / cloud-only.

Splats the dense cloud back onto each trajectory frame (painter's algorithm,
far-to-near) with a per-frame Z clip. Recovered from session 68870bb1
(2026-06-12) where it ran as an inline command.
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from aivision.kitti import load_sequence
from aivision.pocs import common

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")

Z_FAR = 50.0   # per-frame far clip [m] — rho >= 1/50
Z_NEAR = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=Path, default=SRC / "dense_cloud.ply")
    ap.add_argument("--out", type=Path, default=SRC / "triptych_dense.mp4")
    args = ap.parse_args()

    tmp = SRC / "_dense_tri_frames"

    color_seq = load_sequence(SEQ_DIR, camera="image_2")
    W, H = color_seq.width, color_seq.height
    K = color_seq.K
    fx = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    N = color_seq.n_frames

    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")
    R_wc_all = np.transpose(poses[:, :3, :3], (0, 2, 1))
    t_wc_all = -np.einsum("nij,nj->ni", R_wc_all, poses[:, :3, 3])

    t0 = time.time()
    pc = o3d.io.read_point_cloud(str(args.ply))
    points = np.asarray(pc.points, dtype=np.float32)
    colors = np.asarray(pc.colors, dtype=np.float32)
    print(f"[load ply] {len(points)/1e6:.1f}M points in {time.time()-t0:.1f}s")
    print(f"[clip] keeping per-frame {Z_NEAR} m < Z < {Z_FAR} m")

    b1 = N // 3
    b2 = 2 * N // 3

    def mode_for(i):
        if i < b1:
            return ("source", False, "1/3  source (image_2)")
        if i < b2:
            return ("source", True, f"2/3  source + dense cloud (Z<{Z_FAR:.0f} m)")
        return ("black", True, f"3/3  dense cloud only (Z<{Z_FAR:.0f} m)")

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    t0 = time.time()
    for i in range(N):
        bg, cloud, txt = mode_for(i)
        if bg == "source":
            img = cv2.imread(str(color_seq.image_paths[i]), cv2.IMREAD_COLOR)
        else:
            img = np.zeros((H, W, 3), dtype=np.uint8)

        if cloud:
            R_wc, t_wc = R_wc_all[i], t_wc_all[i]
            x_cam = (R_wc @ points.T).T + t_wc
            z = x_cam[:, 2]
            in_front = (z > Z_NEAR) & (z < Z_FAR)
            xf = x_cam[in_front]
            zc = z[in_front]
            u = fx * xf[:, 0] / zc + cx
            v = fx * xf[:, 1] / zc + cy
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u = u[in_img].astype(np.int32)
            v = v[in_img].astype(np.int32)
            z_kept = zc[in_img]
            rgb_kept = colors[in_front][in_img]
            order = np.argsort(-z_kept, kind="stable")  # painter's: far first
            bgr_kept = (rgb_kept[order, ::-1] * 255).clip(0, 255).astype(np.uint8)
            img.reshape(-1, 3)[v[order] * W + u[order]] = bgr_kept

        cv2.putText(img, f"{txt}    f={i:03d}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(tmp / f"f_{i:06d}.png"), img)
    print(f"[render] {N} frames in {time.time()-t0:.1f}s")

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "10",
        "-i", str(tmp / "f_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(args.out),
    ], check=True)
    shutil.rmtree(tmp)
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
