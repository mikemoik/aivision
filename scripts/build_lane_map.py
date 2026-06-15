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

import numpy as np

from aivision import ego_lanes as el
from aivision.map.steps_lanes import CLASS_NAME, CLASS_RGB, build_lane_cloud


def read_poses(path: Path) -> np.ndarray:
    """KITTI 3x4 cam->world rows -> (N,4,4)."""
    raw = np.loadtxt(path, dtype=np.float64)
    n = raw.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :4] = raw.reshape(n, 3, 4)
    return T


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
    args = ap.parse_args()

    poses = read_poses(args.poses)
    cloud = build_lane_cloud(args.seq_dir, poses, thr=args.thr, z_min=args.z_min,
                             z_max=args.z_max, subsample=args.subsample)
    pts, cam_track = cloud["pts"], cloud["cam_track"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- write PLY (open3d) ---
    import open3d as o3d
    all_xyz, all_rgb = [], []
    for cls in (0, 1):
        if len(pts[cls]) == 0:
            continue
        P = pts[cls]
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
        if len(pts[cls]):
            P = pts[cls]
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
