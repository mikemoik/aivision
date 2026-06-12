"""Unproject the fused rho video into a voxel-downsampled dense point cloud.

Reads the 16-bit FFV1 rho video (pixel 0 = invalid, rho = (pixel-1)/65534),
backprojects every valid pixel in rho-space (P_cam = ray / rho), transforms
with the ORB-SLAM3 cam-to-world poses, colourises from image_2, and
voxel-downsamples to a PLY.

Recovered from session 68870bb1 (2026-06-12); Phase 1B defaults the voxel
size to 10 cm (was 5 cm).

Phase 2 (semantic cleanup): pass ``--perception <cache_dir>`` to AND a per-frame
ignore mask of dynamic + sky classes (Mask2Former-Vistas) into the depth
validity mask before backprojection. Kills moving-car smears and sky haze at
the source. Frame indexing of the cache aligns 1:1 with the sequence because
both sort the same image_2 PNGs by filename.
"""

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from aivision.kitti import load_sequence
from aivision.perception import (
    CacheManifest,
    MANIFEST_NAME,
    id2label_from_manifest,
    load_frame_entry,
)
from aivision.pocs import common

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")

RHO_MIN = 0.005  # drop rho < 0.005 -> drop Z > 200 m (mostly sky)
RHO_MAX = 1.5    # drop rho > 1.5   -> drop Z < 0.67 m (near-camera junk)

# Vistas classes to drop pre-unprojection: everything that moves (animals,
# humans, all vehicles), the sky, and the ego vehicle / car mount. Resolved
# to integer ids against the cache's seg model so a renamed class surfaces as
# a startup error rather than a silent miss. Static structure we KEEP: road,
# sidewalk, building, wall, fence, curb, guard-rail, barrier, poles, traffic
# signs/lights, vegetation, terrain, etc.
DROP_CLASS_NAMES: frozenset[str] = frozenset({
    "Bird", "Ground Animal",
    "Person", "Bicyclist", "Motorcyclist", "Other Rider",
    "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle",
    "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow",
    "Sky", "Car Mount", "Ego Vehicle",
})


def _resolve_drop_ids(perception_dir: Path) -> tuple[set[int], dict[int, str]]:
    """Resolve DROP_CLASS_NAMES against the cache's seg model id2label."""
    manifest = CacheManifest.load(perception_dir / MANIFEST_NAME)
    if not manifest.populated.get("semantic", False):
        raise SystemExit(
            f"perception cache {perception_dir} has no semantic field; "
            "rerun `aivision perception-cache ... --segment`"
        )
    id2label = id2label_from_manifest(manifest)
    label2id = {v: k for k, v in id2label.items()}
    missing = sorted(DROP_CLASS_NAMES - set(label2id))
    if missing:
        raise SystemExit(f"drop classes not in seg model id2label: {missing}")
    drop_ids = {label2id[n] for n in DROP_CLASS_NAMES}
    return drop_ids, id2label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=SRC / "inv_depth_fused_mono_rho_16bit.mkv")
    ap.add_argument("--out", type=Path, default=SRC / "dense_cloud.ply")
    ap.add_argument("--voxel", type=float, default=0.10, help="voxel size in metres")
    ap.add_argument(
        "--perception", type=Path, default=None,
        help="perception cache dir; AND a dynamic+sky ignore mask into the depth mask",
    )
    args = ap.parse_args()

    drop_ids: set[int] = set()
    if args.perception is not None:
        drop_ids, id2label = _resolve_drop_ids(args.perception)
        dropped_names = ", ".join(sorted(id2label[i] for i in drop_ids))
        print(f"[semantic]  masking {len(drop_ids)} classes: {dropped_names}")

    color_seq = load_sequence(SEQ_DIR, camera="image_2")
    W, H = color_seq.width, color_seq.height
    K = color_seq.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    N = color_seq.n_frames

    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")
    assert len(poses) == N

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rx = (uu - cx) / fx
    ry = (vv - cy) / fy

    # Stream 16-bit depth frames out of the FFV1 video as raw gray16le
    proc = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(args.video),
        "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
    ], stdout=subprocess.PIPE)
    frame_bytes = W * H * 2

    all_points = []
    all_colors = []
    n_raw = n_kept = n_sem_dropped = 0
    t0 = time.time()
    for i in range(N):
        raw = proc.stdout.read(frame_bytes)
        pix = np.frombuffer(raw, dtype=np.uint16).reshape(H, W)
        rho = (pix.astype(np.float32) - 1.0) / 65534.0
        valid = (pix > 0) & (rho >= RHO_MIN) & (rho <= RHO_MAX)
        n_raw += int((pix > 0).sum())

        # Phase 2: drop dynamic + sky pixels via the cached semantic mask.
        if drop_ids:
            sem = load_frame_entry(args.perception, i)["semantic"]
            if sem.shape != (H, W):
                sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
            keep = ~np.isin(sem, list(drop_ids))
            n_sem_dropped += int((valid & ~keep).sum())
            valid &= keep

        if not valid.any():
            continue

        # rho-space backprojection: P_cam = ray / rho — bounded for rho > RHO_MIN
        inv_rho = 1.0 / rho[valid]
        pts_cam = np.stack([rx[valid] * inv_rho, ry[valid] * inv_rho, inv_rho], axis=1)

        R_cw = poses[i, :3, :3]
        t_cw = poses[i, :3, 3]
        pts_world = (R_cw @ pts_cam.T).T + t_cw

        # Colour from image_2 (image_0/image_2 K match; their few-pixel
        # extrinsic offset is invisible at voxel resolution).
        bgr = cv2.imread(str(color_seq.image_paths[i]), cv2.IMREAD_COLOR)
        rgb = bgr[valid][:, ::-1].astype(np.float32) / 255.0

        all_points.append(pts_world.astype(np.float32))
        all_colors.append(rgb)
        n_kept += int(valid.sum())

    proc.wait()
    print(f"[unproject] {N} frames in {time.time()-t0:.1f}s")
    print(f"[stats]     raw valid={n_raw/1e6:.1f}M  kept (RHO {RHO_MIN}..{RHO_MAX})={n_kept/1e6:.1f}M")
    if drop_ids:
        print(f"[semantic]  dropped {n_sem_dropped/1e6:.1f}M dynamic/sky pixels pre-voxel")

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    print(f"[concat]    {len(points)/1e6:.1f}M points, {points.nbytes/1e9:.1f} GB")

    t0 = time.time()
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pc_down = pc.voxel_down_sample(voxel_size=args.voxel)
    print(f"[voxel]     {len(pc.points)/1e6:.1f}M -> {len(pc_down.points)/1e6:.1f}M points "
          f"at {args.voxel*100:.0f} cm voxel ({time.time()-t0:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out), pc_down, write_ascii=False, compressed=True)
    print(f"[write]     {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
