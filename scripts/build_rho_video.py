"""Build the fused inverse-depth (rho) video for the ORB-SLAM3 dense POC.

Thin CLI over the shared builder ``aivision.map.steps_depth.build_fused_rho``
(SGBM -> guided-filter smoothing -> ±WINDOW pose-warped rho-median fusion ->
Depth Pro mono fill via per-frame affine -> 16-bit FFV1 gray16le MKV).

Encoding: pixel 0 = invalid, pixel in [1, 65535] -> rho = (pixel - 1) / 65534.

Recovered from session 68870bb1 (2026-06-12); the core is now the map-gen T2
depth step's lib fn — this script just wires legacy paths to it.
"""

import argparse
from pathlib import Path

from aivision.kitti import load_sequence
from aivision.map.steps_depth import build_fused_rho
from aivision.pocs import common

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")

FX = 707.0912
BASELINE = 0.537165


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=SEQ_DIR, help="KITTI sequence dir")
    ap.add_argument("--src", type=Path, default=SRC,
                    help="dir holding CameraTrajectory.txt; rho outputs land here")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-stereo", type=Path, default=None,
                    help="sibling stereo-ONLY rho stream (no mono-fill). "
                         "Default: <out stem>_stereo_16bit.mkv")
    ap.add_argument("--no-guided", action="store_true", help="skip Phase 1A guided filter")
    args = ap.parse_args()

    src: Path = args.src
    out_path: Path = args.out or src / "inv_depth_fused_mono_rho_16bit.mkv"
    out_stereo: Path = args.out_stereo or out_path.with_name(
        out_path.stem.replace("_16bit", "").replace("_mono", "") + "_stereo_16bit.mkv")

    left = load_sequence(args.seq_dir, camera="image_0")
    right = load_sequence(args.seq_dir, camera="image_1")
    poses = common.load_kitti_trajectory(src / "CameraTrajectory.txt")

    build_fused_rho(
        left.image_paths, right.image_paths, poses,
        FX, float(left.K[0, 2]), float(left.K[1, 2]), BASELINE,
        out_path, out_stereo, guided=not args.no_guided,
    )


if __name__ == "__main__":
    main()
