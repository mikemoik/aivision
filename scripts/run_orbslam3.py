"""Run ORB-SLAM3 stereo on a KITTI sequence -> CameraTrajectory.txt in --src.

Thin driver over aivision.pocs.orb_slam3: writes the settings YAML from the
sequence intrinsics, invokes the stereo_kitti example binary, leaves
CameraTrajectory.txt (+ KeyFrameTrajectory.txt, logs) in --src.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aivision.kitti import load_sequence
from aivision.pocs.orb_slam3 import write_kitti_yaml, run_stereo_kitti, find_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, required=True, help="KITTI seq dir (image_0/1, times.txt)")
    ap.add_argument("--src", type=Path, required=True, help="output dir for CameraTrajectory.txt")
    ap.add_argument("--baseline", type=float, default=0.5372, help="stereo baseline in metres")
    args = ap.parse_args()

    s = load_sequence(args.seq_dir, camera="image_0")
    args.src.mkdir(parents=True, exist_ok=True)
    p = find_paths()
    yaml = write_kitti_yaml(args.src / "orbslam_stereo.yaml", s.K, s.width, s.height,
                            baseline_m=args.baseline)
    traj = run_stereo_kitti(args.seq_dir, p["binary"], p["vocab"], yaml, args.src)
    n = sum(1 for _ in open(traj))
    print(f"[orbslam3] wrote {traj}  ({n} poses for {s.n_frames} frames)")


if __name__ == "__main__":
    main()
