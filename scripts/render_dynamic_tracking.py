"""Render the geometry-based dynamic-object tracker over KITTI seq 04.

Detects independently-moving regions via the depth-free epipolar residual
(+ depth-reprojection inside the forward-epipole cone), tracks them through the
flow field, and confirms each as dynamic only by temporal vote. Output: source
video with confirmed-dynamic objects in red, tentative in amber, plus per-track
IDs and a HUD. See src/aivision/dynamic.py for the method.
"""

import argparse
from pathlib import Path

from aivision.dynamic import render_dynamic_tracking_video

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=Path, default=SEQ_DIR)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=SRC / "dynamic_tracking.mp4")
    ap.add_argument("--backend", default="raft_large")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--epi-thresh", type=float, default=1.0)
    ap.add_argument("--rep-thresh", type=float, default=6.0)
    ap.add_argument("--cone-deg", type=float, default=22.0)
    ap.add_argument("--vote-window", type=int, default=5)
    ap.add_argument("--min-votes", type=int, default=2)
    args = ap.parse_args()

    render_dynamic_tracking_video(
        seq_dir=args.seq, src_dir=args.src, out_mp4=args.out,
        backend=args.backend, fps=args.fps,
        epi_thresh=args.epi_thresh, rep_thresh=args.rep_thresh,
        cone_deg=args.cone_deg, vote_window=args.vote_window, min_votes=args.min_votes,
    )


if __name__ == "__main__":
    main()
