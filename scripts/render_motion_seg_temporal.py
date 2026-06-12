"""Dense motion segmentation with temporal +/-N persistence (two-pass).

Pass 1 computes the per-pair raw+despeckled label, forward flow (t->t+1) and
backward flow (t+1->t) for every pair. Pass 2 votes each pixel over a +/-window
of flow-warped neighbour labels (motion_seg.temporal_vote): MOVING needs >=m_move
votes (kills 1-frame false positives), UNKNOWN is rescued to STATIC where
neighbours agree (recovers the near-field road that is clean on some pairs).

Outputs static=green / moving=red / unknown=gray, same as the per-pair renderer.
Memory: stores all labels (uint8) + flows (float16) for the range; ~1GB for the
full 271-frame sequence. Use --start/--end to limit.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.flow import compute_flow_pair
from aivision.kitti import load_sequence
from aivision.pocs import common
from aivision.motion_seg import (
    compute_fields, classify, despeckle, temporal_vote,
    STATIC, MOVING, UNKNOWN, TEMP_WINDOW,
)

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")
RHO_STEREO = SRC / "inv_depth_fused_rho_stereo_16bit.mkv"

COLORS = {
    STATIC:  np.array([0, 200, 0], np.float32),
    MOVING:  np.array([0, 0, 255], np.float32),
    UNKNOWN: np.array([130, 130, 130], np.float32),
}


def rho_stream(video: Path, W: int, H: int):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE)
    fb = W * H * 2
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            pix = np.frombuffer(raw, np.uint16).reshape(H, W).astype(np.float32)
            yield np.where(pix > 0, (pix - 1.0) / 65534.0, 0.0)
    finally:
        proc.stdout.close()
        proc.wait()


def overlay(bgr, label, alpha=0.5):
    out = bgr.astype(np.float32)
    color = np.zeros_like(out)
    for cls, col in COLORS.items():
        color[label == cls] = col
    return (out * (1 - alpha) + color * alpha).astype(np.uint8)


def banner(img, t, N, frac, window):
    lines = [
        f"frame {t:03d}/{N-1}  pair {t}->{t+1}   DENSE motion-seg + temporal +/-{window}",
        f"static {frac[STATIC]:4.1f}%   moving {frac[MOVING]:4.1f}%   unknown {frac[UNKNOWN]:4.1f}%"
        f"   [green=static red=moving gray=unknown]",
    ]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (720, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SRC / "motion_seg_temporal.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--backend", default="sea_raft")
    ap.add_argument("--window", type=int, default=TEMP_WINDOW)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    left = load_sequence(SEQ_DIR, camera="image_0")
    K, W, H, N = left.K, left.width, left.height, left.n_frames
    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")
    end = N if args.end < 0 else min(args.end, N)

    # --- pass 1: per-pair labels + flows --------------------------------------
    labels, fwds, bwds = [], [], []
    rit = rho_stream(RHO_STEREO, W, H)
    rho_a = next(rit)
    for _ in range(args.start):
        rho_a = next(rit, rho_a)
    a = cv2.cvtColor(cv2.imread(str(left.image_paths[args.start]),
                                cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    raw_frac = {STATIC: 0.0, MOVING: 0.0, UNKNOWN: 0.0}
    npairs = 0
    for t in tqdm(range(args.start, end - 1), desc="pass1 labels"):
        b = cv2.cvtColor(cv2.imread(str(left.image_paths[t + 1]),
                                    cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        rho_b = next(rit, rho_a)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        f_obs = compute_flow_pair(a, b, backend=args.backend)
        f_ba = compute_flow_pair(b, a, backend=args.backend)
        g = compute_fields(f_obs, f_ba, rho_a, rho_b, K, T_rel)
        lab = despeckle(classify(g), g)
        labels.append(lab)
        fwds.append(f_obs.astype(np.float16))
        bwds.append(f_ba.astype(np.float16))
        for c in raw_frac:
            raw_frac[c] += 100.0 * (lab == c).sum() / lab.size
        npairs += 1
        a, rho_a = b, rho_b

    # --- temporal vote --------------------------------------------------------
    fwds = [f.astype(np.float32) for f in fwds]
    bwds = [f.astype(np.float32) for f in bwds]
    voted = temporal_vote(labels, fwds, bwds, window=args.window)

    # --- pass 2: render -------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"),
                             args.fps, (W, H))
    vote_frac = {STATIC: 0.0, MOVING: 0.0, UNKNOWN: 0.0}
    for i, t in enumerate(tqdm(range(args.start, end - 1), desc="pass2 render")):
        img = cv2.cvtColor(cv2.imread(str(left.image_paths[t]),
                                      cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        lab = voted[i]
        frac = {c: 100.0 * (lab == c).sum() / lab.size for c in COLORS}
        for c in vote_frac:
            vote_frac[c] += frac[c]
        writer.write(banner(overlay(img, lab, args.alpha), t, N, frac, args.window))
    writer.release()

    print(f"[motion-seg temporal] wrote {args.out}")
    print(f"  per-pair  mean: static {raw_frac[STATIC]/npairs:5.1f}  "
          f"moving {raw_frac[MOVING]/npairs:5.2f}  unknown {raw_frac[UNKNOWN]/npairs:5.1f}")
    print(f"  temporal  mean: static {vote_frac[STATIC]/npairs:5.1f}  "
          f"moving {vote_frac[MOVING]/npairs:5.2f}  unknown {vote_frac[UNKNOWN]/npairs:5.1f}")


if __name__ == "__main__":
    main()
