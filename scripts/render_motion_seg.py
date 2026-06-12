"""Render the DENSE per-pixel motion segmentation (static/moving/unknown).

Raw premise-check: no denoising, no region voting, no semantic gating. Every
pixel is routed independently by motion_seg.classify() and painted

    STATIC  -> green
    MOVING  -> red
    UNKNOWN -> gray

so we can see honestly how noisy the per-pixel geometry is before deciding on
any spatial/temporal regularization.

Everything in the image_0 frame (pose + stereo rho + RAFT flow all live there).
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
    compute_fields, classify, despeckle, STATIC, MOVING, UNKNOWN,
)

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")
RHO_STEREO = SRC / "inv_depth_fused_rho_stereo_16bit.mkv"

COLORS = {
    STATIC:  np.array([0, 200, 0], np.float32),     # green  (BGR)
    MOVING:  np.array([0, 0, 255], np.float32),     # red
    UNKNOWN: np.array([130, 130, 130], np.float32),  # gray
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


def overlay(bgr: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = bgr.astype(np.float32)
    color = np.zeros_like(out)
    for cls, col in COLORS.items():
        color[label == cls] = col
    out = out * (1 - alpha) + color * alpha
    return out.astype(np.uint8)


def banner(img: np.ndarray, t: int, N: int, frac: dict[int, float]) -> np.ndarray:
    lines = [
        f"frame {t:03d}/{N-1}  pair {t}->{t+1}   DENSE per-pixel motion-seg (raw, no denoise)",
        f"static {frac[STATIC]:4.1f}%   moving {frac[MOVING]:4.1f}%   unknown {frac[UNKNOWN]:4.1f}%"
        f"   [green=static red=moving gray=unknown]",
    ]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (700, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SRC / "motion_seg.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--backend", default="sea_raft")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    left = load_sequence(SEQ_DIR, camera="image_0")
    K, W, H, N = left.K, left.width, left.height, left.n_frames
    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")

    end = N if args.end < 0 else min(args.end, N)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"),
                             args.fps, (W, H))

    rit = rho_stream(RHO_STEREO, W, H)
    rho_a = next(rit)
    for _ in range(args.start):
        rho_a = next(rit, rho_a)
    a = cv2.cvtColor(cv2.imread(str(left.image_paths[args.start]),
                                cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    for t in tqdm(range(args.start, end - 1), desc="motion-seg"):
        b = cv2.cvtColor(cv2.imread(str(left.image_paths[t + 1]),
                                    cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        rho_b = next(rit, rho_a)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]

        f_obs = compute_flow_pair(a, b, backend=args.backend)
        f_ba = compute_flow_pair(b, a, backend=args.backend)
        g = compute_fields(f_obs, f_ba, rho_a, rho_b, K, T_rel)
        label = despeckle(classify(g), g)

        tot = label.size
        frac = {c: 100.0 * (label == c).sum() / tot for c in COLORS}
        writer.write(banner(overlay(a, label, args.alpha), t, N, frac))
        a, rho_a = b, rho_b

    writer.release()
    print(f"[motion-seg] wrote {args.out}")


if __name__ == "__main__":
    main()
