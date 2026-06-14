"""Standalone validation render for the EgoLanes lane-segmentation detector.

Runs aivision.ego_lanes on the RGB frames of a KITTI sequence (or an arbitrary image
glob) and overlays the three lane masks -- ego-left (blue), ego-right (pink), other
lanes / road edges (green) -- NO fusion. The point is to eyeball this single model's
quality + German transfer on seq04 before wiring it into the multi-model fusion.

Render is on the RGB camera (image_2), never grayscale -- see memory feedback_render_rgb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence
from aivision import ego_lanes as el

SEQ_DIR = Path("data/kitti/04")
# per-class BGR colours (index = class id): ego-left blue, ego-right pink, other green
CLASS_COLORS = {0: (255, 0, 0), 1: (203, 192, 255), 2: (0, 255, 0)}


def draw(img, dets):
    overlay = img.copy()
    for d in dets:
        overlay[d["mask"]] = CLASS_COLORS.get(d["cls"], (255, 255, 255))
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)
    return img


def banner(img, t, N, dets, thr):
    present = {d["name"]: d["score"] for d in dets}
    lines = [
        f"frame {t:03d}/{N-1}   EgoLanes (Autoware)   thr>={thr:.2f}",
        "  ".join(f"{n}:{present[n]:.2f}" for n in
                  ("ego_left_lane", "ego_right_lane", "other_lanes") if n in present)
        or "no lanes",
    ]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (720, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (240, 240, 240), 1, cv2.LINE_AA)
    # legend
    for k, (name, col) in enumerate([("ego-left", (255, 0, 0)),
                                     ("ego-right", (203, 192, 255)),
                                     ("other", (0, 255, 0))]):
        cv2.putText(img, name, (img.shape[1] - 150, 22 + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=SEQ_DIR)
    ap.add_argument("--camera", default="image_2", help="RGB camera to render on")
    ap.add_argument("--out", type=Path, default=Path("outputs/ego_lanes/ego_lanes_seq04.mp4"))
    ap.add_argument("--image-glob", default=None,
                    help="render an arbitrary image set (e.g. A2D2) instead of a KITTI seq")
    ap.add_argument("--thr", type=float, default=el.DEFAULT_THR)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    if args.image_glob:
        import glob
        paths = sorted(glob.glob(args.image_glob, recursive=True))
        if not paths:
            raise SystemExit(f"no images match {args.image_glob}")
        H, W = cv2.imread(paths[0], cv2.IMREAD_COLOR).shape[:2]
    else:
        seq = load_sequence(args.seq_dir, camera=args.camera)
        W, H = seq.width, seq.height
        paths = [str(p) for p in seq.image_paths]
    N = len(paths)
    end = N if args.end < 0 else min(args.end, N)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"),
                             args.fps, (W, H))
    px_total = {0: 0, 1: 0, 2: 0}
    frames_with = {0: 0, 1: 0, 2: 0}
    for t in tqdm(range(args.start, end), desc="ego lanes"):
        im = cv2.imread(paths[t], cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        dets = el.detect(im, thr=args.thr)
        for d in dets:
            px_total[d["cls"]] += int(d["mask"].sum())
            frames_with[d["cls"]] += 1
        writer.write(banner(draw(im, dets), t, N, dets, args.thr))
    writer.release()
    print(f"[ego lanes] wrote {args.out}")
    for c in (0, 1, 2):
        print(f"    {el.CLASS_NAMES[c]:16s} present in {frames_with[c]}/{end-args.start} "
              f"frames, {px_total[c]} px total")


if __name__ == "__main__":
    main()
