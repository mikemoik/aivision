"""Standalone validation render for the RLMD YOLO11-seg road-marking detector.

Runs aivision.road_marking on the RGB frames of a KITTI sequence and draws the
marking detections (box + instance-mask overlay + class + score) -- NO fusion. The
point is to eyeball this single detector's quality and its German transfer on seq04
before it is wired into the multi-model object fusion. Per-class colours keep the
classes distinct; mirrors scripts/render_sign_detections.py in style.

Render is on the RGB camera (image_2), never grayscale -- see memory feedback_render_rgb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence
from aivision import road_marking as rm

SEQ_DIR = Path("data/kitti/04")
# per-class BGR colours (index = YOLO class id)
CLASS_COLORS = [
    (0, 255, 0),     # 0 crosswalk            green
    (0, 165, 255),   # 1 stop_line            orange
    (255, 0, 0),     # 2 left_arrow           blue
    (0, 255, 255),   # 3 straight_arrow       yellow
    (255, 0, 255),   # 4 right_arrow          magenta
    (255, 255, 0),   # 5 left_straight_arrow  cyan
    (128, 0, 255),   # 6 right_straight_arrow pink-red
]


def color_for(cls: int):
    return CLASS_COLORS[cls % len(CLASS_COLORS)]


def draw(img, dets):
    overlay = img.copy()
    for d in dets:
        col = color_for(d["cls"])
        overlay[d["mask"]] = col
    cv2.addWeighted(overlay, 0.40, img, 0.60, 0, dst=img)
    for d in dets:
        col = color_for(d["cls"])
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        tag = f"{d['name']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x0, y0 - th - 4), (x0 + tw, y0), col, -1)
        cv2.putText(img, tag, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img


def banner(img, t, N, n, total, conf):
    lines = [
        f"frame {t:03d}/{N-1}   RLMD YOLO11-seg road markings   conf>={conf:.2f}",
        f"markings this frame {n}   running total {total}",
    ]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (740, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=SEQ_DIR)
    ap.add_argument("--camera", default="image_2", help="RGB camera to render on")
    ap.add_argument("--out", type=Path, default=Path("outputs/rlmd/road_markings_seq04.mp4"))
    ap.add_argument("--conf", type=float, default=rm.DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=rm.DEFAULT_IOU)
    ap.add_argument("--image-glob", default=None,
                    help="render an arbitrary image set (e.g. A2D2 German frames) "
                         "instead of a KITTI sequence; e.g. "
                         "'data/a2d2/**/cam_front_center/*.png'")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    # source = an explicit image glob (A2D2) or a KITTI sequence (seq04 default)
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
    total = 0
    per_class: dict[str, int] = {}
    for t in tqdm(range(args.start, end), desc="road markings"):
        im = cv2.imread(paths[t], cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        dets = rm.detect(im, conf=args.conf, iou=args.iou)
        total += len(dets)
        for d in dets:
            per_class[d["name"]] = per_class.get(d["name"], 0) + 1
        frame = draw(im, dets)
        writer.write(banner(frame, t, N, len(dets), total, args.conf))
    writer.release()
    print(f"[road markings] wrote {args.out}")
    print(f"  {total} detections over {end - args.start} frames")
    for name, c in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {name:24s} {c}")


if __name__ == "__main__":
    main()
