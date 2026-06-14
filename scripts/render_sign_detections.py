"""Standalone validation render for the YOLOv8 GTSRB traffic-sign detector.

Runs aivision.sign_detector on the RGB frames of a KITTI sequence and draws the sign
detections (box + class + score) -- NO flow / rho / motion fusion. The point is to
eyeball this single detector's quality on seq04 before it is wired into the
multi-model object fusion. Magenta boxes keep it visually distinct from the
green/red/orange/blue motion-seg overlay.

--proposals:
  none      run the model on the full frame (BASELINE -- degenerate: GTSRB training
            makes it emit one whole-frame box, so this is expected to look broken).
  semantic  the RELIABLE mode: use the Mapillary 'Traffic Sign' semantic regions as
            proposals, classify each crop. Localization = segmenter, type = YOLOv8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence
from aivision import sign_detector as sd

SEQ_DIR = Path("data/kitti/04")
MAGENTA = (255, 0, 255)  # BGR
SIGN_SEM_NAMES = ("Traffic Sign (Front)", "Traffic Sign (Back)", "Traffic Sign Frame")


def draw(img, dets):
    for d in dets:
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        cv2.rectangle(img, (x0, y0), (x1, y1), MAGENTA, 2)
        tag = f"{d['name']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x0, y0 - th - 4), (x0 + tw, y0), MAGENTA, -1)
        cv2.putText(img, tag, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img


def banner(img, t, N, n, total, conf, mode):
    lines = [
        f"frame {t:03d}/{N-1}   YOLOv8 GTSRB sign detector   proposals={mode}   conf>={conf:.2f}",
        f"signs this frame {n}   running total {total}   [magenta = traffic sign]",
    ]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (760, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=SEQ_DIR)
    ap.add_argument("--camera", default="image_2", help="RGB camera to render on")
    ap.add_argument("--out", type=Path, default=Path("outputs/pocs/orb_slam3/sign_detections.mp4"))
    ap.add_argument("--proposals", choices=["none", "semantic"], default="semantic",
                    help="region proposer: 'none' = full frame (degenerate), 'semantic' = Mapillary signs")
    ap.add_argument("--perception", type=Path,
                    default=Path("outputs/pocs/orb_slam3/perception"),
                    help="perception/ cache dir (for --proposals semantic)")
    ap.add_argument("--min-area", type=int, default=40,
                    help="min semantic sign component area (px) to propose")
    ap.add_argument("--conf", type=float, default=sd.DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=sd.DEFAULT_IOU)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    seq = load_sequence(args.seq_dir, camera=args.camera)
    W, H, N = seq.width, seq.height, seq.n_frames
    end = N if args.end < 0 else min(args.end, N)

    sign_regions = None
    if args.proposals == "semantic":
        from aivision.perception import (
            CacheManifest, MANIFEST_NAME, id2label_from_manifest, load_frame_entry)
        man = CacheManifest.load(args.perception / MANIFEST_NAME)
        id2lab = id2label_from_manifest(man)
        l2i = {v: k for k, v in id2lab.items()}
        sign_ids = [l2i[n] for n in SIGN_SEM_NAMES if n in l2i]

        def sign_regions(t):  # noqa: F811 -- semantic proposal boxes for frame t
            sem = load_frame_entry(args.perception, t)["semantic"]
            if sem.shape != (H, W):
                sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
            m = np.isin(sem, sign_ids).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            boxes = []
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < args.min_area:
                    continue
                x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
                w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
                boxes.append((x, y, x + w, y + h))
            return boxes

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"),
                             args.fps, (W, H))
    total = 0
    per_class: dict[str, int] = {}
    for t in tqdm(range(args.start, end), desc="sign detect"):
        im = cv2.imread(str(seq.image_paths[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        if sign_regions is None:
            dets = sd.detect(im, conf=args.conf, iou=args.iou)
        else:
            dets = sd.detect_in_regions(im, sign_regions(t), conf=args.conf, iou=args.iou)
        total += len(dets)
        for d in dets:
            per_class[d["name"]] = per_class.get(d["name"], 0) + 1
        frame = draw(im, dets)
        writer.write(banner(frame, t, N, len(dets), total, args.conf, args.proposals))
    writer.release()
    print(f"[sign detect] wrote {args.out}")
    print(f"  {total} detections over {end - args.start} frames")
    for name, c in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {name:24s} {c}")


if __name__ == "__main__":
    main()
