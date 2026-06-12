"""Render a video with a semi-transparent red overlay on dynamic regions.

Reads the perception cache's per-frame semantic masks and paints every pixel
belonging to a dynamic (moving-object) Vistas class — animals, humans/riders,
and all vehicles — in translucent red over the source image_2 frames. This is
the visual companion to Phase 2's semantic dense-cloud masking: it shows
exactly which pixels get dropped before unprojection.

Sky / Car Mount / Ego Vehicle are part of the Phase 2 drop set but are NOT
dynamic, so they are deliberately excluded here.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from aivision.kitti import load_sequence
from aivision.perception import (
    CacheManifest,
    MANIFEST_NAME,
    id2label_from_manifest,
    load_frame_entry,
)

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")

# Moving-object classes only (the DROP set minus Sky / Car Mount / Ego Vehicle).
DYNAMIC_CLASS_NAMES: frozenset[str] = frozenset({
    "Bird", "Ground Animal",
    "Person", "Bicyclist", "Motorcyclist", "Other Rider",
    "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle",
    "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow",
})

RED_BGR = np.array([0, 0, 255], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perception", type=Path, default=SRC / "perception")
    ap.add_argument("--out", type=Path, default=SRC / "dynamic_overlay.mp4")
    ap.add_argument("--alpha", type=float, default=0.5, help="red opacity on dynamic pixels")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    manifest = CacheManifest.load(args.perception / MANIFEST_NAME)
    if not manifest.populated.get("semantic", False):
        raise SystemExit(f"{args.perception} has no semantic field; run perception-cache --segment")
    id2label = id2label_from_manifest(manifest)
    label2id = {v: k for k, v in id2label.items()}
    missing = sorted(DYNAMIC_CLASS_NAMES - set(label2id))
    if missing:
        raise SystemExit(f"dynamic classes not in seg model id2label: {missing}")
    dynamic_ids = [label2id[n] for n in DYNAMIC_CLASS_NAMES]

    seq = load_sequence(SEQ_DIR, camera="image_2")
    W, H = seq.width, seq.height
    N = seq.n_frames

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (W, H))

    n_dyn_px = 0
    for i in range(N):
        img = cv2.imread(str(seq.image_paths[i]), cv2.IMREAD_COLOR).astype(np.float32)
        sem = load_frame_entry(args.perception, i)["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.isin(sem, dynamic_ids)
        n_dyn_px += int(mask.sum())
        img[mask] = (1.0 - args.alpha) * img[mask] + args.alpha * RED_BGR
        writer.write(img.astype(np.uint8))

    writer.release()
    print(f"[dynamic-overlay] {N} frames -> {args.out}")
    print(f"[dynamic-overlay] {n_dyn_px/1e6:.1f}M dynamic px painted "
          f"({n_dyn_px/(N*W*H)*100:.1f}% of all pixels), alpha={args.alpha}")


if __name__ == "__main__":
    main()
