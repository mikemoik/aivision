"""Convert RLMD polygon-mask labels -> YOLO11-seg dataset (subset of classes).

RLMD ships per-image label PNGs (data/rlmd/labels/{train,val}/<stem>.png) painted
with the 25-class palette in classes.csv. We train on a SUBSET (see KEEP below) and
remap the kept RLMD ids to contiguous YOLO ids 0..N-1. Everything not kept becomes
unlabeled background.

Output is a fresh YOLO root (default data/rlmd_yolo/) that leaves the raw dataset
untouched: kept images are symlinked under images/{train,val}/ and a YOLO-seg label
.txt (normalized polygons) is written under labels/{train,val}/. A data.yaml is
emitted. Background frames (no kept instance) are kept only as a modest random
sample of hard negatives (BG_KEEP_FRAC).

Mask encoding is auto-detected per file: a single-channel/paletted PNG is read as a
class-id index map; an RGB PNG is matched against the classes.csv palette.

Re-runnable: deletes and rebuilds the output root each run.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- config -----------------------------------------------------------------
# RLMD id -> YOLO id. Edit MERGE below to collapse classes (see --check output).
KEEP = {
    2: 0,    # crosswalk
    3: 1,    # stop_line
    11: 2,   # left_arrow
    12: 3,   # straight_arrow
    13: 4,   # right_arrow
    14: 5,   # left_straight_arrow
    15: 6,   # right_straight_arrow
}
YOLO_NAMES = {
    0: "crosswalk", 1: "stop_line", 2: "left_arrow", 3: "straight_arrow",
    4: "right_arrow", 5: "left_straight_arrow", 6: "right_straight_arrow",
}
# Optional second-pass merge applied to YOLO ids (filled in after --check if a
# subtype is too sparse). Maps an existing YOLO id -> the YOLO id it folds into;
# names are rebuilt to a contiguous set. Leave empty to keep all 7 classes.
MERGE: dict[int, int] = {}

MIN_AREA_PX = 24          # drop contours smaller than this (px^2)
POLY_EPS_FRAC = 0.004     # approxPolyDP epsilon as a fraction of contour perimeter
BG_KEEP_FRAC = 0.08       # fraction of pure-background frames kept as hard negatives
SEED = 0
# ----------------------------------------------------------------------------


def load_palette(classes_csv: Path) -> dict[int, tuple[int, int, int]]:
    pal = {}
    with open(classes_csv) as f:
        for row in csv.DictReader(f):
            pal[int(row["id"])] = (int(row["r"]), int(row["g"]), int(row["b"]))
    return pal


def mask_to_idmap(mask_path: Path, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Load an RLMD label PNG -> HxW int class-id map (auto-detect encoding).

    RLMD masks are flat-filled with EXACT palette colours (no anti-aliasing), so
    RGB masks are decoded by exact colour match: any colour not in the palette
    becomes background (id 0). Single-channel/paletted PNGs are read as id maps.
    """
    im = Image.open(mask_path)
    if im.mode in ("P", "L", "I", "I;16"):
        # paletted / single-channel: pixel value already IS the class id
        return np.array(im, dtype=np.int32)
    rgb = np.array(im.convert("RGB"), dtype=np.uint32)                 # HxWx3
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]    # HxW key
    idmap = np.zeros(packed.shape, dtype=np.int32)                     # default background
    for cid, (r, g, b) in palette.items():
        if cid == 0:
            continue
        idmap[packed == ((r << 16) | (g << 8) | b)] = cid
    return idmap


def polygons_for_class(bin_mask: np.ndarray) -> list[np.ndarray]:
    """External contours of a binary mask -> simplified polygons (px, Nx2)."""
    cnts, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA_PX:
            continue
        eps = POLY_EPS_FRAC * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        polys.append(ap.astype(np.float64))
    return polys


def convert_split(split: str, rlmd: Path, palette, out_root: Path, remap: dict[int, int],
                  rng: random.Random) -> Counter:
    img_dir = rlmd / "images" / split
    lab_dir = rlmd / "labels" / split
    out_img = out_root / "images" / split
    out_lab = out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    images = sorted(img_dir.glob("*.jpg"))
    for img_path in tqdm(images, desc=f"convert {split}"):
        mask_path = lab_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue
        idmap = mask_to_idmap(mask_path, palette)
        H, W = idmap.shape
        lines = []
        for rlmd_id, yolo_id in KEEP.items():
            yolo_id = remap.get(yolo_id, yolo_id)
            present = idmap == rlmd_id
            if not present.any():
                continue
            for poly in polygons_for_class(present):
                counts[yolo_id] += 1
                norm = poly.copy()
                norm[:, 0] /= W
                norm[:, 1] /= H
                norm = norm.clip(0.0, 1.0)
                coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
                lines.append(f"{yolo_id} {coords}")

        keep = bool(lines) or (rng.random() < BG_KEEP_FRAC)
        if not keep:
            continue
        # symlink image into the YOLO root, write label .txt (empty -> hard negative)
        link = out_img / img_path.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(img_path.resolve())
        (out_lab / (img_path.stem + ".txt")).write_text("\n".join(lines))
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rlmd", type=Path, default=Path("data/rlmd"))
    ap.add_argument("--out", type=Path, default=Path("data/rlmd_yolo"))
    ap.add_argument("--check", action="store_true",
                    help="also print per-class instance counts (always printed)")
    args = ap.parse_args()

    palette = load_palette(args.rlmd / "classes.csv")
    rng = random.Random(SEED)

    # build contiguous name map after applying MERGE
    base_names = dict(YOLO_NAMES)
    if MERGE:
        kept_ids = sorted({MERGE.get(i, i) for i in base_names})
        relabel = {old: new for new, old in enumerate(kept_ids)}
        remap = {i: relabel[MERGE.get(i, i)] for i in base_names}
        names = {relabel[i]: base_names[i] for i in kept_ids}
    else:
        remap = {}
        names = base_names

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    total: Counter = Counter()
    for split in ("train", "val"):
        if not (args.rlmd / "images" / split).is_dir():
            print(f"[convert] skip missing split: {split}")
            continue
        c = convert_split(split, args.rlmd, palette, args.out, remap, rng)
        print(f"[convert] {split}: {sum(c.values())} instances over "
              f"{len(list((args.out / 'images' / split).glob('*')))} kept images")
        for yid in sorted(names):
            print(f"    {yid} {names[yid]:24s} {c.get(yid, 0)}")
        total.update(c)

    # data.yaml
    yaml = [f"path: {args.out.resolve()}", "train: images/train", "val: images/val",
            "names:"]
    for yid in sorted(names):
        yaml.append(f"  {yid}: {names[yid]}")
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"[convert] wrote {args.out / 'data.yaml'}")
    print("[convert] TOTAL per-class instances:")
    for yid in sorted(names):
        print(f"    {yid} {names[yid]:24s} {total.get(yid, 0)}")


if __name__ == "__main__":
    main()
