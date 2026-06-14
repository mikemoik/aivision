"""Reconstruct data/rlmd/ from RLMD zip(s) downloaded via a browser.

Google Drive, when you download a folder, returns one or more .zip files whose
internal layout nests the original tree (images/{train,val}/*.jpg,
labels/{train,val}/*.png) under some top-level folder name, and may split the
content across several zips. This script is agnostic to that: it extracts every
zip in the drop dir to a staging area, then places each .jpg/.png into
data/rlmd/<images|labels>/<train|val>/ based on the 'images'/'labels' and
'train'/'val' components of its path inside the archive.

Usage:
    1. download the RLMD Drive folder via your browser into  data/rlmd_zips/
    2. uv run python scripts/ingest_rlmd_zips.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def classify(parts: tuple[str, ...]) -> tuple[str, str] | None:
    """(path parts) -> ('images'|'labels', 'train'|'val') or None if not a data file."""
    lower = [p.lower() for p in parts]
    kind = "images" if "images" in lower else ("labels" if "labels" in lower else None)
    split = "train" if "train" in lower else ("val" if "val" in lower else None)
    if kind and split:
        return kind, split
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zips", type=Path, default=Path("data/rlmd_zips"))
    ap.add_argument("--out", type=Path, default=Path("data/rlmd"))
    args = ap.parse_args()

    zips = sorted(args.zips.glob("*.zip"))
    if not zips:
        raise SystemExit(f"no .zip files found in {args.zips} — "
                         f"download the RLMD Drive folder there first")
    print(f"[ingest] {len(zips)} zip(s): {[z.name for z in zips]}")

    placed = {("images", "train"): 0, ("images", "val"): 0,
              ("labels", "train"): 0, ("labels", "val"): 0}
    other_csv = None
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for z in zips:
            print(f"[ingest] extracting {z.name} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(stage)
        for f in stage.rglob("*"):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if f.name.lower() == "classes.csv":
                other_csv = f
                continue
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            cat = classify(f.relative_to(stage).parts)
            if cat is None:
                continue
            dst_dir = args.out / cat[0] / cat[1]
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst_dir / f.name))
            placed[cat] += 1
        if other_csv is not None:
            shutil.copy(str(other_csv), str(args.out / "classes.csv"))

    print("[ingest] placed files:")
    for k, v in placed.items():
        print(f"    {k[0]}/{k[1]:5s} {v}")
    total_img = placed[("images", "train")] + placed[("images", "val")]
    total_lab = placed[("labels", "train")] + placed[("labels", "val")]
    print(f"[ingest] images={total_img}  labels={total_lab}  (RLMD ~2137 each)")


if __name__ == "__main__":
    main()
