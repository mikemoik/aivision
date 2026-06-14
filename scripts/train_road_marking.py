"""Train a YOLO11-seg road-marking detector on the converted RLMD dataset.

Detector 3 of the object-fusion stack. Trains instance segmentation for crosswalks,
stop lines, and per-direction arrows (see scripts/rlmd_to_yoloseg.py for the taxonomy).

CRITICAL — augmentation: the arrow classes are DIRECTION-SPECIFIC, so horizontal
flip (fliplr) is DISABLED: a left-flip turns left_arrow into right_arrow and silently
corrupts the labels. Vertical flip is also off (road scenes have a fixed up). We lean
on perspective / scale / translate / hsv / mosaic instead to bridge the Taiwan->Germany
viewpoint+style gap. imgsz is large because markings are thin/small.

Best weights are copied to models/road_marking/best.pt for the adapter to load.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import aivision  # noqa: F401  (configures model_cache -> project-local models/)
from aivision.model_cache import models_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/rlmd_yolo/data.yaml"))
    ap.add_argument("--model", default="yolo11m-seg.pt",
                    help="pretrained seg base (yolo11s-seg.pt / yolo11m-seg.pt)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="rlmd_yolo11seg")
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    from ultralytics import YOLO

    out_root = models_dir() / "road_marking"
    out_root.mkdir(parents=True, exist_ok=True)
    # keep the pretrained base + run dir under models/road_marking/
    base = out_root / args.model
    model = YOLO(str(base) if base.exists() else args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(out_root / "runs"),
        name=args.name,
        patience=args.patience,
        # --- augmentation (bridge viewpoint/style gap; NO horizontal/vertical flip) ---
        fliplr=0.0,          # arrows are direction-specific -> flipping corrupts labels
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.1,
        copy_paste_mode="mixup",  # paste instances WITHOUT h-flip; "flip" (the
                                  # default) would mirror arrows -> corrupt left/right

        # ----------------------------------------------------------------------------
        seed=0,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dst = out_root / "best.pt"
        shutil.copy(str(best), str(dst))
        print(f"[train] copied best weights -> {dst}")
    print(f"[train] run dir: {results.save_dir}")


if __name__ == "__main__":
    main()
