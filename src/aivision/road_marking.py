"""Standalone YOLO11-seg road-marking detector (RLMD-trained).

Detector 3 of the multi-model object-fusion stack: a thin adapter that turns one
frame into a NORMALIZED instance-segmentation detection list and nothing more.
Like sign_detector, it is decoupled from any fusion logic -- it only knows how to
run THIS model and emit uniform records, so the later fusion step consumes it
identically to every other detector.

Weights: a YOLO11-seg model trained on RLMD (crosswalks, stop lines, per-direction
arrows -- see scripts/rlmd_to_yoloseg.py / scripts/train_road_marking.py), loaded
from the project-local models/road_marking/best.pt.

Detection record (the shared schema, with an instance mask like detect.py):
    {box: (x0,y0,x1,y1) float64, mask: HxW bool, cls: int, name: str,
     score: float, source: "rlmd_yolo11seg"}
"""

from __future__ import annotations

import numpy as np
import cv2

from aivision.model_cache import models_dir

SOURCE = "rlmd_yolo11seg"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50

_state: dict = {}


def weights_path():
    """Path to the trained best.pt under models/road_marking/."""
    return models_dir() / "road_marking" / "best.pt"


def load():
    """Lazy-load the YOLO11-seg model; cache for the process. Returns (model, names, dev)."""
    if not _state:
        import torch
        from ultralytics import YOLO
        w = weights_path()
        if not w.exists():
            raise FileNotFoundError(
                f"road-marking weights not found at {w} -- train first with "
                f"scripts/train_road_marking.py")
        model = YOLO(str(w))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        _state.update(model=model, names=dict(model.names), dev=dev)
    return _state["model"], _state["names"], _state["dev"]


def class_names() -> dict:
    """The model's class-id -> name map (the kept RLMD road-marking classes)."""
    return dict(load()[1])


def detect(img_bgr: np.ndarray, conf: float = DEFAULT_CONF,
           iou: float = DEFAULT_IOU) -> list[dict]:
    """Detect road markings in a BGR frame -> normalized detection records.

    Masks are rasterized from the predicted polygons (res.masks.xy, already in
    ORIGINAL image coordinates) so each mask is a clean full-frame HxW bool aligned
    to img_bgr -- no letterbox bookkeeping. ultralytics treats a numpy array as BGR.
    """
    model, names, dev = load()
    H, W = img_bgr.shape[:2]
    res = model.predict(img_bgr, conf=conf, iou=iou, device=dev, verbose=False)[0]
    b = res.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    scores = b.conf.cpu().numpy()
    polys = res.masks.xy if res.masks is not None else [None] * len(xyxy)
    out = []
    for box, c, s, poly in zip(xyxy, cls, scores, polys):
        mask = np.zeros((H, W), dtype=np.uint8)
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        out.append(dict(box=box.astype(np.float64), mask=mask.astype(bool),
                        cls=int(c), name=names.get(int(c), str(int(c))),
                        score=float(s), source=SOURCE))
    return out
