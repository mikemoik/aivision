"""Standalone YOLOv12 traffic-delineator (Leitpfosten) detector.

A thin adapter for the multi-model object-fusion stack: one frame -> a NORMALIZED
detection list and nothing more, decoupled from any fusion logic. Mirrors
aivision.sign_detector / aivision.road_marking.

Weights: maco018/YOLOv12_traffic-delineator (YOLOv12x, single class
'traffic-delineator', AGPL-3.0), pulled from the HF hub into the project-local
models/ cache. Boxes only (no masks). Card metrics: mAP50 0.607, precision 0.874,
recall 0.419 -- high precision, modest recall, which the across-frames clustering in
the static-env lift absorbs.

Detection record (the shared schema all detectors emit):
    {box: (x0, y0, x1, y1) float64, cls: int, name: str, score: float, source: str}
"""

from __future__ import annotations

import numpy as np

HF_REPO = "maco018/YOLOv12_traffic-delineator"
HF_FILE = "models/YOLOv12_traffic-delineator.pt"
SOURCE = "yolov12_delineator"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50

_state: dict = {}


def load():
    """Lazy-load the YOLOv12 model; cache for the process. Returns (model, names, dev)."""
    if not _state:
        import torch
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        weights = hf_hub_download(HF_REPO, HF_FILE)
        model = YOLO(weights)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        _state.update(model=model, names=dict(model.names), dev=dev)
    return _state["model"], _state["names"], _state["dev"]


def class_names() -> dict:
    """The model's class-id -> name map (single 'traffic-delineator' class)."""
    return dict(load()[1])


def detect(img_bgr: np.ndarray, conf: float = DEFAULT_CONF,
           iou: float = DEFAULT_IOU) -> list[dict]:
    """Detect traffic delineators in a BGR frame -> normalized detection records.

    ultralytics treats a numpy array as BGR (cv2 convention), so pass img_bgr as-is.
    """
    model, names, dev = load()
    res = model.predict(img_bgr, conf=conf, iou=iou, device=dev, verbose=False)[0]
    b = res.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    scores = b.conf.cpu().numpy()
    out = []
    for box, c, s in zip(xyxy, cls, scores):
        out.append(dict(box=box.astype(np.float64), cls=int(c),
                        name=names.get(int(c), str(int(c))), score=float(s),
                        source=SOURCE))
    return out
