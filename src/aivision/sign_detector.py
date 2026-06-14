"""Standalone GTSRB traffic-sign classifier (full 43-class German GTSRB).

The first independent detector for the multi-model object-fusion stack: a thin
adapter that turns one frame (or sign crops) into a NORMALIZED detection list and
nothing more. It is deliberately decoupled from any fusion logic -- it only knows how
to run THIS model and emit uniform (box, cls, name, score, source) records, so the
later fusion step can consume it identically to every other detector.

Backend: bazyl/gtsrb-model (ViT image-classifier, all 43 GTSRB classes incl. the eight
speed limits 20/30/50/60/70/80/100/120 km/h), pulled from the HF hub into the
project-local models/ cache. The previous Phearith YOLOv8 head carried only a single
`SpeedLimit60` class, so it localised speed signs but always mis-read the value as 60;
this classifier reads the true limit.

Because GTSRB models are trained on tight, centred sign crops, localisation is NOT this
module's job -- a proposer (e.g. Mapillary 'Traffic Sign (Front)' semantic regions)
supplies the box and `detect_in_regions` classifies each crop. `detect` (whole frame) is
the degenerate fallback that just classifies the entire image.

Detection record (the shared schema all detectors emit):
    {box: (x0, y0, x1, y1) float64, cls: int, name: str, score: float, source: str}
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HF_REPO = "bazyl/gtsrb-model"
LOCAL_DIRNAME = "gtsrb_classifier"
SOURCE = "gtsrb_vit_classifier"
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.50          # kept for call-site/signature compatibility (unused by a classifier)

_state: dict = {}


def _prepared_dir() -> Path:
    """Fetch the model into the project cache and patch the config's null label
    (transformers 5.x strict-validates id2label and rejects the repo's `"43": null`)."""
    from huggingface_hub import snapshot_download
    from aivision.model_cache import models_dir

    local = Path(snapshot_download(HF_REPO, local_dir=str(models_dir() / LOCAL_DIRNAME)))
    cfg_path = local / "config.json"
    cfg = json.loads(cfg_path.read_text())
    i2l = cfg.get("id2label", {})
    if any(v is None for v in i2l.values()):
        i2l = {k: (v if v is not None else f"unused_{k}") for k, v in i2l.items()}
        cfg["id2label"] = i2l
        cfg["label2id"] = {v: int(k) for k, v in i2l.items()}
        cfg_path.write_text(json.dumps(cfg))
    return local


def load():
    """Lazy-load the classifier; cache for the process. Returns the state dict."""
    if not _state:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        local = _prepared_dir()
        proc = AutoImageProcessor.from_pretrained(local)
        model = AutoModelForImageClassification.from_pretrained(local)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev).eval()
        names = {int(k): v for k, v in model.config.id2label.items()}
        _state.update(model=model, proc=proc, names=names, dev=dev)
    return _state


def class_names() -> dict:
    """The model's class-id -> name map (43 GTSRB sign classes)."""
    return dict(load()["names"])


def _classify(crops_bgr: list[np.ndarray]) -> list[tuple[int, str, float]]:
    """Classify a batch of BGR sign crops -> [(cls, name, score), ...] (softmax top-1)."""
    import torch
    from PIL import Image

    st = load()
    imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    inp = st["proc"](imgs, return_tensors="pt").to(st["dev"])
    with torch.no_grad():
        prob = st["model"](**inp).logits.softmax(-1)
    score, idx = prob.max(-1)
    return [(int(i), st["names"][int(i)], float(s))
            for i, s in zip(idx.cpu().numpy(), score.cpu().numpy())]


def detect(img_bgr: np.ndarray, conf: float = DEFAULT_CONF,
           iou: float = DEFAULT_IOU) -> list[dict]:
    """Degenerate whole-frame classification -> at most one record (box = full frame).

    A classifier cannot localise; use detect_in_regions with a proposer for real work.
    Kept so call sites that ran the old detector full-frame still function.
    """
    H, W = img_bgr.shape[:2]
    c, name, score = _classify([img_bgr])[0]
    if score < conf:
        return []
    return [dict(box=np.array([0, 0, W, H], np.float64), cls=c, name=name,
                 score=float(score), source=SOURCE)]


def detect_in_regions(img_bgr: np.ndarray, regions, conf: float = DEFAULT_CONF,
                      iou: float = DEFAULT_IOU, pad: int = 8,
                      min_side: int = 12) -> list[dict]:
    """Classify caller-supplied sign PROPOSALS -> normalized records (full-frame coords).

    This is the model's RELIABLE operating mode. Localization must come from a proposer
    (e.g. the Mapillary 'Traffic Sign (Front)' semantic regions); this model supplies only
    the sign TYPE. Each proposal (x0,y0,x1,y1) is padded, cropped, and classified; the
    crop's top-1 class/score is reported AT the proposal box. The proposer is the caller's
    concern, not this module's.
    """
    H, W = img_bgr.shape[:2]
    crops, boxes = [], []
    for x0, y0, x1, y1 in regions:
        xa, ya = max(0, int(x0) - pad), max(0, int(y0) - pad)
        xb, yb = min(W, int(x1) + pad), min(H, int(y1) + pad)
        if xb - xa < min_side or yb - ya < min_side:
            continue
        crops.append(img_bgr[ya:yb, xa:xb])
        boxes.append((x0, y0, x1, y1))
    if not crops:
        return []
    out = []
    for (x0, y0, x1, y1), (c, name, score) in zip(boxes, _classify(crops)):
        if score < conf:
            continue
        out.append(dict(box=np.array([x0, y0, x1, y1], np.float64), cls=c,
                        name=name, score=float(score), source=SOURCE))
    return out
