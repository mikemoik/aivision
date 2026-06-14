"""Standalone EgoLanes lane-segmentation detector (Autoware Vision Pilot).

Detector 4 of the multi-model object-fusion stack. Unlike the box detectors, EgoLanes
is a SEMANTIC SEGMENTATION net: it takes a raw frame and emits three INDEPENDENT
binary lane masks -- ego-left lane, ego-right lane, and all other lanes / road edges
-- at 1/4 input resolution. We wrap it as the same decoupled adapter contract: one
frame -> a normalized detection list, one record per present lane class, each carrying
its full-res mask (and a mask-derived box) so the fusion step consumes it uniformly.

Weights: autowarefoundation/autoware_vision_pilot EgoLanes ONNX FP32 (Apache-2.0),
input 3x320x640 (RGB, ImageNet-normalized), output 3x80x160 logits (per-class sigmoid,
NOT a softmax -- the model is trained with per-class binary cross-entropy). Run via
onnxruntime so no PyTorch architecture vendoring is needed.

Detection record (shared schema, with a mask like detect.py / road_marking.py):
    {box: (x0,y0,x1,y1) float64, mask: HxW bool, cls: int, name: str,
     score: float, source: "autoware_egolanes"}
"""

from __future__ import annotations

import numpy as np
import cv2

from aivision.model_cache import models_dir

SOURCE = "autoware_egolanes"
IN_W, IN_H = 640, 320
DEFAULT_THR = 0.5
# channel order confirmed against seq04: ch0 = ego-left, ch1 = ego-right, ch2 = other
CLASS_NAMES = {0: "ego_left_lane", 1: "ego_right_lane", 2: "other_lanes"}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_state: dict = {}


def weights_path():
    """Path to the EgoLanes ONNX FP32 weights under models/ego_lanes/."""
    return models_dir() / "ego_lanes" / "egolanes_fp32.onnx"


def load():
    """Lazy-load the ONNX session; cache for the process. Returns (session, input_name)."""
    if not _state:
        import onnxruntime as ort
        w = weights_path()
        if not w.exists():
            raise FileNotFoundError(
                f"EgoLanes weights not found at {w} -- download the autoware_vision_pilot "
                f"EgoLanes ONNX FP32 (see scripts/download note in the task)")
        # prefer CUDA if this onnxruntime build exposes it, else CPU
        avail = ort.get_available_providers()
        provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
        sess = ort.InferenceSession(str(w), providers=provs)
        _state.update(sess=sess, inp=sess.get_inputs()[0].name)
    return _state["sess"], _state["inp"]


def class_names() -> dict:
    """The lane class-id -> name map (ego_left / ego_right / other)."""
    return dict(CLASS_NAMES)


def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (IN_W, IN_H)).astype(np.float32) / 255.0
    r = (r - _MEAN) / _STD
    return r.transpose(2, 0, 1)[None]                     # 1x3xHxW


def detect(img_bgr: np.ndarray, thr: float = DEFAULT_THR) -> list[dict]:
    """Segment ego/other lanes in a BGR frame -> normalized detection records.

    Each of the 3 channels is sigmoid-thresholded, upsampled to the original frame
    size, and emitted as one record (with its full-res bool mask and bounding box).
    Empty classes are skipped. score = max per-pixel sigmoid probability in the mask.
    """
    sess, inp = load()
    H, W = img_bgr.shape[:2]
    logits = sess.run(None, {inp: _preprocess(img_bgr)})[0][0]   # 3x80x160
    # clip before exp: logits reach ~-180 (background) -> exp overflow warning otherwise
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    out = []
    for c in range(prob.shape[0]):
        full = cv2.resize(prob[c], (W, H), interpolation=cv2.INTER_LINEAR)
        mask = full >= thr
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)
        out.append(dict(box=box, mask=mask, cls=int(c),
                        name=CLASS_NAMES.get(c, str(c)),
                        score=float(full[mask].max()), source=SOURCE))
    return out
