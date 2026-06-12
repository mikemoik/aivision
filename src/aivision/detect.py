"""Per-frame object detection + tracking — the motion-seg RECALL layer.

The dense geometric classifier (motion_seg) is structurally blind to an object
that moves ALONG its epipolar lines while stereo-invalid and fast (the fast
oncoming car: epi~0, no stereo depth, flow >FLOW_MAX). A COCO Mask R-CNN detects
such objects per frame regardless of displacement; tracking them by IoU and
measuring the EGO-NORMALISED closing rate flags the real movers:

    closing_ratio = -dZ_obs / v_ego        Z = ground-contact depth (road stereo-rho
                                            under the box bottom)
    static object   -> closes at exactly ego speed -> ratio ~ 1   (NOT moving)
    co-moving leader-> holds distance              -> ratio ~ 0   (moving)
    oncoming car    -> closes faster than ego      -> ratio > 1   (moving)
  => |closing_ratio - 1| > CLOSING_MARGIN  AND reliable ground-Z  -> MOVING

A flagged object's MASK pixels are stamped MOVING over the dense map. This is
RECALL ONLY for detected classes -- it adds moving, never overrides the dense
static/sky precision, and parked cars (ratio ~ 1) stay static.
"""

from __future__ import annotations

import numpy as np
import cv2

MOVER_NAMES = ("car", "truck", "bus", "person", "bicycle", "motorcycle", "train")
SCORE_THR = 0.5
MASK_THR = 0.5
CLOSING_MARGIN = 0.5      # |closing_ratio - 1| above this -> moving
Z_MIN, Z_MAX = 4.0, 50.0  # plausible ground-contact depth (m); far Z is too noisy
GROUND_MIN_PX = 20        # min valid road-rho samples under a box to trust its Z
MAX_DZ_EGO = 4.0          # reject |dZ| > this * v_ego as a ground-Z misread

_state: dict = {}


def load_detector():
    """Lazy-load COCO Mask R-CNN; cache for the process. Returns (model, tf, movers, dev)."""
    if not _state:
        import torch
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights as Wt)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        w = Wt.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=w).eval().to(dev)
        cats = w.meta["categories"]
        _state.update(model=model, tf=w.transforms(), cats=cats,
                      movers={cats.index(n) for n in MOVER_NAMES if n in cats}, dev=dev)
    return _state["model"], _state["tf"], _state["movers"], _state["dev"]


def category_name(cls: int) -> str:
    return _state["cats"][cls] if _state else str(cls)


def detect(img_bgr: np.ndarray) -> list[dict]:
    """Detect mover-class instances -> [{box(x0,y0,x1,y1), mask HxW bool, cls, score}]."""
    import torch
    model, tf, movers, dev = load_detector()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = tf(torch.from_numpy(rgb).permute(2, 0, 1)).to(dev)
    with torch.no_grad():
        o = model([x])[0]
    out = []
    for box, lab, sc, mask in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                                  o["scores"].cpu().numpy(), o["masks"].cpu().numpy()):
        if sc < SCORE_THR or int(lab) not in movers:
            continue
        out.append(dict(box=box.astype(np.float64), mask=mask[0] > MASK_THR,
                        cls=int(lab), score=float(sc)))
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def ground_depth(rho: np.ndarray, box: np.ndarray) -> float:
    """Depth (m) at the box's ground contact: median 1/rho in a strip at the box bottom."""
    H, W = rho.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    cx = (x0 + x1) // 2
    ys = slice(max(0, y1 - 4), min(H, y1 + 6))
    xs = slice(max(0, cx - 15), min(W, cx + 15))
    r = rho[ys, xs]
    r = r[r > 0]
    if r.size < GROUND_MIN_PX:
        return np.nan
    return float(1.0 / np.median(r))


def closing_ratio(box_a: np.ndarray, box_b: np.ndarray,
                  rho_a: np.ndarray, rho_b: np.ndarray, v_ego: float):
    """Ego-normalised closing rate of a tracked box between two frames, or None.

    -dZ/v_ego of the ground-contact depth: ~1 for a static object (closes at ego
    speed), ~0 for a co-moving leader, >1 for an oncoming car. None when the
    ground-Z is unreliable (far, off-road box bottom, or a depth jump/mismatch).
    """
    Za, Zb = ground_depth(rho_a, box_a), ground_depth(rho_b, box_b)
    if not (np.isfinite(Za) and np.isfinite(Zb)):
        return None
    if not (Z_MIN < Za < Z_MAX and Z_MIN < Zb < Z_MAX):
        return None
    if abs(Zb - Za) > MAX_DZ_EGO * max(v_ego, 1e-3):
        return None
    return -(Zb - Za) / max(v_ego, 1e-3)


def is_dynamic_closing(ratio) -> bool:
    """A measured closing ratio that deviates from the static baseline (1)."""
    return ratio is not None and abs(ratio - 1.0) > CLOSING_MARGIN


class Tracker:
    """Greedy-IoU multi-frame tracker carrying a latching ``dynamic`` flag.

    Mover-class detections are stitched into tracks across frames so a track can
    accumulate state: once marked dynamic (by the closing test OR the dense core)
    it stays dynamic for life -- a momentary stop never reverts it. Per the
    pot_dyn principle, a track that is never confirmed dynamic stays POT_DYN, not
    static.
    """

    def __init__(self, iou_thr: float = 0.2, max_gap: int = 5):
        self.tracks: dict[int, dict] = {}
        self.next_id = 0
        self.iou_thr = iou_thr
        self.max_gap = max_gap

    def update(self, dets: list[dict], t: int) -> list[int]:
        """Assign a track id to each detection of frame t; returns ids aligned with dets."""
        active = [(tid, tr) for tid, tr in self.tracks.items() if t - tr["last_t"] <= self.max_gap]
        pairs = []
        for di, d in enumerate(dets):
            for tid, tr in active:
                i = box_iou(d["box"], tr["box"])
                if i >= self.iou_thr:
                    pairs.append((i, di, tid))
        pairs.sort(key=lambda p: -p[0])
        assigned: list[int | None] = [None] * len(dets)
        used_t: set[int] = set()
        for _, di, tid in pairs:
            if assigned[di] is None and tid not in used_t:
                assigned[di] = tid
                used_t.add(tid)
        for di, d in enumerate(dets):
            tid = assigned[di]
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": d["box"], "last_t": t, "dynamic": False}
            else:
                self.tracks[tid]["box"] = d["box"]
                self.tracks[tid]["last_t"] = t
            assigned[di] = tid
        return assigned  # type: ignore[return-value]

    def mark_dynamic(self, tid: int) -> None:
        if tid in self.tracks:
            self.tracks[tid]["dynamic"] = True

    def is_dynamic(self, tid: int) -> bool:
        return self.tracks.get(tid, {}).get("dynamic", False)
