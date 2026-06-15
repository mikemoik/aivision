"""Dynamic step (dense motion segmentation) — sits between segmentation and lanes.

Lives outside ``mapgen.py`` so importing the framework never drags in cv2 / torch
/ the Mask R-CNN toolchain; the heavy imports stay inside the builder, which is
only called from :meth:`DynamicStep.process`. Registered as
``(dynamic, motion-disparity)``.

Refactors the validated two-pass pipeline of ``scripts/render_motion_seg_full.py``
into the map-gen role DAG. The one change vs the script: flow is READ from the
upstream ``flow`` step's cache (forward + backward per pair) instead of being
recomputed — that is the whole point of the separate flow role.

Pipeline (per consecutive pair ``t, t+1``):
  1. dense per-pixel motion-disparity classify (``motion_seg.compute_fields`` →
     ``classify`` → ``despeckle``) — the precision core: epipolar (depth-free) +
     looming (stereo-rho) residuals vs the rigid ego-motion prediction;
  2. OBJECT RECALL — Mask R-CNN movers (``detect``) IoU-tracked across frames; a
     track is latched dynamic from its median ego-normalised closing ratio +
     dense-MOVING fraction (recall for fast along-epipolar movers the dense core
     is structurally blind to);
  3. temporal vote ±window (``motion_seg.temporal_vote``, flow-warped);
  4. stamp each tracked object MOVING (latched dynamic) or POT_DYN.

All geometry is in the ``image_0`` frame (pose + stereo-rho + flow consistent).

Inputs: ``input:odometry`` (image_0 + calib), ``slam`` (poses), ``depth``
(stereo-only rho mkv — mono fill hallucinates sky depth, so the motion path must
read the stereo stream), ``flow`` (cached fwd/bwd), ``segmentation`` (optional;
only when ``fuse_segmenter`` folds Mask2Former movable classes into pot_dyn).

Out contract (``map-gen/dynamic/out/``):
  * ``dynamic_NNNNNN.npz`` — one per source frame ``t`` (pairs 0 … N-2), key
    ``label`` = (H, W) uint8 ∈ {0 static, 1 moving, 2 unknown, 3 pot_dyn}
  * ``manifest.json`` — {n_frames, n_pairs, h, w, classes, window, do_objects,
    fuse_segmenter, mean class fractions}
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .mapgen import MapGenContext, MapGenStep, register_step

CLASS_NAMES = {0: "static", 1: "moving", 2: "unknown", 3: "pot_dyn"}

# Mapillary-Vistas movable-object classes folded into pot_dyn when fuse_segmenter
# is on (flicker-gated against the sky→Car mislabel). Same set as the full script.
SEG_MOVABLE_NAMES = frozenset({
    "Bird", "Ground Animal", "Person", "Bicyclist", "Motorcyclist", "Other Rider",
    "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails",
    "Other Vehicle", "Trailer", "Truck", "Wheeled Slow",
})


def _poses_from_trajectory(traj_json: Path) -> np.ndarray:
    rows = np.asarray(json.loads(traj_json.read_text())["poses"], dtype=np.float64)
    n = rows.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = rows.reshape(n, 3, 4)
    return out


def _rho_stream(video: Path, w: int, h: int):
    """Yield per-frame stereo inverse-depth (rho) maps from the 16-bit mkv.

    Encoding (shared with the depth step): pixel 0 = invalid, pixel ∈ [1, 65535]
    → ``rho = (pixel - 1) / 65534`` (1/m); invalid → 0.
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE)
    fb = w * h * 2
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            pix = np.frombuffer(raw, np.uint16).reshape(h, w).astype(np.float32)
            yield np.where(pix > 0, (pix - 1.0) / 65534.0, 0.0)
    finally:
        proc.stdout.close()
        proc.wait()


def build_dynamic_seg(
    image_paths: list[Path],
    rho_stereo_path: Path,
    poses: np.ndarray,
    K: np.ndarray,
    flow_dir: Path,
    out_dir: Path,
    *,
    window: int | None = None,
    do_objects: bool = True,
    fuse_segmenter: bool = False,
    seg_cache_dir: Path | None = None,
    log=None,
) -> dict:
    """Dense motion segmentation over the whole sequence; cache per-frame labels."""
    import cv2

    from .. import detect as det
    from ..motion_seg import (
        MOVING, POT_DYN, STATIC, TEMP_WINDOW, UNKNOWN,
        classify, compute_fields, despeckle, temporal_vote,
    )

    def _say(msg: str) -> None:
        (log.info if log is not None else print)(msg)

    window = TEMP_WINDOW if window is None else window
    n = len(image_paths)
    if len(poses) < n:
        raise ValueError(f"poses ({len(poses)}) fewer than frames ({n})")

    def _bgr(p: Path) -> np.ndarray:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    h, w = _bgr(image_paths[0]).shape[:2]

    seg_ids, id2lab = None, {}
    if fuse_segmenter:
        if seg_cache_dir is None:
            raise ValueError("fuse_segmenter=True needs seg_cache_dir")
        from ..perception import CacheManifest, MANIFEST_NAME, id2label_from_manifest
        man = CacheManifest.load(seg_cache_dir / MANIFEST_NAME)
        id2lab = id2label_from_manifest(man)
        l2i = {v: k for k, v in id2lab.items()}
        seg_ids = [l2i[nm] for nm in SEG_MOVABLE_NAMES if nm in l2i]

    def _load_flow(t: int) -> tuple[np.ndarray, np.ndarray]:
        d = np.load(flow_dir / f"flow_{t:06d}.npz")
        return d["fwd"].astype(np.float32), d["bwd"].astype(np.float32)

    _say(f"[dynamic] {n} frames {w}x{h}  window=±{window}  objects={do_objects} "
         f"fuse_segmenter={fuse_segmenter}  flow←{flow_dir.name}")

    # --- pass 1: dense labels + (cached) flows + tracked detections -----------
    tracker = det.Tracker() if do_objects else None
    labels, fwds, bwds, frame_dets, valids = [], [], [], [], []
    rit = _rho_stream(rho_stereo_path, w, h)
    rho_a = next(rit)
    a = _bgr(image_paths[0])
    dets_a, tids_a = [], []
    if do_objects:
        dets_a = det.detect(a)
        tids_a = tracker.update(dets_a, 0)

    for t in range(n - 1):
        b = _bgr(image_paths[t + 1])
        rho_b = next(rit, rho_a)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        f_obs, f_ba = _load_flow(t)
        g = compute_fields(f_obs, f_ba, rho_a, rho_b, K, T_rel)
        lab = despeckle(classify(g), g)

        dets_this = []
        if do_objects:
            dets_b = det.detect(b)
            tids_b = tracker.update(dets_b, t + 1)
            v_ego = float(np.linalg.norm(T_rel[:3, 3]))
            box_b = {tids_b[k]: dets_b[k]["box"] for k in range(len(dets_b))}
            for k, da in enumerate(dets_a):
                tid = tids_a[k]
                if tid in box_b:
                    tracker.add_ratio(tid, det.closing_ratio(
                        da["box"], box_b[tid], rho_a, rho_b, v_ego))
                m = da["mask"]
                if m.any():
                    tracker.add_frac(tid, float((lab[m] == MOVING).mean()))
                dets_this.append((m, da["box"], det.category_name(da["cls"]), tid))
            dets_a, tids_a = dets_b, tids_b

        labels.append(lab)
        fwds.append(f_obs.astype(np.float16))
        bwds.append(f_ba.astype(np.float16))
        frame_dets.append(dets_this)
        valids.append(g.valid.copy() if fuse_segmenter else None)
        a, rho_a = b, rho_b
        if t % 50 == 0:
            _say(f"[dynamic] pass1 pair {t}/{n - 1}")

    if do_objects:
        tracker.finalize()  # latch each track's dynamic verdict (robust median)

    # --- temporal vote --------------------------------------------------------
    voted = temporal_vote(labels,
                          [f.astype(np.float32) for f in fwds],
                          [f.astype(np.float32) for f in bwds], window=window)

    # --- stamp objects + (optional) segmenter fusion, then write the cache ----
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = {STATIC: 0.0, MOVING: 0.0, UNKNOWN: 0.0, POT_DYN: 0.0}
    boxes_by_frame: list = []  # per-frame tracked mover boxes + latched verdict
    for i in range(len(voted)):
        final = voted[i].copy()
        boxes = []
        for m, box, name, tid in frame_dets[i]:
            dynamic = tracker.is_dynamic(tid)
            final[m] = MOVING if dynamic else POT_DYN
            boxes.append({"box": [round(float(v), 1) for v in box], "name": name,
                          "state": "moving" if dynamic else "pot_dyn", "tid": int(tid)})
        boxes_by_frame.append(boxes)

        if fuse_segmenter:
            from ..perception import load_frame_entry
            sem = load_frame_entry(seg_cache_dir, i)["semantic"]
            if sem.shape != (h, w):
                sem = cv2.resize(sem, (w, h), interpolation=cv2.INTER_NEAREST)
            seg_mov = np.isin(sem, seg_ids)
            sky_like = (~valids[i]) & (final == STATIC)   # geometry says far-static
            final[seg_mov & ~sky_like & (final != MOVING)] = POT_DYN

        np.savez_compressed(out_dir / f"dynamic_{i:06d}.npz", label=final.astype(np.uint8))
        for c in mean:
            mean[c] += 100.0 * (final == c).sum() / final.size / len(voted)

    # Per-frame mover boxes (Mask R-CNN, track-latched) — small structured
    # sidecar for box-mode renders ("draw boxes, tint only inside them").
    (out_dir / "dynamic_boxes.json").write_text(json.dumps(
        {"n_pairs": len(voted), "frame": "image_0", "frames": boxes_by_frame}))

    manifest = {
        "n_frames": n,
        "n_pairs": len(voted),
        "h": h,
        "w": w,
        "classes": CLASS_NAMES,
        "index": "dynamic_{t:06d}.npz holds the pair (t, t+1) label; t in 0..n_pairs-1",
        "window": window,
        "do_objects": do_objects,
        "fuse_segmenter": fuse_segmenter,
        "mean_pct": {CLASS_NAMES[c]: round(mean[c], 3) for c in mean},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _say(f"[dynamic] wrote {len(voted)} npz + manifest  mean%: "
         f"static={mean[STATIC]:.1f} moving={mean[MOVING]:.2f} "
         f"pot_dyn={mean[POT_DYN]:.2f} unknown={mean[UNKNOWN]:.1f}")
    return manifest


@register_step
class DynamicStep(MapGenStep):
    """Dense motion segmentation (static/moving/unknown/pot_dyn) per frame.

    Reads cached flow + the stereo-only rho + SLAM poses, runs the dense
    motion-disparity classifier with Mask R-CNN object recall and a temporal
    vote, and emits a per-frame label cache. Does not (yet) feed lanes/objects —
    it is a standalone perception layer for now.
    """

    role = "dynamic"
    type = "motion-disparity"
    inputs = ["input:odometry", "slam", "depth", "flow", "segmentation"]

    def process(self, ctx: MapGenContext) -> list[str]:
        from ..kitti import load_sequence

        seq_dir = ctx.input("odometry")
        left = load_sequence(seq_dir, camera="image_0")
        poses = _poses_from_trajectory(ctx.upstream_out("slam") / "trajectory.json")
        rho_stereo = ctx.upstream_out("depth") / "inv_depth_rho16_stereo.mkv"
        flow_dir = ctx.upstream_out("flow")
        fuse = bool(self.params.get("fuse_segmenter", False))
        ctx.log.info("dynamic: seq=%s rho=%s flow=%s", seq_dir, rho_stereo.name, flow_dir.name)

        manifest = build_dynamic_seg(
            left.image_paths, rho_stereo, poses, left.K, flow_dir, ctx.out("dynamic"),
            window=self.params.get("window"),
            do_objects=bool(self.params.get("do_objects", True)),
            fuse_segmenter=fuse,
            seg_cache_dir=ctx.upstream_out("segmentation") if fuse else None,
            log=ctx.log,
        )
        ctx.log.info("dynamic: %d pairs labelled", manifest["n_pairs"])
        return [
            "map-gen/dynamic/out/manifest.json",
            "map-gen/dynamic/out/dynamic_boxes.json",
            f"map-gen/dynamic/out/dynamic_NNNNNN.npz ({manifest['n_pairs']} pairs)",
        ]
