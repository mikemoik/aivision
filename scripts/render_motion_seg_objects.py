"""Dense motion segmentation + ALL detected objects framed blue.

Same dense-core + temporal + mover-recall pipeline as render_motion_seg_full, plus
a third object layer: every detected non-mover OBJECT -- so not pot_dyn / dyn -- is
drawn with a BLUE box AND its pixels overlaid blue, semi-transparent. This visually
separates the static scene furniture from the orange (pot_dyn) / red (dyn) movers.

Sources for the blue layer (COCO Mask R-CNN has NO pole and only `stop sign`, so it
flickers poles as fire hydrants -- we fix that with the segmenter):
  * Mapillary-Vistas semantic (perception/ cache): poles, traffic signs (front/back),
    sign frames, traffic/street lights, billboards, hydrants, ... -- per-pixel, stable,
    correctly labelled. Boxes from connected components (--seg-min-area filters specks).
  * COCO Mask R-CNN: the non-mover classes it labels WELL (bench, potted plant, ...).
    Its pole/sign confusions (fire hydrant / parking meter / stop sign / traffic light)
    are SUPPRESSED -- the segmenter owns those.

Colour key:
  green  = static (dense)      red   = dynamic mover (latched)
  gray   = unknown (dense)     orange= potentially-dynamic mover
  blue   = detected non-mover object (box + pixels)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.flow import compute_flow_pair
from aivision.kitti import load_sequence
from aivision.pocs import common
from aivision.motion_seg import (
    compute_fields, classify, despeckle, temporal_vote,
    STATIC, MOVING, UNKNOWN, POT_DYN, TEMP_WINDOW,
)
from aivision import detect as det
from aivision.perception import (
    CacheManifest, MANIFEST_NAME, id2label_from_manifest, load_frame_entry,
)

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")

BLUE = (255, 60, 0)  # BGR -- the detected-object colour

# Mapillary-Vistas discrete street-furniture classes the segmenter labels well and
# COCO lacks (or confuses) -> the semantic blue layer. NB: Bench is deliberately
# absent -> COCO owns it (it detects benches cleanly), avoiding double boxes.
SEG_OBJECT_NAMES = frozenset({
    "Pole", "Utility Pole", "Street Light", "Traffic Light",
    "Traffic Sign (Front)", "Traffic Sign (Back)", "Traffic Sign Frame",
    "Billboard", "Banner", "Fire Hydrant", "Junction Box", "Mailbox",
    "Phone Booth", "CCTV Camera", "Catch Basin", "Bike Rack", "Trash Can",
})
# COCO non-mover classes that are really pole/sign confusions now owned by the
# semantic layer -> dropped from the COCO 'others' to kill the flicker.
COCO_SUPPRESS = frozenset({"fire hydrant", "parking meter", "stop sign", "traffic light"})

COLORS = {
    STATIC:  np.array([0, 200, 0], np.float32),     # green
    MOVING:  np.array([0, 0, 255], np.float32),     # red    (= dynamic)
    UNKNOWN: np.array([130, 130, 130], np.float32),  # gray
    POT_DYN: np.array([0, 140, 255], np.float32),    # orange (potentially dynamic)
}


def rho_stream(video, W, H):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE)
    fb = W * H * 2
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            pix = np.frombuffer(raw, np.uint16).reshape(H, W).astype(np.float32)
            yield np.where(pix > 0, (pix - 1.0) / 65534.0, 0.0)
    finally:
        proc.stdout.close()
        proc.wait()


def overlay(bgr, label, alpha=0.5):
    out = bgr.astype(np.float32)
    color = np.zeros_like(out)
    for cls, col in COLORS.items():
        color[label == cls] = col
    return (out * (1 - alpha) + color * alpha).astype(np.uint8)


def overlay_blue(img, mask, alpha=0.45):
    """Semi-transparent blue over the union of detected-object pixels."""
    if not mask.any():
        return img
    out = img.astype(np.float32)
    col = np.array(BLUE, np.float32)
    out[mask] = out[mask] * (1 - alpha) + col * alpha
    return out.astype(np.uint8)


def draw_boxes(img, boxes):
    """Mover detections: red box = dynamic (latched), orange = pot_dyn."""
    for box, name, dynamic in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        col = (0, 0, 255) if dynamic else (0, 140, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        tag = f"{name} {'DYN' if dynamic else 'pot_dyn'}"
        cv2.putText(img, tag, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)
    return img


def draw_object_boxes(img, objs):
    """Non-mover detected objects -> blue box + class label."""
    for box, name in objs:
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 1)
        cv2.putText(img, name, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, BLUE, 1, cv2.LINE_AA)
    return img


def banner(img, t, N, frac, n_obj, window):
    lines = [
        f"frame {t:03d}/{N-1}  pair {t}->{t+1}   dense + temporal +/-{window} + objects",
        f"static {frac[STATIC]:4.1f}  dynamic {frac[MOVING]:4.1f}  pot_dyn {frac[POT_DYN]:4.1f}"
        f"  unknown {frac[UNKNOWN]:4.1f}   objects(blue) {n_obj}",
        "green=static red=dyn orange=pot_dyn gray=unknown  blue=detected object (semantic+COCO)",
    ]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (960, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=SEQ_DIR)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--perception", type=Path, default=None,
                    help="perception/ cache dir (default <src>/perception)")
    ap.add_argument("--no-segmenter", action="store_true",
                    help="disable the Mapillary semantic object layer (COCO only)")
    ap.add_argument("--seg-min-area", type=int, default=60,
                    help="min connected-component area (px) for a semantic object box")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--backend", default="sea_raft")
    ap.add_argument("--window", type=int, default=TEMP_WINDOW)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--obj-alpha", type=float, default=0.45,
                    help="blend for the blue detected-object pixel overlay")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--display-camera", default="image_2", help="RGB camera for the render base")
    args = ap.parse_args()

    src = args.src
    rho_stereo = src / "inv_depth_fused_rho_stereo_16bit.mkv"
    if args.out is None:
        args.out = src / "motion_seg_objects.mp4"
    if args.perception is None:
        args.perception = src / "perception"

    left = load_sequence(args.seq_dir, camera="image_0")       # geometry frame (gray)
    disp = load_sequence(args.seq_dir, camera=args.display_camera)  # display base
    K, W, H, N = left.K, left.width, left.height, left.n_frames
    poses = common.load_kitti_trajectory(src / "CameraTrajectory.txt")
    end = N if args.end < 0 else min(args.end, N)

    seg_ids, id2lab = None, {}
    if not args.no_segmenter:
        man = CacheManifest.load(args.perception / MANIFEST_NAME)
        id2lab = id2label_from_manifest(man)
        l2i = {v: k for k, v in id2lab.items()}
        seg_ids = [l2i[n] for n in SEG_OBJECT_NAMES if n in l2i]

    def semantic(i):
        sem = load_frame_entry(args.perception, i)["semantic"]
        return sem if sem.shape == (H, W) else cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)

    def bgr(i):  # geometry/detection base: grayscale image_0 (consistent with pose+rho)
        return cv2.cvtColor(cv2.imread(str(left.image_paths[i]), cv2.IMREAD_GRAYSCALE),
                            cv2.COLOR_GRAY2BGR)

    def disp_bgr(i):  # render base: RGB color frame
        im = cv2.imread(str(disp.image_paths[i]), cv2.IMREAD_COLOR)
        return im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H))

    # --- pass 1: dense labels + flows + tracked movers + non-mover COCO objects -
    tracker = det.Tracker()
    labels, fwds, bwds, frame_dets, frame_objs = [], [], [], [], []
    rit = rho_stream(rho_stereo, W, H)
    rho_a = next(rit)
    for _ in range(args.start):
        rho_a = next(rit, rho_a)
    a = bgr(args.start)
    dets_a, oth_a = det.infer(a)
    tids_a = tracker.update(dets_a, args.start)
    for t in tqdm(range(args.start, end - 1), desc="pass1 dense+detect"):
        b = bgr(t + 1)
        rho_b = next(rit, rho_a)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        f_obs = compute_flow_pair(a, b, backend=args.backend)
        f_ba = compute_flow_pair(b, a, backend=args.backend)
        g = compute_fields(f_obs, f_ba, rho_a, rho_b, K, T_rel)
        lab = despeckle(classify(g), g)

        dets_b, oth_b = det.infer(b)
        tids_b = tracker.update(dets_b, t + 1)
        v_ego = float(np.linalg.norm(T_rel[:3, 3]))
        box_b = {tids_b[k]: dets_b[k]["box"] for k in range(len(dets_b))}
        dets_this = []
        for k, da in enumerate(dets_a):
            tid = tids_a[k]
            if tid in box_b:
                tracker.add_ratio(tid, det.closing_ratio(
                    da["box"], box_b[tid], rho_a, rho_b, v_ego))
            m = da["mask"]
            if m.any():
                tracker.add_frac(tid, float((lab[m] == MOVING).mean()))
            dets_this.append((m, da["box"], det.category_name(da["cls"]), tid))
        # non-mover COCO objects for frame t (the 'a' frame): drop COCO's pole/sign
        # confusions -- the segmenter owns those.
        objs_this = [(o["mask"], o["box"], det.category_name(o["cls"]))
                     for o in oth_a if det.category_name(o["cls"]) not in COCO_SUPPRESS]
        dets_a, tids_a, oth_a = dets_b, tids_b, oth_b

        labels.append(lab)
        fwds.append(f_obs.astype(np.float16))
        bwds.append(f_ba.astype(np.float16))
        frame_dets.append(dets_this)
        frame_objs.append(objs_this)
        a, rho_a = b, rho_b

    tracker.finalize()

    # --- temporal vote --------------------------------------------------------
    voted = temporal_vote([l for l in labels],
                          [f.astype(np.float32) for f in fwds],
                          [f.astype(np.float32) for f in bwds], window=args.window)

    # --- pass 2 ---------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"),
                             args.fps, (W, H))
    mean = {STATIC: 0.0, MOVING: 0.0, UNKNOWN: 0.0, POT_DYN: 0.0}
    for i, t in enumerate(tqdm(range(args.start, end - 1), desc="pass2 render")):
        final = voted[i].copy()
        boxes = []
        for m, box, name, tid in frame_dets[i]:
            dynamic = tracker.is_dynamic(tid)
            final[m] = MOVING if dynamic else POT_DYN
            boxes.append((box, name, dynamic))

        obj_mask = np.zeros((H, W), bool)
        obj_boxes = []
        # (a) COCO non-mover instance masks/boxes
        for m, box, name in frame_objs[i]:
            obj_mask |= m
            obj_boxes.append((box, name))
        # (b) Mapillary semantic static-object pixels + connected-component boxes
        if seg_ids is not None:
            sem = semantic(t)
            seg_obj = np.isin(sem, seg_ids)
            obj_mask |= seg_obj
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(seg_obj.astype(np.uint8), 8)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < args.seg_min_area:
                    continue
                x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
                w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
                cid = int(np.bincount(sem[lbl == k]).argmax())
                obj_boxes.append(((x, y, x + w, y + h), id2lab.get(cid, str(cid))))
        # movers (red/orange) take precedence over the blue layer
        obj_mask &= (final != MOVING) & (final != POT_DYN)

        frac = {c: 100.0 * (final == c).sum() / final.size for c in COLORS}
        for c in mean:
            mean[c] += frac[c] / len(voted)

        frame = overlay(disp_bgr(t), final, args.alpha)
        frame = overlay_blue(frame, obj_mask, args.obj_alpha)
        frame = draw_boxes(frame, boxes)
        frame = draw_object_boxes(frame, obj_boxes)
        writer.write(banner(frame, t, N, frac, len(obj_boxes), args.window))
    writer.release()
    print(f"[motion-seg objects] wrote {args.out}")
    print(f"  mean: static {mean[STATIC]:.1f}  dynamic {mean[MOVING]:.2f}  "
          f"unknown {mean[UNKNOWN]:.1f}  pot_dyn {mean[POT_DYN]:.2f}")


if __name__ == "__main__":
    main()
