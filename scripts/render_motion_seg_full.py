"""Full dense motion segmentation: dense geometric core + temporal + object recall.

Pipeline per pair (t, t+1), two-pass:
  pass 1  dense per-pixel classify (motion_seg) -> despeckle
          + OBJECT RECALL: Mask R-CNN movers on both frames, IoU-tracked, flagged
            MOVING by the ego-normalised closing test (detect.object_moving_mask);
            their masks stamped MOVING onto the per-pair label BEFORE voting so
            they inherit the temporal persistence.
  vote    temporal +/-N (motion_seg.temporal_vote)
  pass 2  render static=green / moving=red / unknown=gray

The dense classifier is the precision core; the detector adds RECALL for fast
along-epipolar movers (the oncoming car) it is structurally blind to.
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

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")
RHO_STEREO = SRC / "inv_depth_fused_rho_stereo_16bit.mkv"
DENSE_DYN_FRAC = 0.3   # if the dense core marks > this frac of an object mask MOVING,
                       # the track is dynamic too (recall fallback for far movers)

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


def draw_boxes(img, boxes):
    """Draw the mover detections: red box = dynamic (latched), orange = pot_dyn."""
    for box, name, dynamic in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        col = (0, 0, 255) if dynamic else (0, 140, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        tag = f"{name} {'DYN' if dynamic else 'pot_dyn'}"
        cv2.putText(img, tag, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)
    return img


def banner(img, t, N, frac, window):
    lines = [
        f"frame {t:03d}/{N-1}  pair {t}->{t+1}   dense + temporal +/-{window} + pot_dyn objects",
        f"static {frac[STATIC]:4.1f}  dynamic {frac[MOVING]:4.1f}  pot_dyn {frac[POT_DYN]:4.1f}"
        f"  unknown {frac[UNKNOWN]:4.1f}   [green=static red=dynamic orange=pot_dyn gray=unknown]",
    ]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (880, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SRC / "motion_seg_full.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--backend", default="sea_raft")
    ap.add_argument("--window", type=int, default=TEMP_WINDOW)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--no-objects", action="store_true", help="disable the object recall layer")
    ap.add_argument("--display-camera", default="image_2", help="RGB camera for the render base")
    args = ap.parse_args()

    left = load_sequence(SEQ_DIR, camera="image_0")       # geometry frame (gray)
    disp = load_sequence(SEQ_DIR, camera=args.display_camera)  # RGB display base
    K, W, H, N = left.K, left.width, left.height, left.n_frames
    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")
    end = N if args.end < 0 else min(args.end, N)

    def bgr(i):  # geometry/detection base: grayscale image_0 (consistent with pose+rho)
        return cv2.cvtColor(cv2.imread(str(left.image_paths[i]), cv2.IMREAD_GRAYSCALE),
                            cv2.COLOR_GRAY2BGR)

    def disp_bgr(i):  # render base: RGB color frame
        im = cv2.imread(str(disp.image_paths[i]), cv2.IMREAD_COLOR)
        return im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H))

    # --- pass 1: dense labels + flows + tracked detections --------------------
    # The object layer is a separate post-vote overlay (it carries the latching
    # pot_dyn/dynamic track state), so it is NOT stamped into the dense label here.
    tracker = det.Tracker()
    labels, fwds, bwds, frame_dets = [], [], [], []
    rit = rho_stream(RHO_STEREO, W, H)
    rho_a = next(rit)
    for _ in range(args.start):
        rho_a = next(rit, rho_a)
    a = bgr(args.start)
    dets_a, tids_a = ([], [])
    if not args.no_objects:
        dets_a = det.detect(a)
        tids_a = tracker.update(dets_a, args.start)
    for t in tqdm(range(args.start, end - 1), desc="pass1 dense+track"):
        b = bgr(t + 1)
        rho_b = next(rit, rho_a)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        f_obs = compute_flow_pair(a, b, backend=args.backend)
        f_ba = compute_flow_pair(b, a, backend=args.backend)
        g = compute_fields(f_obs, f_ba, rho_a, rho_b, K, T_rel)
        lab = despeckle(classify(g), g)

        dets_this = []
        if not args.no_objects:
            dets_b = det.detect(b)
            tids_b = tracker.update(dets_b, t + 1)
            v_ego = float(np.linalg.norm(T_rel[:3, 3]))
            box_b = {tids_b[k]: dets_b[k]["box"] for k in range(len(dets_b))}
            for k, da in enumerate(dets_a):
                tid = tids_a[k]
                # confirm dynamic by (i) the ego-normalised closing test, or
                # (ii) the dense core firing on the object mask (far-mover recall)
                if tid in box_b:
                    cr = det.closing_ratio(da["box"], box_b[tid], rho_a, rho_b, v_ego)
                    if det.is_dynamic_closing(cr):
                        tracker.mark_dynamic(tid)
                m = da["mask"]
                if m.any() and (lab[m] == MOVING).mean() > DENSE_DYN_FRAC:
                    tracker.mark_dynamic(tid)
                dets_this.append((m, da["box"], det.category_name(da["cls"]), tid))
            dets_a, tids_a = dets_b, tids_b

        labels.append(lab)
        fwds.append(f_obs.astype(np.float16))
        bwds.append(f_ba.astype(np.float16))
        frame_dets.append(dets_this)
        a, rho_a = b, rho_b

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
            dynamic = tracker.is_dynamic(tid)   # latched over the whole track
            final[m] = MOVING if dynamic else POT_DYN
            boxes.append((box, name, dynamic))
        frac = {c: 100.0 * (final == c).sum() / final.size for c in COLORS}
        for c in mean:
            mean[c] += frac[c] / len(voted)
        frame = overlay(disp_bgr(t), final, args.alpha)
        frame = draw_boxes(frame, boxes)
        writer.write(banner(frame, t, N, frac, args.window))
    writer.release()
    print(f"[motion-seg full] wrote {args.out}")
    print(f"  mean: static {mean[STATIC]:.1f}  dynamic {mean[MOVING]:.2f}  "
          f"unknown {mean[UNKNOWN]:.1f}  pot_dyn {mean[POT_DYN]:.2f}")


if __name__ == "__main__":
    main()
