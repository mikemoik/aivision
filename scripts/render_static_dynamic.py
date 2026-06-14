"""Full scene render: all STATIC objects (from their triangulated 3D maps, reprojected into
every frame) + all DYNAMIC objects (the motion-segmentation movers, per frame).

STATIC layer (reprojected 3D, drawn as glyphs) -- POLES ARE DELIBERATELY EXCLUDED (their poses
are not accurate enough; guideposts ARE kept):
  * road arrows      yellow  filled glyph   (outputs/static_env/arrows_lanemodel/arrows_world.json)
  * traffic signs    magenta box + class    (poles_signs_world.json -> signs)
  * traffic lights   green   housing        (poles_signs_world.json -> traffic_lights)
  * guideposts       white   stick          (poles_signs_world.json -> guideposts)

DYNAMIC layer (per frame, from aivision.motion_seg + COCO mover tracking, same two-pass pipeline
as render_motion_seg_full): moving objects painted + boxed -- red = dynamic (latched), orange =
pot_dyn. The dense static/unknown field is NOT painted, so the static glyphs stay legible.

Static maps were built on the same cam0->world poses (CameraTrajectory.txt) and the P2 (image_2)
intrinsics, and we render on image_2 -- so reprojection is consistent with the lift scripts.
"""

from __future__ import annotations

import argparse
import json
import re
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
ZNEAR = 3.0

ARROW = (0, 255, 255)      # yellow
SIGN = (255, 0, 255)       # magenta (red is taken by dynamic movers)
TLIGHT = (60, 220, 60)     # green
GUIDE = (245, 245, 245)    # white
DYN = (0, 0, 255)          # red
POTDYN = (0, 140, 255)     # orange
LANE_GREEN = (0, 200, 0)   # ego-lane fill
BORDER_BLACK = (0, 0, 0)   # lane borders


def read_calib(p: Path):
    out = {}
    for line in p.read_text().strip().splitlines():
        k, *vals = line.replace(":", "").split()
        out[k] = np.array(vals, np.float64).reshape(3, 4)
    return out


def sign_label(name: str) -> str:
    m = re.search(r"(\d+)\s*km/h", name)
    if m:
        return ("E" if name.lower().startswith("end") else "") + m.group(1)
    return "".join(w[0] for w in re.findall(r"[A-Za-z0-9]+", name))[:4].upper()


def project_pts(Pw, Rinv, C, fx, fy, cx, cy):
    """World points -> int pixels, or None if any vertex is behind the near plane."""
    Pc = (np.asarray(Pw, float) - C) @ Rinv.T
    if np.any(Pc[:, 2] <= ZNEAR):
        return None
    u = fx * Pc[:, 0] / Pc[:, 2] + cx
    v = fy * Pc[:, 1] / Pc[:, 2] + cy
    return np.round(np.stack([u, v], 1)).astype(np.int32)


def rho_stream(video, W, H):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
         "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE)
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


def densify_border(B, step=0.5):
    """Resample a border polyline (world [X,Y,Z], monotonic Z) onto a uniform Z grid via
    linear interpolation -- bridges gaps where fit_lane_model dropped Z-bins, so the overlay
    is continuous (and left/right stay index-aligned by sharing the grid)."""
    z = B[:, 2]
    zg = np.arange(float(z.min()), float(z.max()), step)
    return np.stack([np.interp(zg, z, B[:, 0]), np.interp(zg, z, B[:, 1]), zg], 1)


def draw_lane(im, Lb, Rb, Rinv, C, fx, fy, cx, cy, los_fill, los_border, alpha=0.35):
    """Green ego-lane fill (interior between borders) to los_fill; thin black border lines
    to los_border. Lb/Rb are the world-frame left/right border polylines (same Z sampling)."""
    def px(Pw):
        Pc = (Pw - C) @ Rinv.T
        u = fx * Pc[:, 0] / Pc[:, 2] + cx
        v = fy * Pc[:, 1] / Pc[:, 2] + cy
        return np.round(np.stack([u, v], 1)).astype(np.int32)

    PcL = (Lb - C) @ Rinv.T
    win = (PcL[:, 2] > ZNEAR) & (PcL[:, 2] < los_fill)
    if win.sum() >= 2:
        idx = np.where(win)[0][np.argsort(PcL[win, 2])]
        corridor = np.vstack([Lb[idx], Rb[idx][::-1]])      # closed lane polygon
        ov = im.copy()
        cv2.fillPoly(ov, [px(corridor)], LANE_GREEN)
        cv2.addWeighted(ov, alpha, im, 1 - alpha, 0, dst=im)
    for B in (Lb, Rb):                                       # thin black borders
        Pc = (B - C) @ Rinv.T
        w = (Pc[:, 2] > ZNEAR) & (Pc[:, 2] < los_border)
        if w.sum() >= 2:
            idx = np.where(w)[0][np.argsort(Pc[w, 2])]
            cv2.polylines(im, [px(B[idx])], False, BORDER_BLACK, 1, cv2.LINE_AA)
    return im


def draw_static(im, statics, Rinv, C, fx, fy, cx, cy, UP, los=100.0):
    """Reproject the static 3D objects into this view -- only those within `los` m ahead."""
    arrows, signs, tlights, guideposts = statics

    def zc(p):  # forward distance (m) of a world point/centroid from this camera
        return float((np.asarray(p, float) - C) @ Rinv[2])

    # arrows: filled yellow glyph polygon
    arrows_los = [g for g in arrows if ZNEAR < zc(np.mean(g, 0)) <= los]
    ov = im.copy()
    drew = False
    for g in arrows_los:
        poly = project_pts(g, Rinv, C, fx, fy, cx, cy)
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(ov, [poly], ARROW); drew = True
    if drew:
        cv2.addWeighted(ov, 0.45, im, 0.55, 0, dst=im)
    for g in arrows_los:
        poly = project_pts(g, Rinv, C, fx, fy, cx, cy)
        if poly is not None and len(poly) >= 3:
            cv2.polylines(im, [poly], True, ARROW, 2, cv2.LINE_AA)
    # guideposts: white stick
    for c, h in guideposts:
        if not (ZNEAR < zc(c) <= los):
            continue
        seg = project_pts(np.array([c - 0.5 * h * UP, c + 0.5 * h * UP]), Rinv, C, fx, fy, cx, cy)
        if seg is not None:
            cv2.line(im, tuple(seg[0]), tuple(seg[1]), GUIDE, 3, cv2.LINE_AA)
            cv2.circle(im, tuple(seg[1]), 3, (0, 0, 0), -1, cv2.LINE_AA)
    # traffic lights: green housing
    for c, h in tlights:
        Zc = zc(c)
        if not (ZNEAR < Zc <= los):
            continue
        seg = project_pts(np.array([c - 0.5 * h * UP, c + 0.5 * h * UP]), Rinv, C, fx, fy, cx, cy)
        if seg is not None:
            wpx = max(2, int(0.25 * h / Zc * fy))
            (u0, v0), (u1, v1) = seg[0], seg[1]
            cv2.rectangle(im, (u1 - wpx, v1), (u1 + wpx, v0), TLIGHT, 2, cv2.LINE_AA)
    # signs: magenta billboard + class tag
    for c, sz, name in signs:
        Zc = zc(c)
        if not (ZNEAR < Zc <= los):
            continue
        px = project_pts(c[None], Rinv, C, fx, fy, cx, cy)
        if px is None:
            continue
        rpx = int(0.5 * max(sz, 0.6) / Zc * fy)
        u, v = int(px[0, 0]), int(px[0, 1])
        cv2.rectangle(im, (u - rpx, v - rpx), (u + rpx, v + rpx), SIGN, 2, cv2.LINE_AA)
        cv2.putText(im, sign_label(name), (u - rpx, v - rpx - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, SIGN, 2, cv2.LINE_AA)
    return im


def banner(img, t, N):
    txt = ("ego-lane=green border=black | yellow=arrow magenta=sign green=light white=guidepost"
           " | red=moving orange=pot_dyn (in-box)   (poles excluded)")
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
    cv2.putText(img, f"frame {t:03d}/{N-1}   {txt}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def load_static_maps(arrows_json: Path, poles_signs_json: Path):
    arrows = [np.array(a["glyph_world"], float) for a in json.loads(arrows_json.read_text())]
    ps = json.loads(poles_signs_json.read_text())
    signs = [(np.array(s["center"], float), float(s["size"]), s["name"]) for s in ps.get("signs", [])]
    tlights = [(np.array(t["center"], float), float(t["height"])) for t in ps.get("traffic_lights", [])]
    guideposts = [(np.array(g["center"], float), float(g["height"])) for g in ps.get("guideposts", [])]
    # poles intentionally NOT loaded
    return arrows, signs, tlights, guideposts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=SEQ_DIR)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--arrows", type=Path,
                    default=Path("outputs/static_env/arrows_lanemodel/arrows_world.json"))
    ap.add_argument("--poles-signs", type=Path,
                    default=Path("outputs/static_env/poles_signs/poles_signs_world.json"))
    ap.add_argument("--lane-model", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_polylines_model.json"))
    ap.add_argument("--los-fill", type=float, default=40.0, help="ego-lane green fill horizon (m)")
    ap.add_argument("--los-border", type=float, default=50.0, help="lane border line horizon (m)")
    ap.add_argument("--los-static", type=float, default=100.0,
                    help="line of sight (m) for arrows/signs/lights/guideposts")
    ap.add_argument("--out", type=Path, default=Path("outputs/static_env/static_dynamic.mp4"))
    ap.add_argument("--dynamic-cache", type=Path,
                    default=Path("outputs/static_env/dynamic_cache.npz"),
                    help="per-frame mover masks+boxes; computed once (flow), then reused")
    ap.add_argument("--refresh-dynamic", action="store_true",
                    help="force re-extraction of the dynamic objects even if the cache exists")
    ap.add_argument("--bg-grey", action="store_true",
                    help="render on a plain grey canvas instead of the RGB camera stream")
    ap.add_argument("--bg-grey-until", type=int, default=-1,
                    help="grey canvas for frames < N, RGB from N onward")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--backend", default="sea_raft")
    ap.add_argument("--window", type=int, default=TEMP_WINDOW)
    ap.add_argument("--alpha", type=float, default=0.5, help="mover-mask overlay alpha")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    src = args.src
    rho_stereo = src / "inv_depth_fused_rho_stereo_16bit.mkv"
    if args.bg_grey:
        args.out = args.out.with_name(args.out.stem + "_bg_grey" + args.out.suffix)
    elif args.bg_grey_until > 0:
        args.out = args.out.with_name(f"{args.out.stem}_bg_grey_until{args.bg_grey_until}{args.out.suffix}")

    left = load_sequence(args.seq_dir, camera="image_0")        # geometry frame (gray)
    disp = load_sequence(args.seq_dir, camera="image_2")        # display base (RGB)
    K, W, H, N = left.K, left.width, left.height, left.n_frames
    poses = common.load_kitti_trajectory(src / "CameraTrajectory.txt")
    end = N if args.end < 0 else min(args.end, N)

    calib = read_calib(args.seq_dir / "calib.txt")
    P2 = calib["P2"][:3, :3]
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    up = -np.mean(np.array(poses)[:end, :3, 1], axis=0)
    UP = up / np.linalg.norm(up)
    statics = load_static_maps(args.arrows, args.poles_signs)
    lm = json.loads(args.lane_model.read_text())
    Lb, Rb = densify_border(np.array(lm["left"], float)), densify_border(np.array(lm["right"], float))
    print(f"[scene] static objects: {len(statics[0])} arrows, {len(statics[1])} signs, "
          f"{len(statics[2])} traffic lights, {len(statics[3])} guideposts (poles excluded); "
          f"ego-lane fill {args.los_fill:.0f}m, borders {args.los_border:.0f}m")

    def bgr(i):
        return cv2.cvtColor(cv2.imread(str(left.image_paths[i]), cv2.IMREAD_GRAYSCALE),
                            cv2.COLOR_GRAY2BGR)

    def disp_bgr(i):
        im = cv2.imread(str(disp.image_paths[i]), cv2.IMREAD_COLOR)
        return im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H))

    # --- dynamic objects: load from cache, else extract once (the only flow-heavy step) ----
    cache = args.dynamic_cache
    if cache and cache.exists() and not args.refresh_dynamic:
        npz = np.load(cache)
        if int(npz["start"]) != args.start or int(npz["end"]) != end:
            raise SystemExit(f"dynamic cache {cache} was built for start/end "
                             f"{int(npz['start'])}/{int(npz['end'])}, not {args.start}/{end}; "
                             f"pass --refresh-dynamic")
        mover_labels = list(npz["mover_labels"])
        boxes_per_frame = json.loads(cache.with_suffix(".boxes.json").read_text())
        print(f"[scene] loaded dynamic objects from cache {cache} ({len(mover_labels)} frames)")
    else:
        tracker = det.Tracker()
        labels, fwds, bwds, frame_dets = [], [], [], []
        rit = rho_stream(rho_stereo, W, H)
        rho_a = next(rit)
        for _ in range(args.start):
            rho_a = next(rit, rho_a)
        a = bgr(args.start)
        dets_a = det.detect(a)
        tids_a = tracker.update(dets_a, args.start)
        for t in tqdm(range(args.start, end - 1), desc="pass1 dense+track (flow)"):
            b = bgr(t + 1)
            rho_b = next(rit, rho_a)
            T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
            f_obs = compute_flow_pair(a, b, backend=args.backend)
            f_ba = compute_flow_pair(b, a, backend=args.backend)
            g = compute_fields(f_obs, f_ba, rho_a, rho_b, K, T_rel)
            lab = despeckle(classify(g), g)

            dets_b = det.detect(b)
            tids_b = tracker.update(dets_b, t + 1)
            v_ego = float(np.linalg.norm(T_rel[:3, 3]))
            box_b = {tids_b[k]: dets_b[k]["box"] for k in range(len(dets_b))}
            dets_this = []
            for k, da in enumerate(dets_a):
                tid = tids_a[k]
                if tid in box_b:
                    tracker.add_ratio(tid, det.closing_ratio(da["box"], box_b[tid], rho_a, rho_b, v_ego))
                m = da["mask"]
                if m.any():
                    tracker.add_frac(tid, float((lab[m] == MOVING).mean()))
                dets_this.append((m, da["box"], det.category_name(da["cls"]), tid))
            dets_a, tids_a = dets_b, tids_b

            labels.append(lab)
            fwds.append(f_obs.astype(np.float16))
            bwds.append(f_ba.astype(np.float16))
            frame_dets.append(dets_this)
            a, rho_a = b, rho_b

        tracker.finalize()
        voted = temporal_vote(list(labels), [f.astype(np.float32) for f in fwds],
                              [f.astype(np.float32) for f in bwds], window=args.window)

        # bake per-frame mover label (1=MOVING, 2=POT_DYN) + boxes; throw the flow away
        mover_labels, boxes_per_frame = [], []
        for i in range(len(voted)):
            final = voted[i].copy()
            boxes = []
            for m, box, name, tid in frame_dets[i]:
                dynamic = bool(tracker.is_dynamic(tid))
                final[m] = MOVING if dynamic else POT_DYN
                boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                              name, dynamic])
            ml = np.zeros((H, W), np.uint8)
            ml[final == MOVING] = 1
            ml[final == POT_DYN] = 2
            mover_labels.append(ml)
            boxes_per_frame.append(boxes)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, mover_labels=np.stack(mover_labels),
                                start=args.start, end=end)
            cache.with_suffix(".boxes.json").write_text(json.dumps(boxes_per_frame))
            print(f"[scene] cached dynamic objects -> {cache} (reuse with no flow next time)")

    # --- render: clean RGB + mover overlay + static glyphs ---------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (W, H))
    for i, t in enumerate(tqdm(range(args.start, end - 1), desc="render")):
        ml = mover_labels[i]
        boxes = [(b[:4], b[4], b[5]) for b in boxes_per_frame[i]]

        use_grey = args.bg_grey or (0 < args.bg_grey_until and t < args.bg_grey_until)
        im = np.full((H, W, 3), 128, np.uint8) if use_grey else disp_bgr(t)
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        Rinv = R.T
        # ego-lane green fill (40 m) + thin black borders (50 m), under everything
        im = draw_lane(im, Lb, Rb, Rinv, C, fx, fy, cx, cy, args.los_fill, args.los_border)

        # dynamic movers (red=moving, orange=pot_dyn) -- ONLY inside a mover box
        box_mask = np.zeros((H, W), bool)
        for box, name, dynamic in boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            box_mask[max(0, y0):y1, max(0, x0):x1] = True
        ov = im.copy()
        ov[(ml == 1) & box_mask] = DYN
        ov[(ml == 2) & box_mask] = POTDYN
        mov = (ml > 0) & box_mask
        if mov.any():
            m3 = mov[..., None]
            im = np.where(m3, (im * (1 - args.alpha) + ov * args.alpha).astype(np.uint8), im)
        for box, name, dynamic in boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            col = DYN if dynamic else POTDYN
            cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
            cv2.putText(im, f"{name} {'DYN' if dynamic else 'pot_dyn'}", (x0, max(10, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # static objects reprojected (within line of sight)
        im = draw_static(im, statics, Rinv, C, fx, fy, cx, cy, UP, los=args.los_static)
        writer.write(banner(im, t, N))
    writer.release()
    print(f"[scene] wrote {args.out}")


if __name__ == "__main__":
    main()
