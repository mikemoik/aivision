"""Render the calibrated dynamic-VETO overlay (Layer-1 dynamic attribute).

New pass vs the old dynamic_tracking.mp4:
  - STEREO rho stream (mono out of the motion path) -> sky is rho0/invalid.
  - calibrated combination (dynamic_probe.py): per movable region,
        epi  > 1.0                                  (#1 everywhere)
      | in_cone(15) & fb_ok & B>2 & d  > 0.5         (#2 deficit, corroborator)
      | in_cone(15) & fb_ok &        loom > 0.6      (#3 looming, primary @FOE)
  - fwd/back OCCLUSION gate (fb_ok) — suppresses the near off-axis parked-car
    false-fire (an occlusion artefact in d/loom).
  - region-median decision + DynamicTracker ±N persistence (red=confirmed).

Everything in the image_0 frame (pose + stereo rho + flow all live there).
"""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.flow import compute_flow_pair
from aivision.kitti import load_sequence
from aivision.pocs import common
from aivision.perception import (
    CacheManifest, MANIFEST_NAME, id2label_from_manifest, load_frame_entry,
)
from aivision.dynamic import (
    Region, DynamicTracker, warp_mask_forward, skew, sampson_distance,
    MOVABLE_CLASS_NAMES, _bbox,
)

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")
RHO_STEREO = SRC / "inv_depth_fused_rho_stereo_16bit.mkv"

# calibrated thresholds (dynamic_probe.py)
# near-field error amplification: a static car passing close hits epi~1.3 then
# ~7 in its last 2 frames (huge flow x small pose error). Real lateral movers are
# 3-6 sustained. tau=2.0 + persistence (min_votes=2) makes the static pass-by a
# <=1-frame amber blip while sustained movers still confirm red.
TAU_EPI = 2.0
CONE_DEG = 15.0
B_MIN = 2.0
B_MAX = 8.0          # above this, object is near+lateral -> parallax-scaled
                     # depth error makes loom/d unreliable (parked-car FP);
                     # epipolar decides there. leader B~3 (in), parked B~14 (out).
TAU_D = 0.5
TAU_LOOM = 0.6
RHODOT_FLOOR = 5e-4
TAU_FB = 1.5          # px fwd/back inconsistency -> occlusion gate
FLOW_MAX = 80.0       # px — above this, near-field error amplification swamps
                      # epipolar (a static car passed at 2 m gives 100-256 px
                      # flow, epi 7; the leader gives ~5 px). caps #1 only; the
                      # leader rides #3 looming in the far band, untouched.
MIN_CORE = 150


def rho_stream(video: Path, W: int, H: int):
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


def fb_inconsistency(f_ab, f_ba):
    H, W = f_ab.shape[:2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    xb = np.clip(uu + f_ab[..., 0], 0, W - 1).astype(np.float32)
    yb = np.clip(vv + f_ab[..., 1], 0, H - 1).astype(np.float32)
    bx = cv2.remap(f_ba[..., 0], xb, yb, cv2.INTER_LINEAR)
    by = cv2.remap(f_ba[..., 1], xb, yb, cv2.INTER_LINEAR)
    return np.sqrt((f_ab[..., 0] + bx) ** 2 + (f_ab[..., 1] + by) ** 2)


def compute_geometry(a, b, rho, rho_next, K, ray, uu, vv, src_xy, T_rel, backend):
    H, W = a.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]

    f_obs = compute_flow_pair(a, b, backend=backend)
    f_ba = compute_flow_pair(b, a, backend=backend)
    fb = fb_inconsistency(f_obs, f_ba)
    flow_mag = np.linalg.norm(f_obs, axis=-1)

    # #1 epipolar
    F = Kinv.T @ skew(t_rel) @ R_rel @ Kinv
    epi = sampson_distance(F, src_xy, src_xy + f_obs.reshape(-1, 2)).reshape(H, W)

    # rigid field at rho and 0 -> deviation, budget B, ratio d
    valid = rho > 0
    Rr = ray @ R_rel.T
    P = Rr + rho[..., None] * t_rel

    def proj(Pm):
        z = np.maximum(Pm[..., 2], 1e-6)
        return np.stack([fx * Pm[..., 0] / z + cx, fy * Pm[..., 1] / z + cy], -1)

    grid = np.stack([uu, vv], -1)
    f_static = proj(P) - grid
    f_rot = proj(Rr) - grid
    deviation = np.linalg.norm(f_obs - f_static, axis=-1)
    B = np.linalg.norm(f_static - f_rot, axis=-1)
    d = deviation / np.maximum(B, B_MIN)

    # #3 looming
    Pz = np.maximum(P[..., 2], 1e-6)
    rhodot_static = rho / Pz - rho
    wx = np.clip(uu + f_obs[..., 0], 0, W - 1).astype(np.float32)
    wy = np.clip(vv + f_obs[..., 1], 0, H - 1).astype(np.float32)
    rho_obs_next = cv2.remap(rho_next, wx, wy, cv2.INTER_LINEAR)
    rhodot_obs = rho_obs_next - rho
    loom_valid = valid & (rho_obs_next > 0) & (rhodot_static > RHODOT_FLOOR)
    loom = np.where(loom_valid,
                    (rhodot_static - rhodot_obs) / np.maximum(rhodot_static, RHODOT_FLOOR), 0.0)

    # cone around epipole dir
    epi_dir = -R_rel.T @ t_rel
    n = np.linalg.norm(epi_dir)
    epi_dir = epi_dir / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    rn = ray / (np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-9)
    cone = (rn @ epi_dir) >= np.cos(np.deg2rad(CONE_DEG))
    fb_ok = fb < TAU_FB
    return dict(epi=epi, d=d, B=B, loom=loom, loom_valid=loom_valid,
                cone=cone, fb_ok=fb_ok, valid=valid, flow=f_obs, flow_mag=flow_mag)


def detect(sem, g, movable_ids):
    H, W = sem.shape
    movable = np.isin(sem, list(movable_ids))
    if not movable.any():
        return []
    lateral = (g["epi"] > TAU_EPI) & (g["flow_mag"] < FLOW_MAX)
    # FOE base gate: inside the cone, occlusion-clean, and NOT near-field
    # (B<B_MAX, else parallax-scaled depth error dominates). NO lower-B bound
    # here: looming (drho/dt) is exactly the cue that works at SMALL B (far /
    # at the FOE) — gating it by B>B_MIN throws away the far leaders.
    foe = g["cone"] & g["fb_ok"] & (g["B"] < B_MAX)
    looming = foe & g["loom_valid"] & (g["loom"] > TAU_LOOM)        # #3, any small B
    deficit = foe & (g["B"] > B_MIN) & (g["d"] > TAU_D)             # #2, needs budget
    cand = movable & (lateral | deficit | looming)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)

    regions = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_CORE:
            continue
        core = lbl == k
        cls = int(np.bincount(sem[core]).argmax())
        # grow core onto its movable instance FIRST, capped 4x (avoid sky flood)
        full = movable & (sem == cls)
        flbl = cv2.connectedComponentsWithStats(full.astype(np.uint8), 8)[1]
        touch = np.unique(flbl[core]); touch = touch[touch > 0]
        grown = np.isin(flbl, touch) if touch.size else core
        mask = grown if int(grown.sum()) <= core.sum() * 4 else \
            cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        # region-median vote over the WHOLE OBJECT mask, not the thresholded core
        # (the core is biased high by construction). A static car whose body
        # median epi is 1.3 stays below tau even if a sliver spikes.
        em = float(np.median(g["epi"][mask]))
        fmag = float(np.median(g["flow_mag"][mask]))
        foe_m = mask & g["cone"] & g["fb_ok"] & (g["B"] < B_MAX)
        lm = foe_m & g["loom_valid"]                       # #3: any small B
        dm = foe_m & (g["B"] > B_MIN)                      # #2: needs budget
        loom_med = float(np.median(g["loom"][lm])) if lm.sum() > 20 else 0.0
        d_med = float(np.median(g["d"][dm])) if dm.sum() > 20 else 0.0
        ok = (em > TAU_EPI and fmag < FLOW_MAX) or (loom_med > TAU_LOOM) or (d_med > TAU_D)
        if not ok:
            continue
        x0, y0, x1, y1 = _bbox(mask)
        ys, xs = np.where(mask)
        regions.append(Region(
            cls=cls, mask=mask, area=int(mask.sum()), bbox=(x0, y0, x1, y1),
            centroid=(float(xs.mean()), float(ys.mean())),
            epi_med=em, rep_med=loom_med, in_cone=bool(foe_m.sum() > 0.3 * core.sum()),
            is_dynamic=True))
    return regions


def draw(bgr, tracks, id2label, t, N):
    out = bgr.copy().astype(np.float32)
    red, amber = np.array([0, 0, 255.]), np.array([0, 170, 255.])
    n_conf = 0
    for tr in tracks:
        if tr.missed > 0:
            continue
        conf = tr.confirmed
        tent = (not conf) and len(tr.votes) and tr.votes[-1]
        if not (conf or tent):
            continue
        col = red if conf else amber
        a = tr.mask[..., None].astype(np.float32) * 0.45
        out = out * (1 - a) + col * a
        n_conf += int(conf)
    out = out.astype(np.uint8)
    for tr in tracks:
        if tr.missed > 0:
            continue
        conf = tr.confirmed
        tent = (not conf) and len(tr.votes) and tr.votes[-1]
        if not (conf or tent):
            continue
        x0, y0, x1, y1 = _bbox(tr.mask)
        col = (0, 0, 255) if conf else (0, 170, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        name = id2label.get(tr.cls, f"id{tr.cls}")
        cue = "epi" if tr.epi_med > TAU_EPI else "loom"
        val = tr.epi_med if cue == "epi" else tr.rep_med
        cv2.putText(out, f"#{tr.tid} {name} {'DYN' if conf else '?'} {cue}{val:.1f} {sum(tr.votes)}/{len(tr.votes)}",
                    (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    lines = [f"frame {t:03d}/{N-1}  pair {t}->{t+1}   [stereo-rho veto: epi>1.0 | cone15 loom>0.6 | d>0.5 + fb-gate]",
             f"confirmed dynamic={n_conf}   red=confirmed amber=tentative"]
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (640, 8 + 22 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, dst=out)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SRC / "dynamic_veto.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--backend", default="raft_large")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    left = load_sequence(SEQ_DIR, camera="image_0")
    K, W, H, N = left.K, left.width, left.height, left.n_frames
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")

    perc = SRC / "perception"
    man = CacheManifest.load(perc / MANIFEST_NAME)
    id2label = id2label_from_manifest(man)
    l2i = {v: k for k, v in id2label.items()}
    movable_ids = {l2i[n] for n in MOVABLE_CLASS_NAMES if n in l2i}

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).astype(np.float64)
    src_xy = np.stack([uu.ravel(), vv.ravel()], -1).astype(np.float64)
    uu = uu.astype(np.float64); vv = vv.astype(np.float64)

    end = N if args.end < 0 else min(args.end, N)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (W, H))
    tracker = DynamicTracker(vote_window=5, min_votes=2)

    rit = rho_stream(RHO_STEREO, W, H)
    rho_a = next(rit)
    for _ in range(args.start):
        rho_a = next(rit, rho_a)
    a = cv2.cvtColor(cv2.imread(str(left.image_paths[args.start]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    for t in tqdm(range(args.start, end - 1), desc="dynamic-veto"):
        b = cv2.cvtColor(cv2.imread(str(left.image_paths[t + 1]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        rho_b = next(rit, rho_a)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        g = compute_geometry(a, b, rho_a, rho_b, K, ray, uu, vv, src_xy, T_rel, args.backend)

        sem = load_frame_entry(perc, t)["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        regions = detect(sem, g, movable_ids)
        tracks = tracker.update(regions)
        writer.write(draw(a, tracks, id2label, t, N))
        tracker.propagate(g["flow"])
        a, rho_a = b, rho_b

    writer.release()
    print(f"[dynamic-veto] wrote {args.out}")


if __name__ == "__main__":
    main()
