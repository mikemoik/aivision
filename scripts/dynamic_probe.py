"""Dynamic-veto calibration probe.

Measures the three dynamic residuals on known reference objects so the four
thresholds {tau_epi, B_min, tau_d, tau_loom} are read off real separation
rather than guessed, and so we can prove the FOE machinery (#2 deficit, #3
looming) catches a leading car where epipolar (#1) is blind.

Residuals (all in the image_0 frame, on the STEREO rho stream, mono out):
  #1 epi   = sampson(F, x, x+f_obs)                          depth-free
  #2 d     = |f_obs - f_static(rho)| / max(B, B_floor),
             B = |f_static(rho) - f_rot|  (predicted parallax / budget)
  #3 loom  = (rhodot_static - rhodot_obs) / max(rhodot_static, floor)
     f_static(x;rho) = pi(K (R_rel r + rho t_rel)) - x,  r = K^-1 [u,v,1]
     static new rho': rho' = rho / P_z,  P = R_rel r + rho t_rel

Reference objects (auto-found, classified by position):
  - parked car   : real Car (Car-class & stereo-valid) on the LEFT, lower half
  - leading car  : real Car near image centre (the FOE region)
  - static bg    : Road/Building (valid) baseline
  - sky-Car blob : the mislabeled sky (epi only; depth residuals abstain)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from aivision.flow import compute_flow_pair
from aivision.kitti import load_sequence
from aivision.pocs import common
from aivision.perception import (
    CacheManifest, MANIFEST_NAME, id2label_from_manifest, load_frame_entry,
)

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")
RHO_STEREO = SRC / "inv_depth_fused_rho_stereo_16bit.mkv"
B_FLOOR = 1.0      # px — parallax-budget floor below which #2 abstains
RHODOT_FLOOR = 5e-4


def read_rho_frames(video: Path, W: int, H: int, idxs: set[int]) -> dict[int, np.ndarray]:
    """Decode just the requested gray16 rho frames -> {idx: rho float32 (0=invalid)}."""
    want = max(idxs) + 1
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE)
    fb = W * H * 2
    out: dict[int, np.ndarray] = {}
    try:
        for i in range(want):
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            if i in idxs:
                pix = np.frombuffer(raw, np.uint16).reshape(H, W).astype(np.float32)
                rho = np.where(pix > 0, (pix - 1.0) / 65534.0, 0.0)
                out[i] = rho
    finally:
        proc.stdout.close()
        proc.wait()
    return out


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson(F, src, dst):
    s = np.hstack([src, np.ones((src.shape[0], 1))])
    d = np.hstack([dst, np.ones((dst.shape[0], 1))])
    Fs = (F @ s.T).T
    Ftd = (F.T @ d.T).T
    num = np.einsum("ij,ij->i", d, Fs) ** 2
    den = Fs[:, 0] ** 2 + Fs[:, 1] ** 2 + Ftd[:, 0] ** 2 + Ftd[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def st(a, q=(10, 50, 90)):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "(empty)"
    p = np.percentile(a, q)
    return f"n={a.size:6d}  p{q[0]}={p[0]:6.2f}  med={p[1]:6.2f}  p{q[2]}={p[2]:6.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=int, nargs="+", default=[99, 100, 101])
    ap.add_argument("--out", type=Path, default=SRC / "dynamic_probe")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    left = load_sequence(SEQ_DIR, camera="image_0")
    K, W, H = left.K, left.width, left.height
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")

    perc = SRC / "perception"
    man = CacheManifest.load(perc / MANIFEST_NAME)
    l2i = {v: k for k, v in id2label_from_manifest(man).items()}
    CAR, ROAD, BLD = l2i.get("Car"), l2i.get("Road"), l2i.get("Building")

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).astype(np.float64)
    src_xy = np.stack([uu.ravel(), vv.ravel()], -1).astype(np.float64)

    need = set()
    for t in args.targets:
        need |= {t, t + 1}
    rho_frames = read_rho_frames(RHO_STEREO, W, H, need)

    for t in args.targets:
        if t not in rho_frames or (t + 1) not in rho_frames:
            print(f"frame {t}: missing rho"); continue
        La = cv2.imread(str(left.image_paths[t]), cv2.IMREAD_GRAYSCALE)
        Lb = cv2.imread(str(left.image_paths[t + 1]), cv2.IMREAD_GRAYSCALE)
        a, b = cv2.cvtColor(La, cv2.COLOR_GRAY2BGR), cv2.cvtColor(Lb, cv2.COLOR_GRAY2BGR)
        f_obs = compute_flow_pair(a, b, backend="raft_large")

        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]
        tnorm = float(np.linalg.norm(t_rel))

        # #1 epipolar
        F = Kinv.T @ skew(t_rel) @ R_rel @ Kinv
        epi = sampson(F, src_xy, src_xy + f_obs.reshape(-1, 2)).reshape(H, W)

        # rigid field at rho and at 0 -> deviation, budget B, ratio d
        rho = rho_frames[t]
        valid = rho > 0
        Rr = ray @ R_rel.T                                   # H,W,3
        P = Rr + rho[..., None] * t_rel
        P0 = Rr
        def proj(Pm):
            z = np.maximum(Pm[..., 2], 1e-6)
            return np.stack([fx * Pm[..., 0] / z + cx, fy * Pm[..., 1] / z + cy], -1)
        xy_stat = proj(P)
        xy_rot = proj(P0)
        f_static = xy_stat - np.stack([uu, vv], -1)
        f_rot = xy_rot - np.stack([uu, vv], -1)
        deviation = np.linalg.norm(f_obs - f_static, axis=-1)
        B = np.linalg.norm(f_static - f_rot, axis=-1)
        d = deviation / np.maximum(B, B_FLOOR)

        # #3 looming: rhodot_static vs rhodot_obs (warp rho_{t+1} back via flow)
        Pz = np.maximum(P[..., 2], 1e-6)
        rho_pred = rho / Pz                                  # static new rho
        rhodot_static = rho_pred - rho
        rho_next = rho_frames[t + 1]
        wx = np.clip(uu + f_obs[..., 0], 0, W - 1).astype(np.float32)
        wy = np.clip(vv + f_obs[..., 1], 0, H - 1).astype(np.float32)
        rho_obs_next = cv2.remap(rho_next, wx, wy, cv2.INTER_LINEAR)
        rhodot_obs = rho_obs_next - rho
        loom_valid = valid & (rho_obs_next > 0) & (rhodot_static > RHODOT_FLOOR)
        loom = np.where(loom_valid,
                        (rhodot_static - rhodot_obs) / np.maximum(rhodot_static, RHODOT_FLOOR),
                        np.nan)

        # reference regions
        sem = load_frame_entry(perc, t)["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        car_real = (sem == CAR) & valid
        ncc, lbl, stats, cent = cv2.connectedComponentsWithStats(
            car_real.astype(np.uint8), connectivity=8)
        cars = [(int(stats[k, cv2.CC_STAT_AREA]), float(cent[k][0]), float(cent[k][1]), k)
                for k in range(1, ncc) if stats[k, cv2.CC_STAT_AREA] > 600]
        cars.sort(reverse=True)

        def classify(area, cx_, cy_):
            if cx_ < 0.40 * W:
                return "parked-LEFT"
            if abs(cx_ - 0.5 * W) < 0.18 * W and cy_ < 0.62 * H:
                return "leading-CENTRE"
            return "car-other"

        print(f"\n===== frame {t}->{t+1}   |t_rel|={tnorm:.3f} m =====")
        print(f"{'region':16s}  {'epi #1':>22s} | {'d #2':>22s} | {'loom #3':>22s}  [B med]")
        refs = []
        for area, cx_, cy_, k in cars:
            m = lbl == k
            tag = classify(area, cx_, cy_)
            refs.append((tag, m, area, cx_, cy_))
        # static baselines
        for nm, cid in [("static-Road", ROAD), ("static-Bldg", BLD)]:
            if cid is not None:
                refs.append((nm, (sem == cid) & valid, int(((sem == cid) & valid).sum()), 0, 0))
        # sky-Car blob (mostly stereo-invalid upper Car)
        sky = (sem == CAR) & (~valid)
        ncc2, lbl2, stats2, cent2 = cv2.connectedComponentsWithStats(sky.astype(np.uint8), 8)
        for k in range(1, ncc2):
            if stats2[k, cv2.CC_STAT_AREA] > 3000 and cent2[k][1] < 0.5 * H:
                refs.append(("sky-Car-blob", lbl2 == k, int(stats2[k, cv2.CC_STAT_AREA]), 0, 0))
                break

        for tag, m, area, cx_, cy_ in refs:
            md = m & (B > B_FLOOR)
            ml = m & loom_valid
            es = f"med={np.median(epi[m]):5.2f} p90={np.percentile(epi[m],90):5.2f}"
            ds = (f"med={np.median(d[md]):5.2f} p90={np.percentile(d[md],90):5.2f}"
                  if md.sum() > 30 else "  (no budget)")
            ls = (f"med={np.nanmedian(loom[ml]):5.2f} p90={np.nanpercentile(loom[ml],90):5.2f}"
                  if ml.sum() > 30 else "  (no loom)")
            bmed = np.median(B[m]) if m.any() else 0
            loc = f"@({cx_:.0f},{cy_:.0f})" if cx_ else ""
            print(f"{tag:16s}  {es:>22s} | {ds:>22s} | {ls:>22s}  [B={bmed:4.1f}] a={area} {loc}")

        # visualization: image / epi / d / loom, ref boxes drawn
        def heat(x, hi, cmap):
            return cv2.applyColorMap(np.clip(np.nan_to_num(x) / hi * 255, 0, 255).astype(np.uint8), cmap)
        epih = heat(epi, 6.0, cv2.COLORMAP_TURBO)
        dh = heat(d, 1.5, cv2.COLORMAP_TURBO)
        loomh = heat(np.nan_to_num(loom, nan=0.0), 1.5, cv2.COLORMAP_TURBO)
        canvas = a.copy()
        for tag, m, area, cx_, cy_ in refs:
            if "car" in tag.lower() or "lead" in tag.lower() or "park" in tag.lower() or "sky" in tag.lower():
                ys, xs = np.where(m)
                if xs.size:
                    cv2.rectangle(canvas, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 1)
                    cv2.putText(canvas, tag.split('-')[0], (xs.min(), max(0, ys.min() - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        panel = np.vstack([canvas, epih, dh, loomh])
        cv2.putText(panel, f"{t}->{t+1}  img+refs / epi(0..6) / d(0..1.5) / loom(0..1.5)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.out / f"dyn_{t:03d}.png"), panel)

    print(f"\npanels -> {args.out}/dyn_*.png")


if __name__ == "__main__":
    main()
