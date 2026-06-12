"""Parallax-separation probe — validates the sky-veto premise.

parallax(x) = | f_obs(x) - f_rot(x) |  ~ rho * |A(x).t|, where f_rot is the
EXACT pure-rotation flow from the relative pose (homography K R_rel K^-1,
depth-free) and f_obs is RAFT flow. A point at infinity (sky) -> parallax ~= 0;
a finite-depth surface rides past the camera -> parallax > 0.

This version tests the premise three ways, none relying on image rows:
  (A) PHYSICS: bin stereo-VALID pixels by rho, show median parallax rises with
      rho (confirms parallax ∝ rho, gives the dynamic range).
  (B) FLICKER SKY: the seg flickers sky->Car, so on a flicker frame the big
      upper "Car" blob IS the sky. Measure parallax inside it + the forward/
      backward flow consistency there (is the low/զ reading trustworthy or is
      RAFT degenerate on textureless sky?).
  (C) REFERENCE: parallax in real near surfaces (road/building) for contrast.

All in the image_0 (gray left) frame — same frame as trajectory + SGBM.
"""

from __future__ import annotations

import argparse
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
FX = 707.0912
BASELINE = 0.537165


def rotation_flow(K, R_rel, W, H):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    x = np.stack([uu.ravel(), vv.ravel(), np.ones(W * H)], axis=0)
    Hmat = K @ R_rel @ np.linalg.inv(K)
    xp = Hmat @ x
    f = np.stack([xp[0] / xp[2] - x[0], xp[1] / xp[2] - x[1]], axis=-1)
    return f.reshape(H, W, 2).astype(np.float32)


def sgbm_validity(L, R):
    block = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=384, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(L, R).astype(np.float32) / 16.0
    valid = disp > 0
    rho = np.where(valid, disp / (FX * BASELINE), 0.0)
    return valid, rho


def fb_inconsistency(f_ab, f_ba):
    """|f_ab(x) + f_ba(x + f_ab(x))| — high where flow is unreliable."""
    H, W = f_ab.shape[:2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    xb = np.clip(uu + f_ab[..., 0], 0, W - 1).astype(np.float32)
    yb = np.clip(vv + f_ab[..., 1], 0, H - 1).astype(np.float32)
    fba_x = cv2.remap(f_ba[..., 0], xb, yb, cv2.INTER_LINEAR)
    fba_y = cv2.remap(f_ba[..., 1], xb, yb, cv2.INTER_LINEAR)
    return np.sqrt((f_ab[..., 0] + fba_x) ** 2 + (f_ab[..., 1] + fba_y) ** 2)


def st(a):
    if a.size == 0:
        return "(empty)"
    p = np.percentile(a, [10, 50, 90])
    return f"n={a.size:7d}  p10={p[0]:6.2f}  med={p[1]:6.2f}  p90={p[2]:6.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=int, nargs="+", default=[99, 100, 101])
    ap.add_argument("--out", type=Path, default=SRC / "parallax_probe")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    left = load_sequence(SEQ_DIR, camera="image_0")
    right = load_sequence(SEQ_DIR, camera="image_1")
    K, W, H = left.K, left.width, left.height
    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")

    perc = SRC / "perception"
    man = CacheManifest.load(perc / MANIFEST_NAME)
    id2 = id2label_from_manifest(man)
    l2i = {v: k for k, v in id2.items()}
    CAR, ROAD, BLD = l2i.get("Car"), l2i.get("Road"), l2i.get("Building")

    for t in args.targets:
        La = cv2.imread(str(left.image_paths[t]), cv2.IMREAD_GRAYSCALE)
        Lb = cv2.imread(str(left.image_paths[t + 1]), cv2.IMREAD_GRAYSCALE)
        Rr = cv2.imread(str(right.image_paths[t]), cv2.IMREAD_GRAYSCALE)
        a, b = cv2.cvtColor(La, cv2.COLOR_GRAY2BGR), cv2.cvtColor(Lb, cv2.COLOR_GRAY2BGR)

        f_obs = compute_flow_pair(a, b, backend="raft_large")
        f_ba = compute_flow_pair(b, a, backend="raft_large")
        fb = fb_inconsistency(f_obs, f_ba)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        f_rot = rotation_flow(K, T_rel[:3, :3], W, H)
        par = np.linalg.norm(f_obs - f_rot, axis=-1)
        tnorm = float(np.linalg.norm(T_rel[:3, 3]))

        valid, rho = sgbm_validity(La, Rr)
        sem = load_frame_entry(perc, t)["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)

        print(f"\n===== frame {t}->{t+1}   |t_rel|={tnorm:.3f} m =====")

        # (A) physics: parallax vs rho among valid pixels
        print("(A) parallax vs rho  [stereo-valid, far->near]")
        v = valid & np.isfinite(par)
        rv, pv = rho[v], par[v]
        qs = np.quantile(rv, [0, .2, .4, .6, .8, 1.0])
        for i in range(5):
            m = (rv >= qs[i]) & (rv < qs[i + 1] + (i == 4))
            zmed = 1.0 / np.median(rv[m]) if m.any() else float("nan")
            print(f"    rho[{qs[i]:.3f},{qs[i+1]:.3f}) ~Z={zmed:6.1f}m : {st(pv[m])}")

        # (B) flicker sky = biggest upper-half Car connected component
        car = (sem == CAR)
        ncc, lbl, stats, cent = cv2.connectedComponentsWithStats(
            car.astype(np.uint8), connectivity=8)
        sky_blob = None
        best = 0
        for k in range(1, ncc):
            area = stats[k, cv2.CC_STAT_AREA]
            cy = cent[k][1]
            if area > best and cy < 0.5 * H and area > 3000:
                best, sky_blob = area, k
        if sky_blob is not None:
            m = lbl == sky_blob
            print(f"(B) FLICKER sky-as-Car blob (area={best}, the mislabeled sky)")
            print(f"    parallax        : {st(par[m])}")
            print(f"    flow fb-inconsist: {st(fb[m])}   (high => flow untrustworthy)")
            print(f"    stereo-valid frac: {valid[m].mean():.1%}")
        else:
            print("(B) no large upper Car blob on this frame")

        # (C) reference near surfaces
        for nm, cid in [("Road", ROAD), ("Building", BLD)]:
            if cid is None:
                continue
            m = (sem == cid) & valid
            if m.sum() > 500:
                print(f"(C) {nm:8s} (valid): {st(par[m])}")

        # panel: image / parallax heat / fb-inconsistency / sky-blob overlay
        pmax = np.percentile(par, 99)
        heat = cv2.applyColorMap(np.clip(par / max(pmax, 1e-6) * 255, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        fbmax = np.percentile(fb, 99)
        fbim = cv2.applyColorMap(np.clip(fb / max(fbmax, 1e-6) * 255, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_MAGMA)
        ov = a.copy()
        if sky_blob is not None:
            ov[lbl == sky_blob] = (0, 0, 255)
        panel = np.vstack([a, heat, fbim, ov])
        cv2.putText(panel, f"{t}->{t+1} | img / parallax(0..{pmax:.0f}) / fb-inconsist / sky-Car blob",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.out / f"probe_{t:03d}.png"), panel)
    print(f"\npanels -> {args.out}/probe_*.png")


if __name__ == "__main__":
    main()
