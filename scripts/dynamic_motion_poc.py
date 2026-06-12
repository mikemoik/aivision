"""POC: geometry-based dynamic-region detection (no reliance on the flickery
per-frame semantic label).

Given, per consecutive frame pair (t, t+1):
  - observed dense optical flow  f_obs        (SEA-RAFT / raft_large)
  - fused inverse depth ρ_t  → metric depth Z = 1/ρ
  - relative camera pose from the ORB-SLAM3 trajectory (metric scale)

we compute two independent "is this pixel moving on its own?" residuals:

  1. EPIPOLAR (Sampson) residual — depth-FREE. A static point must move along
     its epipolar line (essential matrix from the relative pose), whatever its
     depth. So the bad-depth sky (static) reads ~0 here even though its fused
     depth is a wrong ~55 m. Blind only to motion *along* the epipole.

  2. REPROJECTION residual — depth-BASED. Reproject P_t = Z·K⁻¹x via the
     relative pose to get the predicted "rigid" pixel; |f_obs − f_induced| is
     the independent motion. Covers the forward-degenerate case the epipolar
     test misses — but is contaminated wherever the depth is wrong (e.g. sky),
     which is exactly why we DON'T use it alone.

The pair of panels makes the design point visible: epipolar cleanly separates
the genuinely-moving car from the static-but-wrong-depth sky; reprojection
lights the sky up as a false positive. The detector to ship pools the EPIPOLAR
residual over segmentation regions (label-agnostic) and falls back to the
depth residual only inside the forward-epipole cone.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from aivision.kitti import load_sequence
from aivision.perception import (
    CacheManifest,
    MANIFEST_NAME,
    id2label_from_manifest,
    load_frame_entry,
)
from aivision.flow import compute_flow_pair

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")


def skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def decode_rho_frames(video: Path, W: int, H: int, want: set[int]) -> dict[int, np.ndarray]:
    """Sequentially decode the gray16 rho video, keep wanted frame indices."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE,
    )
    fb = W * H * 2
    out: dict[int, np.ndarray] = {}
    i = 0
    hi = max(want)
    while True:
        raw = proc.stdout.read(fb)
        if len(raw) < fb:
            break
        if i in want:
            out[i] = np.frombuffer(raw, dtype=np.uint16).reshape(H, W).copy()
        if i >= hi:
            break
        i += 1
    proc.stdout.close()
    proc.wait()
    return out


def sampson_distance(F: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """First-order epipolar (Sampson) distance in pixels. src/dst: (N,2)."""
    s = np.hstack([src, np.ones((src.shape[0], 1))])
    d = np.hstack([dst, np.ones((dst.shape[0], 1))])
    Fs = (F @ s.T).T            # (N,3)
    Ftd = (F.T @ d.T).T         # (N,3)
    num = np.einsum("ij,ij->i", d, Fs) ** 2
    den = Fs[:, 0] ** 2 + Fs[:, 1] ** 2 + Ftd[:, 0] ** 2 + Ftd[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def heat(x: np.ndarray, valid: np.ndarray, vmax: float, label: str) -> np.ndarray:
    norm = np.clip(x / vmax, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~valid] = (40, 40, 40)
    cv2.putText(bgr, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, nargs="+", default=[50, 99, 100, 103, 142])
    ap.add_argument("--video", type=Path, default=SRC / "inv_depth_fused_mono_rho_gf_16bit.mkv")
    ap.add_argument("--perception", type=Path, default=SRC / "perception")
    ap.add_argument("--out-dir", type=Path, default=SRC / "dynamic_poc")
    ap.add_argument("--backend", default="raft_large")
    ap.add_argument("--epi-vmax", type=float, default=3.0, help="epipolar heat clip (px)")
    ap.add_argument("--rep-vmax", type=float, default=12.0, help="reproj heat clip (px)")
    ap.add_argument("--dyn-thresh", type=float, default=1.5, help="epipolar px to flag dynamic")
    args = ap.parse_args()

    seq = load_sequence(SEQ_DIR, camera="image_2")
    W, H, K = seq.width, seq.height, seq.K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    poses = __import__("aivision.pocs.common", fromlist=["load_kitti_trajectory"]).load_kitti_trajectory(
        SRC / "CameraTrajectory.txt")

    manifest = CacheManifest.load(args.perception / MANIFEST_NAME)
    id2label = id2label_from_manifest(manifest)
    l2i = {v: k for k, v in id2label.items()}
    veh_ids = [l2i[n] for n in ("Car", "Truck", "Bus", "Other Vehicle") if n in l2i]

    want = set(args.frames) | {f + 1 for f in args.frames}
    rho_frames = decode_rho_frames(args.video, W, H, want)

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)  # H,W,3
    src_xy = np.stack([uu.ravel(), vv.ravel()], axis=-1).astype(np.float64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'frame':>5} | {'sky-band epi':>12} {'sky-band rep':>12} | "
          f"{'vehicle epi':>11} {'vehicle rep':>11} {'veh_px':>7}")
    print("-" * 76)

    for t in args.frames:
        bgr_a = cv2.imread(str(seq.image_paths[t]), cv2.IMREAD_COLOR)
        bgr_b = cv2.imread(str(seq.image_paths[t + 1]), cv2.IMREAD_COLOR)
        flow = compute_flow_pair(bgr_a, bgr_b, backend=args.backend)  # H,W,2

        # relative pose cam_t -> cam_{t+1}:  T_rel = inv(T_wc_{t+1}) @ T_wc_t
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]

        # --- epipolar (depth-free) ---
        E = skew(t_rel) @ R_rel
        F = Kinv.T @ E @ Kinv
        dst_xy = src_xy + flow.reshape(-1, 2)
        epi = sampson_distance(F, src_xy, dst_xy).reshape(H, W)

        # --- reprojection (depth-based) ---
        pix = rho_frames[t]
        rho = (pix.astype(np.float32) - 1.0) / 65534.0
        valid = pix > 0
        Z = np.where(valid, 1.0 / np.maximum(rho, 1e-6), 0.0)
        P_t = ray * Z[..., None]                                  # H,W,3
        P_j = P_t @ R_rel.T + t_rel                              # H,W,3
        Zj = np.maximum(P_j[..., 2], 1e-6)
        u_j = fx * P_j[..., 0] / Zj + cx
        v_j = fy * P_j[..., 1] / Zj + cy
        induced = np.stack([u_j - uu, v_j - vv], axis=-1)
        rep = np.linalg.norm(flow - induced, axis=-1)
        rep_valid = valid & (P_j[..., 2] > 0)

        # --- region stats ---
        sem = load_frame_entry(args.perception, t)["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        skyband = np.zeros((H, W), bool); skyband[:60, :] = True
        veh = np.isin(sem, veh_ids) & (vv > 90)  # vehicles below horizon-ish

        def med(arr, m):
            return float(np.median(arr[m])) if m.any() else float("nan")

        sb_epi = med(epi, skyband)
        sb_rep = med(rep, skyband & rep_valid)
        v_epi = med(epi, veh)
        v_rep = med(rep, veh & rep_valid)
        print(f"{t:5d} | {sb_epi:12.2f} {sb_rep:12.2f} | "
              f"{v_epi:11.2f} {v_rep:11.2f} {int(veh.sum()):7d}")

        # --- panel ---
        src_panel = bgr_a.copy()
        cv2.putText(src_panel, f"frame {t}->{t+1}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        flow_mag = np.linalg.norm(flow, axis=-1)
        flow_panel = heat(flow_mag, np.ones((H, W), bool),
                          float(np.percentile(flow_mag, 98)) or 1.0, "observed flow |f|")
        epi_panel = heat(epi, np.ones((H, W), bool), args.epi_vmax,
                         f"EPIPOLAR residual (depth-free, clip {args.epi_vmax}px)")
        rep_panel = heat(rep, rep_valid, args.rep_vmax,
                         f"REPROJ residual (depth-based, clip {args.rep_vmax}px)")
        dyn = (epi > args.dyn_thresh)
        dyn_panel = bgr_a.copy().astype(np.float32)
        red = np.zeros_like(dyn_panel); red[..., 2] = 255
        a = (dyn[..., None] * 0.5)
        dyn_panel = (dyn_panel * (1 - a) + red * a).astype(np.uint8)
        cv2.putText(dyn_panel, f"dynamic = epipolar > {args.dyn_thresh}px",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        panel = np.vstack([src_panel, flow_panel, epi_panel, rep_panel, dyn_panel])
        out = args.out_dir / f"pair_{t:06d}.png"
        cv2.imwrite(str(out), panel)

    print(f"\n[dynamic-poc] panels in {args.out_dir}")


if __name__ == "__main__":
    main()
