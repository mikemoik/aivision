"""Build the fused inverse-depth (rho) video for the ORB-SLAM3 dense POC.

Pipeline: SGBM stereo -> guided-filter smoothing (Phase 1A) -> multi-frame
rho-space median fusion (+/-WINDOW, pose-warped) -> Depth Pro mono fill via
per-frame affine fit -> 16-bit FFV1 gray16le MKV.

Encoding: pixel 0 = invalid, pixel in [1, 65535] -> rho = (pixel - 1) / 65534.

Recovered from session 68870bb1 (2026-06-12) where it ran as an inline
command; Phase 1A adds the guided filter on each SGBM rho map before it
enters the fusion buffer.
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from aivision.kitti import load_sequence
from aivision.pocs import common
from aivision.perception import mono_depth_frame, _get_depth_model

SEQ_DIR = Path("data/kitti/04")
SRC = Path("outputs/pocs/orb_slam3")

FX = 707.0912
BASELINE = 0.537165
WINDOW = 5
MIN_VALID = 3

# Phase 1A guided filter: guide is the left gray image scaled to [0, 1].
# r=4/eps=1e-2 chosen by sweep on frame 0: best local-sigma reduction with the
# least gradient inflation at validity borders (larger radii halo there).
GF_RADIUS = 4
GF_EPS = 1e-2
GF_MIN_SUPPORT = 0.3  # min filtered validity weight to trust a smoothed pixel


def encode_16(rho: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rho_clipped = np.clip(rho, 0.0, 1.0)
    pix = np.zeros(rho.shape, dtype=np.uint16)
    pix[valid_mask] = np.round(1.0 + rho_clipped[valid_mask] * 65534.0).astype(np.uint16)
    return pix


def guided_smooth_rho(rho: np.ndarray, guide_gray: np.ndarray) -> np.ndarray:
    """Edge-aware smoothing of a sparse rho map via normalised convolution.

    Filters rho*valid and valid separately with the same guided filter so
    invalid (zero) pixels carry no weight, then re-masks to the original
    validity — smoothing only, no inpainting.
    """
    valid = np.isfinite(rho) & (rho > 0)
    guide = guide_gray.astype(np.float32) / 255.0
    v = valid.astype(np.float32)
    src = np.where(valid, rho, 0.0).astype(np.float32)
    num = cv2.ximgproc.guidedFilter(guide, src, GF_RADIUS, GF_EPS)
    den = cv2.ximgproc.guidedFilter(guide, v, GF_RADIUS, GF_EPS)
    out = np.full(rho.shape, np.nan, dtype=np.float32)
    ok = valid & (den > GF_MIN_SUPPORT)
    out[ok] = num[ok] / den[ok]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SRC / "inv_depth_fused_mono_rho_16bit.mkv")
    ap.add_argument("--out-stereo", type=Path, default=None,
                    help="sibling stereo-ONLY rho stream (no mono-fill). "
                         "Default: <out stem>_stereo_16bit.mkv")
    ap.add_argument("--no-guided", action="store_true", help="skip Phase 1A guided filter")
    args = ap.parse_args()

    out_path: Path = args.out
    out_stereo: Path = args.out_stereo or out_path.with_name(
        out_path.stem.replace("_16bit", "").replace("_mono", "") + "_stereo_16bit.mkv")
    tmp = SRC / "_invd_rho_frames"
    tmp_stereo = SRC / "_invd_rho_stereo_frames"
    bf = FX * BASELINE

    left_seq = load_sequence(SEQ_DIR, camera="image_0")
    right_seq = load_sequence(SEQ_DIR, camera="image_1")
    W, H = left_seq.width, left_seq.height
    N = left_seq.n_frames
    cx, cy = left_seq.K[0, 2], left_seq.K[1, 2]

    poses = common.load_kitti_trajectory(SRC / "CameraTrajectory.txt")
    R_wc_all = np.transpose(poses[:, :3, :3], (0, 2, 1))
    t_wc_all = -np.einsum("nij,nj->ni", R_wc_all, poses[:, :3, 3])

    # --- 1. SGBM, stored as rho = disp / bf (never converted to Z) ----------
    block = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=384, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    rho_all = np.full((N, H, W), np.nan, dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        L = cv2.imread(str(left_seq.image_paths[i]), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(right_seq.image_paths[i]), cv2.IMREAD_GRAYSCALE)
        disp = sgbm.compute(L, R).astype(np.float32) / 16.0
        rho = np.where(disp > 0, disp / bf, np.nan)  # rho [1/m]
        if not args.no_guided:
            rho = guided_smooth_rho(rho, L)
        rho_all[i] = rho
    label = "sgbm" if args.no_guided else "sgbm+gf"
    print(f"[{label}] {N} frames in {time.time()-t0:.1f}s, "
          f"median rho={np.nanmedian(rho_all):.4f} 1/m")

    # --- 2. Mono Depth Pro, converted once to rho_mono = 1/Z ----------------
    _get_depth_model()
    mono_rho_all = np.zeros((N, H, W), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        bgr = cv2.imread(str(left_seq.image_paths[i]), cv2.IMREAD_COLOR)
        d, _ = mono_depth_frame(bgr)
        d = d.astype(np.float32)
        mono_rho_all[i] = np.where(d > 0, 1.0 / d, 0.0)
    print(f"[mono] {N} frames in {time.time()-t0:.1f}s")

    # --- 3. Per-frame multi-frame fusion done entirely in rho-space ---------
    # Identity: for a pixel of frame j with rho_j, after the relative
    # transform (R_ij = R_wc_i @ R_cw_j, t_ij = R_wc_i @ t_cw_j + t_wc_i):
    #
    #     rho_j * P_i = R_ij @ ray_j + rho_j * t_ij      (bounded — no 1/Z)
    #
    # so u_i = fx*(rho_j*P_i)[0]/(rho_j*P_i)[2] + cx and
    # rho_i = rho_j / (rho_j*P_i)[2]. Z is never materialised.
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rx = (uu - cx) / FX
    ry = (vv - cy) / FX  # fy = fx

    for d in (tmp, tmp_stereo):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    cover_valid = 0.0
    calib_n = 0
    a_sum = b_sum = 0.0
    t0 = time.time()
    for i in range(N):
        js = [j for j in range(i - WINDOW, i + WINDOW + 1) if 0 <= j < N]
        K = len(js)
        stack = np.full((K, H, W), np.nan, dtype=np.float32)
        own_idx = js.index(i)
        stack[own_idx] = rho_all[i]
        R_wc_i, t_wc_i = R_wc_all[i], t_wc_all[i]
        for k, j in enumerate(js):
            if j == i:
                continue
            rho_j_all = rho_all[j]
            valid_j = np.isfinite(rho_j_all) & (rho_j_all > 0)
            if not valid_j.any():
                continue
            rho_j = rho_j_all[valid_j]                              # (M,)
            ray_j = np.stack([rx[valid_j], ry[valid_j],
                              np.ones_like(rx[valid_j])], axis=0)   # (3, M)
            R_ij = R_wc_i @ poses[j, :3, :3]
            t_ij = R_wc_i @ poses[j, :3, 3] + t_wc_i
            rho_P_i = R_ij @ ray_j + np.outer(t_ij, rho_j)          # (3, M)
            z_scaled = rho_P_i[2]                                   # = rho_j * Z_i
            in_front = z_scaled > 1e-6
            if not in_front.any():
                continue
            u_i = FX * rho_P_i[0, in_front] / z_scaled[in_front] + cx
            v_i = FX * rho_P_i[1, in_front] / z_scaled[in_front] + cy
            rho_i = rho_j[in_front] / z_scaled[in_front]            # = 1/Z_i
            in_img = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
            u_i = u_i[in_img].astype(np.int32)
            v_i = v_i[in_img].astype(np.int32)
            rho_i = rho_i[in_img]
            # Z-buffer in rho-space: closer wins, i.e. larger rho wins.
            flat = v_i * W + u_i
            buf = np.full(H * W, -np.inf, dtype=np.float32)
            np.maximum.at(buf, flat, rho_i)
            warped = buf.reshape(H, W)
            warped[~np.isfinite(warped) | (warped <= 0)] = np.nan
            stack[k] = warped

        valid_counts = np.sum(np.isfinite(stack), axis=0)
        fused_rho = np.nanmedian(stack, axis=0)  # median in rho — uniform sigma
        fused_rho = np.where(valid_counts >= MIN_VALID, fused_rho, rho_all[i])
        stereo_ok = np.isfinite(fused_rho) & (fused_rho > 0)
        inv_stereo = np.where(stereo_ok, fused_rho, 0.0)

        rho_mono = mono_rho_all[i]
        mono_ok = rho_mono > 0

        if stereo_ok.sum() > 2000 and mono_ok.sum() > 2000:
            joint = stereo_ok & mono_ok
            s = inv_stereo[joint]
            m = rho_mono[joint]
            A = np.stack([m, np.ones_like(m)], axis=1)
            ab, *_ = np.linalg.lstsq(A, s, rcond=None)
            res = np.abs(A @ ab - s)
            keep = res < np.percentile(res, 90)
            ab, *_ = np.linalg.lstsq(A[keep], s[keep], rcond=None)
            a, b = float(ab[0]), float(ab[1])
            a_sum += a
            b_sum += b
            calib_n += 1
            cal_mono = a * rho_mono + b
            final_inv = np.where(stereo_ok, inv_stereo, cal_mono)
            valid_mask = stereo_ok | mono_ok
        else:
            final_inv = inv_stereo
            valid_mask = stereo_ok

        pix16 = encode_16(final_inv, valid_mask)
        cv2.imwrite(str(tmp / f"f_{i:06d}.png"), pix16)
        cover_valid += valid_mask.mean()

        # stereo-ONLY stream (no mono-fill): sky stays rho=0/invalid by
        # construction (SGBM disp~0 there). This is the stream all motion/veto
        # reasoning must read — mono hallucinates finite sky depth.
        cv2.imwrite(str(tmp_stereo / f"f_{i:06d}.png"),
                    encode_16(inv_stereo, stereo_ok))

    print(f"[fuse+encode] {N} frames in {time.time()-t0:.1f}s")
    print(f"[mean] valid coverage = {cover_valid/N:.1%}")
    print(f"[calib] mean affine a={a_sum/calib_n:.3f}  b={b_sum/calib_n:.4f}  "
          f"(rho_stereo = a*rho_mono + b)")

    def encode_mkv(frame_dir: Path, dst: Path) -> None:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "10",
            "-i", str(frame_dir / "f_%06d.png"),
            "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
            "-pix_fmt", "gray16le",
            str(dst),
        ], check=True)

    encode_mkv(tmp, out_path)
    encode_mkv(tmp_stereo, out_stereo)
    shutil.rmtree(tmp)
    shutil.rmtree(tmp_stereo)
    print(f"[write] {out_path}")
    print(f"[write] {out_stereo}  (stereo-only, no mono-fill)")


if __name__ == "__main__":
    main()
