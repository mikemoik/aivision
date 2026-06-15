"""Depth step (fused inverse-depth / rho video) — T2 of the map-gen workflow.

Lives outside ``mapgen.py`` so importing the framework never drags in cv2 / the
Depth Pro (torch) toolchain; the heavy imports stay inside the builder, which is
only called from :meth:`DepthStep.process`. Registered as
``(depth, inv-depth-fused-mono-rho-gf)``.

Pipeline (recovered from ``scripts/build_rho_video.py``, now the lib core here):
SGBM stereo → guided-filter edge-aware smoothing of each rho map → ±WINDOW
pose-warped rho-median fusion → Depth Pro mono fill via a per-frame robust
affine → 16-bit FFV1 ``gray16le`` MKV.

Everything is carried in *rho* (inverse depth, 1/m) — Z is never materialised, so
the fusion stays numerically bounded even for far/sky pixels. Encoding: pixel 0
= invalid, pixel ∈ [1, 65535] → ``rho = (pixel - 1) / 65534`` and ``Z = 1/rho``.

Out contract (``map-gen/depth/out/``):
  * ``inv_depth_rho16.mkv``         — N×H×W gray16 mono-fused rho stream
  * ``inv_depth_rho16_stereo.mkv``  — stereo-ONLY sibling (no mono fill; sky stays
                                      invalid — the stream motion/veto reasoning
                                      must read, since mono hallucinates sky depth)
  * ``inv_depth_rho16.json``        — sidecar {fx, baseline_m, rho encoding,
                                      n_frames, valid coverage, mono affine, ...}
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .mapgen import MapGenContext, MapGenStep, register_step

# Fusion + guided-filter knobs (defaults pulled from the original producer; the
# guided filter r=4/eps=1e-2 was chosen by a frame-0 sweep — best local-sigma
# reduction with least gradient inflation at validity borders).
WINDOW = 5          # ±frames pose-warped into the fusion buffer
MIN_VALID = 3       # min warped views to trust the fused-median over the raw rho
GF_RADIUS = 4
GF_EPS = 1e-2
GF_MIN_SUPPORT = 0.3  # min filtered validity weight to trust a smoothed pixel

RHO_ENCODING = "pixel 0 = invalid; pixel in [1,65535] -> rho=(pixel-1)/65534, Z=1/rho"


def _encode_16(rho: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rho_clipped = np.clip(rho, 0.0, 1.0)
    pix = np.zeros(rho.shape, dtype=np.uint16)
    pix[valid_mask] = np.round(1.0 + rho_clipped[valid_mask] * 65534.0).astype(np.uint16)
    return pix


def _guided_smooth_rho(rho: np.ndarray, guide_gray: np.ndarray) -> np.ndarray:
    """Edge-aware smoothing of a sparse rho map via normalised convolution.

    Filters ``rho*valid`` and ``valid`` with the same guided filter so invalid
    (zero) pixels carry no weight, then re-masks to the original validity —
    smoothing only, no inpainting.
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


def _encode_mkv(frame_dir: Path, dst: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "10",
        "-i", str(frame_dir / "f_%06d.png"),
        "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
        "-pix_fmt", "gray16le",
        str(dst),
    ], check=True)


def build_fused_rho(
    left_paths: list[Path],
    right_paths: list[Path],
    poses_twc: np.ndarray,
    fx: float,
    cx: float,
    cy: float,
    baseline_m: float,
    out_path: Path,
    out_stereo_path: Path,
    *,
    guided: bool = True,
    log=None,
) -> dict:
    """Build the fused-rho + stereo-only 16-bit mkvs; return a sidecar-stats dict.

    ``poses_twc`` is (N, 4, 4) camera-to-world (Twc) metric. Mono Depth Pro is run
    on the *left* image read as colour (matching the original producer). ``fx`` is
    used for both axes (KITTI fy == fx). Returns {n_frames, fx, baseline_m,
    mean_valid_coverage, mono_affine_a/b, median_rho, rho_encoding, window, min_valid}.
    """
    def _say(msg: str) -> None:
        (log.info if log is not None else print)(msg)

    from ..perception import _get_depth_model, mono_depth_frame

    N = len(left_paths)
    if len(right_paths) != N:
        raise ValueError(f"left/right frame count mismatch: {N} vs {len(right_paths)}")
    if poses_twc.shape[0] < N:
        raise ValueError(f"{poses_twc.shape[0]} poses for {N} frames")
    sample = cv2.imread(str(left_paths[0]), cv2.IMREAD_GRAYSCALE)
    H, W = sample.shape[:2]
    bf = fx * baseline_m

    R_wc_all = np.transpose(poses_twc[:, :3, :3], (0, 2, 1))           # R_cw
    t_wc_all = -np.einsum("nij,nj->ni", R_wc_all, poses_twc[:, :3, 3])  # t_cw

    # --- 1. SGBM, stored as rho = disp / bf (never converted to Z) -----------
    block = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=384, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    rho_all = np.full((N, H, W), np.nan, dtype=np.float32)
    for i in range(N):
        L = cv2.imread(str(left_paths[i]), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(right_paths[i]), cv2.IMREAD_GRAYSCALE)
        disp = sgbm.compute(L, R).astype(np.float32) / 16.0
        rho = np.where(disp > 0, disp / bf, np.nan)  # rho [1/m]
        if guided:
            rho = _guided_smooth_rho(rho, L)
        rho_all[i] = rho
    median_rho = float(np.nanmedian(rho_all))
    _say(f"[sgbm{'+gf' if guided else ''}] {N} frames, median rho={median_rho:.4f} 1/m")

    # --- 2. Mono Depth Pro, converted once to rho_mono = 1/Z -----------------
    _get_depth_model()
    mono_rho_all = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        bgr = cv2.imread(str(left_paths[i]), cv2.IMREAD_COLOR)
        d, _ = mono_depth_frame(bgr)
        d = d.astype(np.float32)
        mono_rho_all[i] = np.where(d > 0, 1.0 / d, 0.0)
    _say(f"[mono] {N} frames")

    # --- 3. Per-frame multi-frame fusion done entirely in rho-space ----------
    # For a pixel of frame j with rho_j, after the relative transform
    # (R_ij = R_cw_i @ R_wc_j, t_ij = R_cw_i @ t_wc_j + t_cw_i):
    #     rho_j * P_i = R_ij @ ray_j + rho_j * t_ij      (bounded — no 1/Z)
    # so u_i = fx*(rho_j*P_i)[0]/(rho_j*P_i)[2] + cx and rho_i = rho_j/(rho_j*P_i)[2].
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rx = (uu - cx) / fx
    ry = (vv - cy) / fx  # fy = fx

    tmp = out_path.parent / "_invd_rho_frames"
    tmp_stereo = out_path.parent / "_invd_rho_stereo_frames"
    for d in (tmp, tmp_stereo):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    cover_valid = 0.0
    calib_n = 0
    a_sum = b_sum = 0.0
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
            R_ij = R_wc_i @ poses_twc[j, :3, :3]
            t_ij = R_wc_i @ poses_twc[j, :3, 3] + t_wc_i
            rho_P_i = R_ij @ ray_j + np.outer(t_ij, rho_j)          # (3, M)
            z_scaled = rho_P_i[2]                                   # = rho_j * Z_i
            in_front = z_scaled > 1e-6
            if not in_front.any():
                continue
            u_i = fx * rho_P_i[0, in_front] / z_scaled[in_front] + cx
            v_i = fx * rho_P_i[1, in_front] / z_scaled[in_front] + cy
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

        cv2.imwrite(str(tmp / f"f_{i:06d}.png"), _encode_16(final_inv, valid_mask))
        cover_valid += valid_mask.mean()
        # stereo-ONLY stream (no mono-fill): sky stays rho=0/invalid by
        # construction (SGBM disp~0 there).
        cv2.imwrite(str(tmp_stereo / f"f_{i:06d}.png"),
                    _encode_16(inv_stereo, stereo_ok))

    mean_cover = cover_valid / N
    a_mean = a_sum / calib_n if calib_n else float("nan")
    b_mean = b_sum / calib_n if calib_n else float("nan")
    _say(f"[fuse+encode] {N} frames, mean valid coverage={mean_cover:.1%}")
    _say(f"[calib] mean affine a={a_mean:.3f} b={b_mean:.4f} (rho_stereo = a*rho_mono + b)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _encode_mkv(tmp, out_path)
    _encode_mkv(tmp_stereo, out_stereo_path)
    shutil.rmtree(tmp)
    shutil.rmtree(tmp_stereo)
    _say(f"[write] {out_path}")
    _say(f"[write] {out_stereo_path}  (stereo-only, no mono-fill)")

    return {
        "n_frames": N,
        "width": W,
        "height": H,
        "fx": fx,
        "cx": cx,
        "cy": cy,
        "baseline_m": baseline_m,
        "rho_encoding": RHO_ENCODING,
        "mean_valid_coverage": mean_cover,
        "mono_affine_a": a_mean,
        "mono_affine_b": b_mean,
        "median_rho": median_rho,
        "window": WINDOW,
        "min_valid": MIN_VALID,
        "guided": guided,
    }


def _baseline_from_calib(seq_dir: Path, fx: float) -> float:
    """Stereo baseline (m) from the odometry calib: b = -P1[0,3] / fx."""
    from ..kitti import _read_calib

    p1 = _read_calib(seq_dir / "calib.txt")["P1"]
    return float(-p1[0, 3] / fx)


def _poses_from_trajectory(traj_json: Path) -> np.ndarray:
    """trajectory.json (Nx12 Twc row-major) → (N, 4, 4) homogeneous Twc."""
    rows = np.asarray(json.loads(traj_json.read_text())["poses"], dtype=np.float64)
    n = rows.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = rows.reshape(n, 3, 4)
    return out


@register_step
class DepthStep(MapGenStep):
    """Fused inverse-depth (rho) video from stereo + mono, in the SLAM frame.

    Consumes the odometry slice (stereo image_0/1 + mono on image_0) and the
    upstream ``slam`` trajectory (poses drive the multi-frame rho fusion). Emits
    the rho mkv contract that ``objects`` reads.
    """

    role = "depth"
    type = "inv-depth-fused-mono-rho-gf"
    inputs = ["input:odometry", "slam"]

    def process(self, ctx: MapGenContext) -> list[str]:
        from ..kitti import load_sequence

        seq_dir = ctx.input("odometry")
        left = load_sequence(seq_dir, camera="image_0")
        right = load_sequence(seq_dir, camera="image_1")
        fx = float(left.K[0, 0])
        cx = float(left.K[0, 2])
        cy = float(left.K[1, 2])
        baseline_m = float(self.params.get("baseline_m") or _baseline_from_calib(seq_dir, fx))

        poses = _poses_from_trajectory(ctx.upstream_out("slam") / "trajectory.json")
        ctx.log.info("depth: seq=%s fx=%.3f baseline=%.4f n=%d",
                     seq_dir, fx, baseline_m, left.n_frames)

        out_dir = ctx.out("depth")
        out_path = out_dir / "inv_depth_rho16.mkv"
        out_stereo = out_dir / "inv_depth_rho16_stereo.mkv"
        stats = build_fused_rho(
            left.image_paths, right.image_paths, poses,
            fx, cx, cy, baseline_m, out_path, out_stereo,
            guided=bool(self.params.get("guided", True)), log=ctx.log,
        )
        (out_dir / "inv_depth_rho16.json").write_text(json.dumps(stats, indent=2))
        return [
            "map-gen/depth/out/inv_depth_rho16.mkv",
            "map-gen/depth/out/inv_depth_rho16_stereo.mkv",
            "map-gen/depth/out/inv_depth_rho16.json",
        ]
