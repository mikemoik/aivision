"""Post-hoc trajectory smoothing for SfM reconstructions.

Diagnostic tool: applies a Savitzky-Golay filter to per-frame camera centres
to suppress the per-step pose noise observed on forward-only dashcam
trajectories (see memory: task_pose_refinement_round2). Rotations are kept
intact on the first pass; only translations are rewritten.

Writes a new ``sparse/`` directory containing the smoothed cameras and the
*original* 3D points and observations. Reprojection error will degrade
slightly — that's expected; the point of this module is to verify that the
recon's *average* motion is correct and only the per-step direction is
noisy. If the visual trail snaps to a smooth line after smoothing, the
deeper fix (smoothness prior in BA, or inverse-depth BA) is the right next
step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
from scipy.signal import savgol_filter


def _image_name_sort_key(name: str) -> int:
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def _trajectory_stats(centres: np.ndarray) -> dict[str, float]:
    """Step statistics. Forward axis = unit vector from first to last centre.
    Reports both total-3D step length and forward-projected step length
    (the latter is what reveals straight-line BA conditioning failure)."""
    steps = np.diff(centres, axis=0)
    step_lens = np.linalg.norm(steps, axis=1)
    if len(step_lens) < 2:
        return {
            "mean_step": 0.0, "std_step": 0.0, "cv_step": 0.0,
            "mean_fwd": 0.0, "std_fwd": 0.0, "cv_fwd": 0.0,
        }
    fwd_axis = centres[-1] - centres[0]
    fwd_axis /= max(np.linalg.norm(fwd_axis), 1e-9)
    fwd_steps = steps @ fwd_axis
    return {
        "mean_step": float(step_lens.mean()),
        "std_step": float(step_lens.std()),
        "cv_step": float(step_lens.std() / max(step_lens.mean(), 1e-9)),
        "mean_fwd": float(fwd_steps.mean()),
        "std_fwd": float(fwd_steps.std()),
        "cv_fwd": float(fwd_steps.std() / max(abs(fwd_steps.mean()), 1e-9)),
    }


def smooth_reconstruction(
    reconstruction_dir: Path,
    out_dir: Path,
    window: int = 11,
    poly: int = 3,
) -> pycolmap.Reconstruction:
    """Read a recon, smooth camera centres with Savitzky-Golay, write a new recon.

    Args:
        reconstruction_dir: input COLMAP sparse/ directory.
        out_dir: output sparse/ directory (created/overwritten).
        window: S-G window length (must be odd, >= poly+2).
        poly: polynomial order for the local fit.

    Returns:
        the in-memory smoothed `pycolmap.Reconstruction`.
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if window <= poly + 1:
        raise ValueError(
            f"window ({window}) must be > poly ({poly}) + 1; "
            "increase --window or decrease --poly"
        )

    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    img_pairs = sorted(
        rec.images.items(), key=lambda kv: _image_name_sort_key(kv[1].name)
    )
    n = len(img_pairs)
    if n < window:
        raise ValueError(
            f"recon has {n} images but window={window}; need at least "
            f"window images"
        )

    centres = np.stack(
        [
            -img.cam_from_world().rotation.matrix().T
            @ np.asarray(img.cam_from_world().translation, dtype=np.float64)
            for _, img in img_pairs
        ],
        axis=0,
    )

    pre = _trajectory_stats(centres)
    smoothed = savgol_filter(
        centres, window_length=window, polyorder=poly, axis=0, mode="interp"
    )
    post = _trajectory_stats(smoothed)
    centre_shift = np.linalg.norm(smoothed - centres, axis=1)

    print(
        f"[smooth] {n} cameras, S-G window={window}, poly={poly}\n"
        f"[smooth]   3D-step mean: {pre['mean_step']:.3f} m  →  "
        f"{post['mean_step']:.3f} m\n"
        f"[smooth]   3D-step std:  {pre['std_step']:.3f} m  →  "
        f"{post['std_step']:.3f} m  (CV "
        f"{pre['cv_step']:.2f} → {post['cv_step']:.2f})\n"
        f"[smooth]   fwd  mean:    {pre['mean_fwd']:.3f} m  →  "
        f"{post['mean_fwd']:.3f} m\n"
        f"[smooth]   fwd  std:     {pre['std_fwd']:.3f} m  →  "
        f"{post['std_fwd']:.3f} m  (CV "
        f"{pre['cv_fwd']:.2f} → {post['cv_fwd']:.2f})\n"
        f"[smooth]   per-camera centre shift: mean={centre_shift.mean():.3f} m, "
        f"max={centre_shift.max():.3f} m"
    )

    for (_, img), new_c in zip(img_pairs, smoothed):
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        new_t = -R @ new_c
        img.frame.set_cam_from_world(
            img.camera_id,
            pycolmap.Rigid3d(rotation=pose.rotation, translation=new_t),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    print(f"[smooth] wrote smoothed recon → {out_dir}")
    return rec
