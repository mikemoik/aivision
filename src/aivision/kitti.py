"""KITTI odometry sequence loader for GT-anchored mono-SfM benchmarking.

Expected per-sequence layout (flat — one dir per sequence under data/kitti/):
    <seq_dir>/
      image_0/000000.png ...   # left grayscale, rectified (default)
      image_1/                 # right grayscale (optional, not used here)
      times.txt                # one float per line, seconds since frame 0
      calib.txt                # rows: P0 P1 P2 P3 Tr  (each 3x4)
      poses.txt                # 12 floats per line, 3x4 cam0-to-world

For mono benchmarking we read `image_0/` and use K from the upper-left 3x3 of
P0 — both are the cam0 frame the GT poses live in, so no extrinsic offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .camera import Intrinsics


@dataclass(frozen=True)
class KittiSequence:
    seq_dir: Path
    image_paths: list[Path]
    times_s: np.ndarray            # (N,) seconds since frame 0
    K: np.ndarray                  # (3, 3) intrinsics from P0
    width: int
    height: int
    gt_poses: np.ndarray | None    # (N, 4, 4) cam-to-world, or None

    @property
    def n_frames(self) -> int:
        return len(self.image_paths)

    @property
    def intrinsics(self) -> Intrinsics:
        return Intrinsics(
            fx=float(self.K[0, 0]),
            fy=float(self.K[1, 1]),
            cx=float(self.K[0, 2]),
            cy=float(self.K[1, 2]),
            width=self.width,
            height=self.height,
        )

    def iter_frames(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (name, BGR image) — name matches the `frame_NNNNNN` convention used in io.py.

        image_0 is single-channel grayscale; we promote to BGR so the rest of the
        pipeline (which assumes 3-channel) doesn't need a branch.
        """
        for i, p in enumerate(self.image_paths):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"cv2 failed to read {p}")
            yield f"frame_{i:06d}", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def gt_positions(self) -> np.ndarray:
        """(N, 3) camera centres in world frame from GT poses."""
        if self.gt_poses is None:
            raise ValueError(f"Sequence {self.seq} has no GT poses available")
        return self.gt_poses[:, :3, 3]


def _read_calib(calib_path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in calib_path.read_text().strip().splitlines():
        key, _, rest = line.partition(":")
        vals = np.fromstring(rest, sep=" ", dtype=np.float64)
        out[key.strip()] = vals.reshape(3, 4)
    return out


def _read_poses(poses_path: Path) -> np.ndarray:
    raw = np.loadtxt(poses_path, dtype=np.float64)
    n = raw.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = raw.reshape(n, 3, 4)
    return out


def load_sequence(seq_dir: Path | str, camera: str = "image_0") -> KittiSequence:
    """Load a KITTI odometry sequence dir for mono benchmarking.

    `camera` picks the imagery subdir (and the matching P-matrix for K):
    image_0/P0 = left grayscale, image_1/P1 = right grayscale,
    image_2/P2 = left color, image_3/P3 = right color.
    """
    seq_dir = Path(seq_dir)
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"No such sequence directory: {seq_dir}")

    p_key = {"image_0": "P0", "image_1": "P1", "image_2": "P2", "image_3": "P3"}[camera]

    image_paths = sorted((seq_dir / camera).glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNGs in {seq_dir / camera}")

    times_s = np.loadtxt(seq_dir / "times.txt", dtype=np.float64)
    if times_s.shape[0] != len(image_paths):
        raise ValueError(
            f"times.txt has {times_s.shape[0]} rows, {len(image_paths)} images on disk"
        )

    calib = _read_calib(seq_dir / "calib.txt")
    K = calib[p_key][:3, :3].copy()

    sample = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    height, width = sample.shape[:2]

    poses_path = seq_dir / "poses.txt"
    gt_poses = _read_poses(poses_path) if poses_path.exists() else None

    return KittiSequence(
        seq_dir=seq_dir,
        image_paths=image_paths,
        times_s=times_s,
        K=K,
        width=width,
        height=height,
        gt_poses=gt_poses,
    )


# ----------------------------------------------------------------------------
# Sim(3)-aligned ATE — the standard mono-VO metric since scale is unobservable.
# ----------------------------------------------------------------------------


def umeyama_similarity(
    src: np.ndarray, dst: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate (s, R, t) such that s * R @ src.T + t ≈ dst.T. Both (N, 3)."""
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s_c = src - mu_s
    d_c = dst - mu_d
    var_s = (s_c**2).sum() / src.shape[0]
    H = (d_c.T @ s_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def absolute_trajectory_error(
    est_xyz: np.ndarray, gt_xyz: np.ndarray
) -> dict[str, float]:
    """Sim(3)-aligned ATE. Returns rmse/mean/median/max in metres, plus the recovered scale."""
    s, R, t = umeyama_similarity(est_xyz, gt_xyz)
    aligned = (s * R @ est_xyz.T).T + t
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": s,
    }


def relative_pose_error(
    est_xyz: np.ndarray, gt_xyz: np.ndarray, delta: int = 1
) -> dict[str, float]:
    """Per-step translation error after Sim(3) alignment.

    Uses a fixed frame gap `delta` (10 Hz → delta=10 means 1 s windows). The
    norm of the step difference is the noise-isolating metric we wanted in
    place of fwd_CV — it does not respond to real speed variation.
    """
    s, R, t = umeyama_similarity(est_xyz, gt_xyz)
    aligned = (s * R @ est_xyz.T).T + t
    est_step = aligned[delta:] - aligned[:-delta]
    gt_step = gt_xyz[delta:] - gt_xyz[:-delta]
    err = np.linalg.norm(est_step - gt_step, axis=1)
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "delta": delta,
    }
