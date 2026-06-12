"""Lightweight chained two-view VO — a fallback when full SfM fails.

For each consecutive frame pair we compute relative pose via the essential
matrix, chain the poses globally (translation is unit-length per step — scale
is arbitrary and not the focus), triangulate, and accumulate the points in the
global frame. No bundle adjustment, no loop closure, so it drifts. But unlike
COLMAP it is robust to the pure-forward-motion case that dashcam footage hits.

Returns a `Scene` dataclass with the same surface the renderer needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .camera import Intrinsics
from .features import detect_orb, match_mutual
from .io import iter_image_dir


@dataclass
class Scene:
    """A minimal sparse reconstruction: coloured 3D points + camera poses.

    Poses are stored in OpenCV/COLMAP `cam_from_world` convention: a world
    point `x_world` maps to camera coordinates as `x_cam = cam_R @ x_world + cam_t`.
    """

    points_xyz: np.ndarray   # (N, 3) float64
    points_rgb: np.ndarray   # (N, 3) float64 in [0, 1]
    cam_R: np.ndarray         # (M, 3, 3) cam_from_world rotation per camera
    cam_t: np.ndarray         # (M, 3) cam_from_world translation per camera
    image_names: list[str]

    @property
    def camera_centers(self) -> np.ndarray:
        """World-frame camera positions: C_i = -R_i^T @ t_i."""
        return np.einsum("mij,mj->mi", np.transpose(self.cam_R, (0, 2, 1)), -self.cam_t)


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping src → dst (Umeyama 1991).

    Returns (scale, R, t) such that dst ≈ scale * R @ src + t.
    """
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    var_src = (src_c ** 2).sum() / n
    if var_src < 1e-12:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(S) @ D) / var_src)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def anchor_scene_to_gps(scene: Scene, gps_positions: np.ndarray) -> Scene:
    """Fit a similarity transform mapping VO camera centres to GPS positions,
    and apply it consistently to points and full camera poses.

    Derivation: under the world remap x' = s*R_a @ x + t_a, camera rotations
    update as R_new = R_old @ R_a.T (directions don't scale) and translations
    as t_new = s*t_old - R_new @ t_a (from the cam-centre invariant
    t_new = -R_new @ C_new).
    """
    if len(gps_positions) != len(scene.camera_centers):
        raise ValueError(
            f"gps_positions has {len(gps_positions)} rows but Scene has "
            f"{len(scene.camera_centers)} cameras"
        )
    s, R_a, t_a = _umeyama_similarity(scene.camera_centers, gps_positions)

    points_world = (s * (R_a @ scene.points_xyz.T)).T + t_a

    cam_R_new = scene.cam_R @ R_a.T  # (M, 3, 3) @ (3, 3) — broadcast on first axis
    # cam_t_new[i] = s * cam_t[i] - cam_R_new[i] @ t_a
    cam_t_new = s * scene.cam_t - np.einsum("mij,j->mi", cam_R_new, t_a)

    return Scene(
        points_xyz=points_world,
        points_rgb=scene.points_rgb,
        cam_R=cam_R_new,
        cam_t=cam_t_new,
        image_names=scene.image_names,
    )


def _sample_color(img_bgr: np.ndarray, x: float, y: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    b, g, r = img_bgr[yi, xi]
    return np.array([r, g, b], dtype=np.float64) / 255.0


def run_vo(
    image_dir: Path,
    hfov_deg: float = 60.0,
    stride: int = 1,
    min_matches: int = 30,
    min_inliers: int = 20,
    max_depth: float = 200.0,
) -> Scene:
    """Chain two-view triangulation across a directory of sequential frames."""
    items = list(iter_image_dir(image_dir))[::stride]
    if len(items) < 2:
        raise RuntimeError(f"Need >=2 frames, got {len(items)}")

    h, w = items[0][1].shape[:2]
    intr = Intrinsics.guess_from_hfov(width=w, height=h, hfov_deg=hfov_deg)
    K = intr.K

    # Global pose of each frame as cam_from_world (R, t). Frame 0 is the world.
    R_global = np.eye(3)
    t_global = np.zeros(3)
    Rs: list[np.ndarray] = [R_global.copy()]
    ts: list[np.ndarray] = [t_global.copy()]
    names = [items[0][0]]
    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []

    prev_feats = detect_orb(items[0][1])
    prev_img = items[0][1]

    for (name_a, _), (name_b, img_b) in tqdm(
        list(zip(items[:-1], items[1:])), desc="vo"
    ):
        feats_b = detect_orb(img_b)
        pairs = match_mutual(prev_feats, feats_b)
        if len(pairs) < min_matches:
            prev_feats, prev_img = feats_b, img_b
            continue

        pts_a = prev_feats.xy[pairs[:, 0]]
        pts_b = feats_b.xy[pairs[:, 1]]

        E, mask = cv2.findEssentialMat(
            pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            prev_feats, prev_img = feats_b, img_b
            continue
        mask = mask.ravel().astype(bool)
        if mask.sum() < min_inliers:
            prev_feats, prev_img = feats_b, img_b
            continue

        _, R_rel, t_rel, pose_mask = cv2.recoverPose(E, pts_a[mask], pts_b[mask], K)
        pose_mask = pose_mask.ravel().astype(bool)
        t_rel = t_rel.ravel()  # unit norm

        # Chain global pose: cam_from_world updates by left-multiplying the relative.
        R_prev_global, t_prev_global = R_global.copy(), t_global.copy()
        R_global = R_rel @ R_prev_global
        t_global = R_rel @ t_prev_global + t_rel

        in_a = pts_a[mask][pose_mask]
        in_b = pts_b[mask][pose_mask]
        if len(in_a) >= 8:
            P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P1 = K @ np.hstack([R_rel, t_rel.reshape(3, 1)])
            pts4d = cv2.triangulatePoints(P0, P1, in_a.T, in_b.T)
            pts3d_local = (pts4d[:3] / pts4d[3]).T  # in prev frame's coords

            # Lift to world: x_world = R_prev^T @ x_local - R_prev^T @ t_prev
            pts3d_world = (
                R_prev_global.T @ pts3d_local.T
                - (R_prev_global.T @ t_prev_global)[:, None]
            ).T

            # Reject behind-camera, very-far, and very-near points.
            z = pts3d_local[:, 2]
            keep = (z > 0.5) & (z < max_depth)
            if keep.any():
                colors = np.array([_sample_color(img_b, x, y) for x, y in in_b[keep]])
                all_xyz.append(pts3d_world[keep])
                all_rgb.append(colors)

        Rs.append(R_global.copy())
        ts.append(t_global.copy())
        names.append(name_b)
        prev_feats, prev_img = feats_b, img_b

    if not all_xyz:
        raise RuntimeError("VO produced no points")
    return Scene(
        points_xyz=np.vstack(all_xyz),
        points_rgb=np.vstack(all_rgb),
        cam_R=np.array(Rs),
        cam_t=np.array(ts),
        image_names=names,
    )
