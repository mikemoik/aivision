"""Two-view geometry: essential matrix, pose recovery, triangulation.

This is the building block of a full SfM pipeline. Multi-view bundle adjustment
should be delegated to pycolmap once the two-view step is solid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Intrinsics


@dataclass
class TwoViewResult:
    R: np.ndarray  # 3x3 rotation from frame A to frame B
    t: np.ndarray  # 3x1 translation (unit norm — scale is unobservable from two views)
    mask: np.ndarray  # (N,) bool — inliers from RANSAC + cheirality
    points_3d: np.ndarray  # (M, 3) triangulated inlier points in A's frame
    pts_a: np.ndarray  # (M, 2) inlier pixel coords in A
    pts_b: np.ndarray  # (M, 2) inlier pixel coords in B


def estimate_two_view(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    intr: Intrinsics,
    ransac_threshold: float = 1.0,
    prob: float = 0.999,
) -> TwoViewResult | None:
    """Estimate relative pose between two frames from matched pixel correspondences.

    Returns None if too few matches survive.
    """
    if len(pts_a) < 8:
        return None

    K = intr.K
    E, mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=prob, threshold=ransac_threshold
    )
    if E is None:
        return None
    mask = mask.ravel().astype(bool)
    if mask.sum() < 8:
        return None

    _, R, t, pose_mask = cv2.recoverPose(E, pts_a[mask], pts_b[mask], K)
    pose_mask = pose_mask.ravel().astype(bool)

    inlier_a = pts_a[mask][pose_mask]
    inlier_b = pts_b[mask][pose_mask]
    if len(inlier_a) < 8:
        return None

    # Triangulate in the canonical pair (P0 = K[I|0], P1 = K[R|t]).
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P0, P1, inlier_a.T, inlier_b.T)
    pts3d = (pts4d[:3] / pts4d[3]).T  # (M, 3)

    # Combine masks back to original indexing.
    combined = np.zeros_like(mask)
    inlier_idx = np.flatnonzero(mask)[pose_mask]
    combined[inlier_idx] = True

    return TwoViewResult(
        R=R, t=t, mask=combined, points_3d=pts3d, pts_a=inlier_a, pts_b=inlier_b
    )
