"""Visualisation helpers — match overlays and 3D point clouds."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    out_path: Path,
    max_draw: int = 200,
) -> None:
    n = min(len(pts_a), max_draw)
    sel = np.random.default_rng(0).choice(len(pts_a), size=n, replace=False) if len(pts_a) > n else np.arange(len(pts_a))

    kps_a = [cv2.KeyPoint(float(x), float(y), 4) for x, y in pts_a[sel]]
    kps_b = [cv2.KeyPoint(float(x), float(y), 4) for x, y in pts_b[sel]]
    dmatches = [cv2.DMatch(i, i, 0) for i in range(n)]

    out = cv2.drawMatches(
        img_a, kps_a, img_b, kps_b, dmatches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def save_point_cloud_ply(points: np.ndarray, out_path: Path) -> None:
    """Write a minimal ASCII PLY — viewable in MeshLab / Open3D / CloudCompare."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")
