"""Depth-aware bundle adjustment for forward-motion dashcam reconstructions.

On forward motion, far tracks (inv_d → 0) carry mostly rotation evidence —
their reprojection cost is nearly insensitive to translation. Folding ~6k
far tracks into a standard BA over a 14k-point cloud lets noise-dominated
translation gradients drag the poses around. This module runs a *near-only*
BA: drop far tracks from the cost entirely, refine poses + intrinsics +
near points only, then re-triangulate the full cloud against the refined
poses to recover the far points.

If the conditioning hypothesis is right, the camera-balls overlay should
tighten visibly compared to the global-BA baseline in
`outputs/ar_run/sfm/sparse/`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from .depth_analysis import analyze_track_depths


def run_depth_aware_ba(
    reconstruction_dir: Path,
    database_path: Path,
    image_dir: Path,
    out_dir: Path,
    far_depth_m: float = 50.0,
    retriangulate_after: bool = True,
) -> pycolmap.Reconstruction:
    """Near-only BA → re-triangulate → return refined reconstruction.

    Tracks with median observed depth >= ``far_depth_m`` are excluded from
    the BA cost; the remaining near tracks drive pose/intrinsic refinement.
    With ``retriangulate_after=True``, the full cloud is regenerated against
    the refined poses via ``pycolmap.triangulate_points``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    print(f"[depth-ba] loaded recon: {rec.num_reg_images()} images, "
          f"{rec.num_points3D()} points")

    analysis = analyze_track_depths(rec)
    arr = analysis["arrays"]
    point_ids = arr["point_ids"]
    med_d = arr["median_depth"]
    near_mask = med_d < far_depth_m
    near_ids = set(int(pid) for pid in point_ids[near_mask])
    far_ids = set(int(pid) for pid in point_ids[~near_mask])
    n_near = len(near_ids)
    n_far = len(far_ids)
    print(f"[depth-ba] threshold {far_depth_m:.0f} m → "
          f"{n_near} near tracks (variable), {n_far} far tracks (excluded from cost)")

    config = pycolmap.BundleAdjustmentConfig()
    for img_id in rec.images:
        config.add_image(img_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
    for pid in rec.points3D:
        if int(pid) in near_ids:
            config.add_variable_point(pid)
    print(f"[depth-ba] BA config: {config.num_variable_points()} variable points, "
          f"{config.num_residuals(rec)} residuals")

    ba_opts = pycolmap.BundleAdjustmentOptions()
    ba_opts.ceres.solver_options.minimizer_progress_to_stdout = False
    ba_opts.print_summary = True
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(ba_opts, config, rec)
    adjuster.solve()

    sparse_pre = out_dir / "sparse_pre_retri"
    sparse_pre.mkdir(parents=True, exist_ok=True)
    rec.write(str(sparse_pre))
    print(f"[depth-ba] near-only BA done; pre-retri recon → {sparse_pre}")

    if retriangulate_after:
        sparse_out = out_dir / "sparse"
        if sparse_out.exists():
            shutil.rmtree(sparse_out)
        sparse_out.mkdir(parents=True)
        # triangulate_points needs a clean recon with poses but empty points.
        seed = pycolmap.Reconstruction()
        seed.read(str(sparse_pre))
        seed.clear_points3D() if hasattr(seed, "clear_points3D") else None
        # If clear_points3D isn't available, write a text recon with empty points.
        seed_dir = out_dir / "sparse_seed"
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        seed_dir.mkdir(parents=True)
        # Reuse the existing helper from triangulate.py to write a text recon
        # with current poses + empty points3D.
        _write_recon_with_empty_points(rec, seed_dir)
        seed_recon = pycolmap.Reconstruction()
        seed_recon.read_text(str(seed_dir))
        print(f"[depth-ba] re-triangulating full cloud against refined poses…")
        refined = pycolmap.triangulate_points(
            reconstruction=seed_recon,
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(sparse_out),
            refine_intrinsics=False,
        )
        print(f"[depth-ba] re-triangulation: {refined.num_points3D()} points "
              f"(was {rec.num_points3D()}) → {sparse_out}")
        rec = refined

    diagnostics = {
        "threshold_m": far_depth_m,
        "n_near": n_near,
        "n_far": n_far,
        "n_points_final": int(rec.num_points3D()),
        "n_images": int(rec.num_reg_images()),
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[depth-ba] done. Final recon: {rec.num_points3D()} points / "
          f"{rec.num_reg_images()} images")
    return rec


def _write_recon_with_empty_points(rec: pycolmap.Reconstruction, out_dir: Path) -> None:
    """Write a text-format COLMAP reconstruction (cameras + images with poses,
    empty points3D) for triangulate_points to consume."""
    from scipy.spatial.transform import Rotation

    cameras_lines = [
        "# Camera list (auto-generated)",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for cam_id, cam in rec.cameras.items():
        params = " ".join(f"{p}" for p in cam.params)
        cameras_lines.append(
            f"{cam_id} {cam.model_name} {cam.width} {cam.height} {params}"
        )
    (out_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n")

    images_lines = [
        "# Image list (auto-generated)",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] (one line per image, empty here)",
    ]
    for img_id, img in rec.images.items():
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        images_lines.append(
            f"{img_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} "
            f"{img.camera_id} {img.name}"
        )
        images_lines.append("")
    (out_dir / "images.txt").write_text("\n".join(images_lines) + "\n")

    (out_dir / "points3D.txt").write_text(
        "# 3D point list (empty — to be filled by triangulate_points)\n"
    )
