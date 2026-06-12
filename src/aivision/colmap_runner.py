"""Thin wrapper around pycolmap: features → sequential match → incremental SfM.

Designed for monocular video without prior calibration. Uses CameraMode.SINGLE
so all frames share one camera model, and pycolmap self-calibrates the focal
length during bundle adjustment.

When per-image position priors are supplied (e.g. interpolated GPS), they are
written into the database as `PosePrior` records in a local CARTESIAN frame and
`use_prior_position` is enabled so bundle adjustment uses them as soft
constraints. This fixes the initial-pair-selection failure on pure forward
motion and removes long-term VO-style drift.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pycolmap


def _inject_position_priors(
    db_path: Path, priors: dict[str, tuple[np.ndarray, float]]
) -> int:
    """Write per-image CARTESIAN PosePriors into a COLMAP database.

    `priors` maps image filename → (xyz_metres, accuracy_metres).
    Accuracy is converted to a diagonal isotropic covariance.
    Returns the number of priors written.
    """
    db = pycolmap.Database.open(str(db_path))
    try:
        images = db.read_all_images()
        written = 0
        for img in images:
            if img.name not in priors:
                continue
            xyz, acc = priors[img.name]
            pp = pycolmap.PosePrior()
            pp.position = np.asarray(xyz, dtype=np.float64)
            pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            pp.position_covariance = np.eye(3) * float(acc) ** 2
            pp.corr_data_id = img.data_id  # link prior to this image's sensor data
            db.write_pose_prior(pp)
            written += 1
        return written
    finally:
        db.close()


def run_sfm(
    image_dir: Path,
    workspace: Path,
    overlap: int = 20,
    use_gpu: bool = False,  # PyPI pycolmap wheel is CPU-only for SIFT
    position_priors: dict[str, tuple[np.ndarray, float]] | None = None,
) -> pycolmap.Reconstruction:
    """Run COLMAP incremental SfM on a directory of sequential frames.

    Parameters
    ----------
    image_dir : directory containing the input frames (sorted lexicographically).
    workspace : directory where database, sparse model, and intermediates go.
    overlap   : sequential-matcher overlap window (each image matched to next N).
    use_gpu   : run SIFT on GPU when available.

    Returns the largest reconstruction (most registered images).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    if db_path.exists():
        db_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir()

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        device=device,
    )

    if position_priors:
        n_written = _inject_position_priors(db_path, position_priors)
        print(f"      wrote {n_written} position priors into COLMAP database")

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pycolmap.match_sequential(
        database_path=db_path,
        pairing_options=pairing,
        device=device,
    )

    # Tune for monocular forward-driving footage. COLMAP's defaults assume
    # casual orbit-style capture; dashcam pairs are nearly pure forward motion
    # with tiny triangulation angles, which the defaults reject outright.
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.multiple_models = True
    mapper_opts.max_num_models = 10
    mapper_opts.min_num_matches = 10
    mapper_opts.min_model_size = 5
    mapper_opts.init_num_trials = 500

    # Initial pair selection — accept narrow-baseline forward motion.
    mapper_opts.mapper.init_max_forward_motion = 1.0   # default 0.95 rejects dashcam pairs
    mapper_opts.mapper.init_min_tri_angle = 4.0        # default 16.0 too strict
    mapper_opts.mapper.init_min_num_inliers = 50       # default 100
    mapper_opts.mapper.init_max_reg_trials = 5         # default 2

    # Subsequent image registration — relax PnP thresholds.
    mapper_opts.mapper.abs_pose_min_num_inliers = 15   # default 30
    mapper_opts.mapper.abs_pose_min_inlier_ratio = 0.15  # default 0.25
    mapper_opts.mapper.filter_min_tri_angle = 0.5      # default 1.5

    # Triangulation — keep shallow-angle points (cars/buildings far ahead).
    mapper_opts.triangulation.min_angle = 0.5
    mapper_opts.triangulation.ignore_two_view_tracks = False

    # Use GPS-derived position priors when they were injected above.
    if position_priors:
        mapper_opts.use_prior_position = True
        mapper_opts.use_robust_loss_on_prior_position = True

    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_dir,
        options=mapper_opts,
    )
    if not reconstructions:
        raise RuntimeError("incremental_mapping produced no reconstruction")

    best_id = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
    return reconstructions[best_id]
