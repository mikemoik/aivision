"""Densify a VO + GPS reconstruction via COLMAP triangulation.

The classical incremental SfM fails on pure forward motion (initial-pair
selection is geometric, not pose-driven, so it rejects every dashcam pair).
This module sidesteps that: VO already gave us camera poses, GPS already
anchored them in metres, so we just need to triangulate the features
against our known poses.

Pipeline:
    1. Extract features and match sequentially. Two backends:
         - "sift": COLMAP's built-in SIFT extract + sequential matcher (CPU).
         - "disk": DISK + LightGlue on CUDA (denser, low-texture-friendly);
           we keep `extract_features` for its side effect of populating the
           camera + image tables, then overwrite keypoints / matches /
           two-view geometries with the DISK + LightGlue output.
    2. Take the GPS-anchored VO poses and write them as a text-format COLMAP
       reconstruction (cameras + images with poses, points3D empty).
    3. Call pycolmap.triangulate_points — this uses the matches in the
       database to triangulate 3D points against our known poses.
    4. Run bundle_adjustment to jointly refine poses, intrinsics, and points.

Result: a metric, GPS-anchored sparse cloud that is typically 5-20× denser
than the ORB-based VO output, with the DISK backend roughly doubling that on
low-texture road and sky regions.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from .vo import Scene


def _rotation_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation → COLMAP quaternion (qw, qx, qy, qz)."""
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()  # scipy returns xyzw
    return float(qw), float(qx), float(qy), float(qz)


def _write_text_reconstruction(
    out_dir: Path,
    scene: Scene,
    cameras: list[pycolmap.Camera],
    name_to_image_id: dict[str, int],
) -> None:
    """Write COLMAP text-format reconstruction (cameras.txt, images.txt,
    points3D.txt) using `scene`'s poses. Points file is empty — to be filled
    by triangulate_points."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras_lines = [
        "# Camera list (auto-generated)",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for cam in cameras:
        params = " ".join(f"{p}" for p in cam.params)
        cameras_lines.append(
            f"{cam.camera_id} {cam.model_name} {cam.width} {cam.height} {params}"
        )
    (out_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n")

    # All images use the first camera (CameraMode.SINGLE).
    shared_camera_id = cameras[0].camera_id

    images_lines = [
        "# Image list (auto-generated)",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] (one line per image, empty here)",
    ]
    written = 0
    for i, stem in enumerate(scene.image_names):
        name = stem + ".jpg"
        if name not in name_to_image_id:
            continue
        image_id = name_to_image_id[name]
        qw, qx, qy, qz = _rotation_to_quat_wxyz(scene.cam_R[i])
        tx, ty, tz = (float(x) for x in scene.cam_t[i])
        images_lines.append(
            f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {shared_camera_id} {name}"
        )
        images_lines.append("")  # empty 2D-observation line
        written += 1
    (out_dir / "images.txt").write_text("\n".join(images_lines) + "\n")

    (out_dir / "points3D.txt").write_text(
        "# 3D point list (empty — to be filled by triangulate_points)\n"
    )

    if written == 0:
        raise RuntimeError(
            "No scene images matched database entries — name mismatch?"
        )


def _sift_extract_match(image_dir: Path, db_path: Path, overlap: int) -> None:
    """COLMAP's built-in SIFT extract + sequential matcher (CPU)."""
    print("[densify] SIFT extract + sequential match…")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        device=pycolmap.Device.cpu,
    )
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pycolmap.match_sequential(
        database_path=db_path,
        pairing_options=pairing,
        device=pycolmap.Device.cpu,
    )


def _epipolar_sampson_residuals_px(
    kp1_norm: np.ndarray,  # (N, 2) normalised+undistorted image coords from cam1
    kp2_norm: np.ndarray,  # (M, 2)
    matches: np.ndarray,   # (P, 2) indices [i, j]
    R1: np.ndarray, t1: np.ndarray,  # cam_from_world for img1
    R2: np.ndarray, t2: np.ndarray,  # cam_from_world for img2
    focal: float,
) -> np.ndarray:
    """Per-match Sampson epipolar distance in pixels under the relative pose
    (R2, t2) @ (R1, t1)^{-1}.

    Distances are computed in normalised image space (intrinsics are baked
    into ``kp{1,2}_norm`` via ``cam.cam_from_img``) then scaled by ``focal``
    so the result is interpretable as pixels.

    Split out from the threshold mask so B4's per-instance motion test can
    consume the raw residuals alongside the standard epi-threshold filter.
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    t_cross = np.array([
        [0.0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0.0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0.0],
    ])
    E = t_cross @ R_rel  # essential matrix

    p1 = kp1_norm[matches[:, 0]]
    p2 = kp2_norm[matches[:, 1]]
    p1_h = np.concatenate([p1, np.ones((len(p1), 1))], axis=1)
    p2_h = np.concatenate([p2, np.ones((len(p2), 1))], axis=1)

    l2 = p1_h @ E.T  # epipolar lines for p1 in image 2
    l1 = p2_h @ E    # epipolar lines for p2 in image 1
    numer = np.sum(p2_h * l2, axis=1)
    denom = l1[:, 0] ** 2 + l1[:, 1] ** 2 + l2[:, 0] ** 2 + l2[:, 1] ** 2
    sampson_norm_sq = numer ** 2 / np.maximum(denom, 1e-18)
    return np.sqrt(sampson_norm_sq) * focal


def _epipolar_sampson_mask(
    kp1_norm: np.ndarray,
    kp2_norm: np.ndarray,
    matches: np.ndarray,
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
    focal: float,
    threshold_px: float,
) -> np.ndarray:
    """Boolean mask: keep matches with Sampson epipolar distance below
    ``threshold_px``. Thin wrapper around `_epipolar_sampson_residuals_px`."""
    return _epipolar_sampson_residuals_px(
        kp1_norm, kp2_norm, matches, R1, t1, R2, t2, focal,
    ) < threshold_px


def _triangulation_angle_mask(
    p1_norm: np.ndarray,  # (P, 2) normalised image coords for surviving matches in cam1
    p2_norm: np.ndarray,  # (P, 2)                                            ... in cam2
    R1: np.ndarray, t1: np.ndarray,  # cam_from_world for img1
    R2: np.ndarray, t2: np.ndarray,
    min_angle_deg: float,
) -> np.ndarray:
    """Boolean mask: keep matches whose triangulated 3D point makes an
    inter-ray angle ≥ ``min_angle_deg`` at the two camera centres.

    DLT triangulation per match in world frame, then arccos of the angle
    between (X - c1) and (X - c2). On forward motion this rejects matches
    near the epipole, where the rays are nearly parallel — those carry
    almost no parallax even though they satisfy the epipolar constraint.
    """
    P1 = np.concatenate([R1, t1[:, None]], axis=1)  # 3x4 cam_from_world
    P2 = np.concatenate([R2, t2[:, None]], axis=1)

    # DLT: row [u*P[2] - P[0]; v*P[2] - P[1]] for each view, stack 4 rows.
    u1 = p1_norm[:, 0]; v1 = p1_norm[:, 1]
    u2 = p2_norm[:, 0]; v2 = p2_norm[:, 1]
    A = np.stack([
        u1[:, None] * P1[2] - P1[0],
        v1[:, None] * P1[2] - P1[1],
        u2[:, None] * P2[2] - P2[0],
        v2[:, None] * P2[2] - P2[1],
    ], axis=1)  # (P, 4, 4)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]              # (P, 4) homogeneous null vector
    w = Xh[:, 3:4]
    safe_w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    X = Xh[:, :3] / safe_w         # (P, 3) world point

    c1 = -R1.T @ t1
    c2 = -R2.T @ t2
    r1 = X - c1
    r2 = X - c2
    n1 = np.linalg.norm(r1, axis=1)
    n2 = np.linalg.norm(r2, axis=1)
    safe = (n1 > 1e-9) & (n2 > 1e-9)
    cos_a = np.zeros(len(X))
    cos_a[safe] = np.einsum("ij,ij->i", r1[safe], r2[safe]) / (n1[safe] * n2[safe])
    cos_a = np.clip(cos_a, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_a))
    # Require cheirality too — kill matches that triangulate behind either camera.
    in_front1 = (R1 @ X.T + t1[:, None])[2] > 0
    in_front2 = (R2 @ X.T + t2[:, None])[2] > 0
    return safe & (angles >= min_angle_deg) & in_front1 & in_front2


_DYNAMIC_VISTAS_NAMES: tuple[str, ...] = (
    "Car", "Truck", "Bus", "Motorcycle", "Bicycle",
    "Person", "Caravan", "Trailer", "On Rails",
)


# B4 defaults — calibrated on dashcam pairs at overlap=50. Numbers worth
# revisiting if the static residual distribution turns out heavy-tailed in
# practice (then a robust statistic like MAD would beat mean+σ).
_B4_SIGMA_THRESHOLD = 3.0
_B4_MIN_STATIC_FOR_BASELINE = 50
_B4_MIN_OBS_PER_INSTANCE = 5


def _b4_drop_moving_instances(
    cls1: np.ndarray, inst1: np.ndarray,
    cls2: np.ndarray, inst2: np.ndarray,
    sampson_px: np.ndarray,
    dynamic_ids: np.ndarray,
    *,
    sigma_threshold: float = _B4_SIGMA_THRESHOLD,
    min_static_for_baseline: int = _B4_MIN_STATIC_FOR_BASELINE,
    min_obs_per_instance: int = _B4_MIN_OBS_PER_INSTANCE,
) -> tuple[np.ndarray, int, int, int]:
    """Per-instance geometric motion test for candidate-dynamic matches.

    For each side independently, group matches whose endpoint is a thing
    instance of a dynamic class (``cls in dynamic_ids`` and ``inst != 0``).
    Compute the per-instance mean Sampson residual; if it sits more than
    ``sigma_threshold`` σ above the static-baseline mean (matches with no
    dynamic endpoint on either side), classify that instance as moving and
    drop every match that touches it.

    Per-instance rather than per-pixel because a single high-residual pixel
    is dominated by noise — but a parked car's whole mask agreeing with the
    epipolar geometry is a strong static signal, and conversely a moving
    car's mask drifts off the epipolar lines coherently across all its
    keypoints. Drop the whole instance, not individual matches.

    Returns ``(keep_mask, n_moving, n_static, n_dropped)`` where
    ``n_moving + n_static`` is the number of distinct instance×side keys
    that passed the ``min_obs_per_instance`` gate.
    """
    n = len(cls1)
    if n == 0:
        return np.ones(0, dtype=bool), 0, 0, 0

    dyn1 = np.isin(cls1, dynamic_ids) & (inst1 != 0)
    dyn2 = np.isin(cls2, dynamic_ids) & (inst2 != 0)
    dyn_match = dyn1 | dyn2

    if not dyn_match.any():
        return np.ones(n, dtype=bool), 0, 0, 0

    static_mask = ~dyn_match
    n_static_obs = int(static_mask.sum())
    if n_static_obs < min_static_for_baseline:
        # Too few static matches to fit a baseline; skip B4 rather than
        # invent a threshold from inadequate data.
        return np.ones(n, dtype=bool), 0, 0, 0

    baseline_mean = float(sampson_px[static_mask].mean())
    baseline_std = float(sampson_px[static_mask].std())
    threshold = baseline_mean + sigma_threshold * max(baseline_std, 1e-6)

    drop = np.zeros(n, dtype=bool)
    n_moving = 0
    n_static_inst = 0
    # Pack (class_id, instance_id) into one int64 key. cls fits in 8 bits,
    # inst in 16 — well under 32 bits combined.
    for dyn, cls, inst in [(dyn1, cls1, inst1), (dyn2, cls2, inst2)]:
        dyn_idx = np.where(dyn)[0]
        if dyn_idx.size == 0:
            continue
        keys = (cls.astype(np.int64) << 16) | inst.astype(np.int64)
        keys_dyn = keys[dyn_idx]
        res_dyn = sampson_px[dyn_idx]

        unique_keys, inverse = np.unique(keys_dyn, return_inverse=True)
        sums = np.zeros(len(unique_keys), dtype=np.float64)
        counts = np.zeros(len(unique_keys), dtype=np.int64)
        np.add.at(sums, inverse, res_dyn)
        np.add.at(counts, inverse, 1)
        means = sums / np.maximum(counts, 1)

        eligible = counts >= min_obs_per_instance
        moving_inst = eligible & (means > threshold)
        n_moving += int(moving_inst.sum())
        n_static_inst += int(eligible.sum() - moving_inst.sum())

        if moving_inst.any():
            moving_keys = unique_keys[moving_inst]
            side_drop = np.isin(keys, moving_keys) & dyn
            drop |= side_drop

    return ~drop, n_moving, n_static_inst, int(drop.sum())


def _disk_lightglue_match(
    image_dir: Path,
    db_path: Path,
    overlap: int,
    n_keypoints: int = 8192,
    guide_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    epi_threshold_px: float = 4.0,
    min_tri_angle_deg: float = 1.0,
    perception_cache: Path | None = None,
) -> None:
    """Replace COLMAP's SIFT with DISK + LightGlue on CUDA.

    Still runs `extract_features` first for the side effect of populating
    the cameras + images tables with auto-calibrated intrinsics — but then
    clears SIFT keypoints / descriptors / matches / two-view geometries and
    writes DISK keypoints + LightGlue matches + geometrically-verified
    two-view geometries.

    If ``guide_poses`` is supplied (mapping image filename → ``(R, t)``,
    cam_from_world), each LightGlue match set is pre-filtered in two stages
    under the guide-pose geometry, before being handed to
    ``estimate_two_view_geometry``:

      1. Sampson epipolar distance < ``epi_threshold_px``.
      2. DLT-triangulated inter-ray angle ≥ ``min_tri_angle_deg``
         and positive depth in both cameras.

    Stage 2 is what makes guided matching work on forward motion: on
    pure-forward dashcam pairs the epipole sits inside the FOV, so matches
    near the principal point pass the epipolar test even when the rays are
    nearly parallel and carry no parallax. Triangulation-angle filtering
    rejects those degenerate matches and is the key to keeping the
    refined trajectory smooth.

    If ``perception_cache`` is supplied (path to a Phase-A perception cache
    built with the same source/step as the frames here), each LightGlue match
    set is additionally filtered against the cached panoptic segmentation:

      * **B2 — sky drop (always-on when cache present).** Matches whose
        either endpoint's Vistas class is ``Sky`` are dropped. These are
        infinity-points; keeping them poisons translation magnitude in BA.
      * **B3 — candidate-dynamic flag (instrumentation only).** Matches whose
        either endpoint's class is in {Car, Truck, Bus, Motorcycle, Bicycle,
        Person, Caravan, Trailer, On Rails} are counted but NOT dropped.
      * **B4 — per-instance geometric motion test (requires ``guide_poses``).**
        Group candidate-dynamic matches by (class, instance) on each side
        independently; flag instances whose mean Sampson residual is more
        than 3σ above the static-baseline distribution as moving, and drop
        every match touching them. Static baseline comes from non-candidate-
        dynamic matches on the same pair. Skipped when fewer than 50 static
        matches are available (baseline is undercooked). See
        `_b4_drop_moving_instances` for the per-knob defaults.

    Vistas class ids are resolved at call-time against the cache manifest's
    seg model (`id2label_from_manifest`), so a model swap surfaces as a
    label-resolution warning rather than as silent miscategorisation.
    """
    import torch
    import kornia.feature as KF
    from kornia.feature import laf_from_center_scale_ori

    # Phase-B semantic filter setup. None unless perception_cache is given.
    tagger = None
    sky_id: int | None = None
    dynamic_ids: np.ndarray | None = None
    if perception_cache is not None:
        from .perception import (
            KeypointTagger,
            id2label_from_manifest,
            _FRAME_NAME_RE,
        )
        tagger = KeypointTagger(Path(perception_cache))
        id2label = id2label_from_manifest(tagger.manifest)
        name_to_id = {v: int(k) for k, v in id2label.items()}
        if "Sky" not in name_to_id:
            raise RuntimeError(
                f"perception cache {perception_cache}: 'Sky' not in seg model "
                f"id2label — refusing to enable B2 filter without it. "
                f"seg_model={tagger.manifest.seg_model!r}"
            )
        sky_id = name_to_id["Sky"]
        dyn_resolved = sorted(name_to_id[n] for n in _DYNAMIC_VISTAS_NAMES if n in name_to_id)
        dyn_missing = [n for n in _DYNAMIC_VISTAS_NAMES if n not in name_to_id]
        dynamic_ids = np.asarray(dyn_resolved, dtype=np.uint8)
        print(
            f"[densify] perception filter: cache={perception_cache}, "
            f"sky_id={sky_id}, dynamic_ids={dyn_resolved}"
        )
        if dyn_missing:
            print(f"[densify] perception: dynamic classes missing from id2label: {dyn_missing}")

    print("[densify] seeding DB with COLMAP extract_features (for cameras + images)…")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        device=pycolmap.Device.cpu,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[densify] loading DISK + LightGlue on {device}…")
    disk = KF.DISK.from_pretrained("depth", device=device).eval()
    lg = KF.LightGlueMatcher("disk").to(device).eval()

    db = pycolmap.Database.open(str(db_path))
    try:
        images_db = db.read_all_images()
        cameras = db.read_all_cameras()
        cam = cameras[0]  # CameraMode.SINGLE
        name_to_id = {img.name: img.image_id for img in images_db}

        db.clear_keypoints()
        db.clear_descriptors()
        db.clear_matches()
        db.clear_two_view_geometries()

        sorted_names = sorted(name_to_id.keys())
        feats: dict[str, tuple[np.ndarray, "torch.Tensor", "torch.Tensor", tuple[int, int]]] = {}

        print(f"[densify] running DISK on {len(sorted_names)} images (n={n_keypoints})…")
        for name in sorted_names:
            img = cv2.cvtColor(cv2.imread(str(image_dir / name)), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            t = torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device) / 255.0
            with torch.inference_mode():
                f = disk(t, n=n_keypoints, pad_if_not_divisible=True)[0]
            kp = f.keypoints
            desc = f.descriptors
            N = kp.shape[0]
            scale = torch.ones(1, N, 1, 1, device=device)
            ori = torch.zeros(1, N, 1, device=device)
            lafs = laf_from_center_scale_ori(kp[None], scale, ori)
            kp_np = kp.detach().cpu().numpy().astype(np.float32)
            feats[name] = (kp_np, desc, lafs, (h, w))
            db.write_keypoints(name_to_id[name], kp_np)

        # Pre-tag every DISK keypoint with its semantic class + instance id,
        # if filtering. Instances are needed for B4's per-instance motion test.
        kp_classes: dict[str, np.ndarray] = {}
        kp_instances: dict[str, np.ndarray] = {}
        if tagger is not None:
            for name in sorted_names:
                m = _FRAME_NAME_RE.search(name)
                if not m:
                    raise ValueError(
                        f"perception filter requires frame_NNNNNN.jpg image "
                        f"names; got {name!r}"
                    )
                frame_idx = int(m.group(1))
                kp_xy = feats[name][0]
                tags = tagger.tag(frame_idx, kp_xy)
                kp_classes[name] = tags.semantic
                kp_instances[name] = tags.instance

        # Pre-undistort keypoints once for epipolar filtering, if guided.
        if guide_poses is not None:
            print(f"[densify] guided matching: epi<{epi_threshold_px:.1f} px, "
                  f"tri-angle≥{min_tri_angle_deg:.2f}°, chirality")
            kp_norm_by_name: dict[str, np.ndarray] = {
                name: cam.cam_from_img(feats[name][0].astype(np.float64))
                for name in sorted_names
            }
            focal = float(cam.params[0])

        print(f"[densify] LightGlue matching sequential pairs (overlap={overlap})…")
        n_pairs = 0
        n_inliers_total = 0
        n_epi_dropped = 0
        n_tri_dropped = 0
        n_sky_dropped = 0
        n_dynamic_flagged = 0
        n_match_total_pre_perception = 0
        n_b4_moving_instances = 0
        n_b4_static_instances = 0
        n_b4_dropped = 0
        for i, n1 in enumerate(sorted_names):
            kp1, desc1, laf1, hw1 = feats[n1]
            for j in range(i + 1, min(i + 1 + overlap, len(sorted_names))):
                n2 = sorted_names[j]
                kp2, desc2, laf2, hw2 = feats[n2]
                with torch.inference_mode():
                    _, matches = lg(desc1, desc2, laf1, laf2, hw1=hw1, hw2=hw2)
                if matches.shape[0] < 15:
                    continue
                matches_np = matches.detach().cpu().numpy().astype(np.uint32)

                cls1 = cls2 = None
                if tagger is not None:
                    n_match_total_pre_perception += int(matches_np.shape[0])
                    cls1 = kp_classes[n1][matches_np[:, 0]]
                    cls2 = kp_classes[n2][matches_np[:, 1]]
                    # B2 — sky drop on either endpoint.
                    sky_mask = (cls1 == sky_id) | (cls2 == sky_id)
                    if sky_mask.any():
                        matches_np = matches_np[~sky_mask]
                        n_sky_dropped += int(sky_mask.sum())
                        if matches_np.shape[0] < 15:
                            continue
                        cls1 = kp_classes[n1][matches_np[:, 0]]
                        cls2 = kp_classes[n2][matches_np[:, 1]]
                    # B3 — candidate-dynamic flag (instrumentation only).
                    if dynamic_ids is not None and dynamic_ids.size:
                        dyn_mask = np.isin(cls1, dynamic_ids) | np.isin(cls2, dynamic_ids)
                        n_dynamic_flagged += int(dyn_mask.sum())

                if guide_poses is not None:
                    if n1 not in guide_poses or n2 not in guide_poses:
                        continue
                    R1, t1 = guide_poses[n1]
                    R2, t2 = guide_poses[n2]

                    # Sampson residual (px) per surviving match — shared by
                    # B4's per-instance test and the standard epi threshold.
                    sampson_px = _epipolar_sampson_residuals_px(
                        kp_norm_by_name[n1], kp_norm_by_name[n2], matches_np,
                        R1, t1, R2, t2, focal,
                    )

                    # B4 — per-instance geometric motion test on candidate-
                    # dynamic matches. Requires both guide poses (residuals)
                    # and perception cache (class+instance tags).
                    if cls1 is not None and dynamic_ids is not None and dynamic_ids.size:
                        inst1 = kp_instances[n1][matches_np[:, 0]]
                        inst2 = kp_instances[n2][matches_np[:, 1]]
                        b4_keep, n_mov, n_stat, n_drop = _b4_drop_moving_instances(
                            cls1, inst1, cls2, inst2, sampson_px, dynamic_ids,
                        )
                        n_b4_moving_instances += n_mov
                        n_b4_static_instances += n_stat
                        n_b4_dropped += n_drop
                        if n_drop:
                            matches_np = matches_np[b4_keep]
                            sampson_px = sampson_px[b4_keep]
                            cls1 = cls1[b4_keep]
                            cls2 = cls2[b4_keep]
                            if matches_np.shape[0] < 15:
                                continue

                    n_pre = int(matches_np.shape[0])
                    epi_keep = sampson_px < epi_threshold_px
                    matches_np = matches_np[epi_keep]
                    n_epi_dropped += n_pre - int(matches_np.shape[0])
                    if matches_np.shape[0] < 15:
                        continue

                    p1n = kp_norm_by_name[n1][matches_np[:, 0]]
                    p2n = kp_norm_by_name[n2][matches_np[:, 1]]
                    n_pre = int(matches_np.shape[0])
                    tri_keep = _triangulation_angle_mask(
                        p1n, p2n, R1, t1, R2, t2, min_tri_angle_deg,
                    )
                    matches_np = matches_np[tri_keep]
                    n_tri_dropped += n_pre - int(matches_np.shape[0])
                    if matches_np.shape[0] < 15:
                        continue

                tvg = pycolmap.estimate_two_view_geometry(
                    cam,
                    kp1.astype(np.float64),
                    cam,
                    kp2.astype(np.float64),
                    matches=matches_np,
                )
                if tvg.inlier_matches.shape[0] < 15:
                    continue
                db.write_matches(name_to_id[n1], name_to_id[n2], matches_np)
                db.write_two_view_geometry(name_to_id[n1], name_to_id[n2], tvg)
                n_pairs += 1
                n_inliers_total += int(tvg.inlier_matches.shape[0])
        msg = (f"[densify] DISK+LG: {n_pairs} verified pairs, "
               f"{n_inliers_total} total inlier matches")
        if guide_poses is not None:
            msg += (f" (epi dropped {n_epi_dropped}, "
                    f"tri-angle dropped {n_tri_dropped} pre-RANSAC)")
        if tagger is not None:
            total = max(n_match_total_pre_perception, 1)
            msg += (
                f" (perception pre-filter: {n_match_total_pre_perception} matches; "
                f"sky dropped {n_sky_dropped} = {100 * n_sky_dropped / total:.2f}%; "
                f"candidate-dynamic flagged {n_dynamic_flagged} = "
                f"{100 * n_dynamic_flagged / total:.2f}%)"
            )
            if guide_poses is not None:
                msg += (
                    f" (B4 motion test: {n_b4_moving_instances} moving / "
                    f"{n_b4_static_instances} static instance-keys; "
                    f"dropped {n_b4_dropped} matches = "
                    f"{100 * n_b4_dropped / total:.2f}%)"
                )
        print(msg)
    finally:
        db.close()


def _gps_anchored_bundle_adjustment(
    rec: pycolmap.Reconstruction,
    gps_priors: dict[str, tuple[np.ndarray, float]],
    pin_every: int = 10,
) -> None:
    """BA with every Nth camera locked to its GPS position.

    pycolmap's dedicated pose-prior BA fits an internal Sim(3) alignment via
    RANSAC; for forward-only dashcam motion the camera centers are nearly
    collinear and the Sim(3) is under-constrained, so the alignment fails
    and the priors are silently discarded. The pragmatic alternative is to
    override every Nth image's translation with its GPS position and mark
    those poses constant in the BA config — the result is a soft GPS
    constraint (anchored every ``pin_every`` frames) plus a regular reprojection
    BA over the rest of the poses, points, and intrinsics.
    """
    img_pairs = sorted(rec.images.items(), key=lambda kv: kv[1].name)

    pinned = 0
    for k, (img_id, img) in enumerate(img_pairs):
        if k % pin_every != 0:
            continue
        if img.name not in gps_priors:
            continue
        xyz, _ = gps_priors[img.name]
        pose = img.cam_from_world()
        R = pose.rotation.matrix()
        new_t = -R @ np.asarray(xyz, dtype=np.float64)
        img.frame.set_cam_from_world(
            img.camera_id,
            pycolmap.Rigid3d(rotation=pose.rotation, translation=new_t),
        )
        pinned += 1

    config = pycolmap.BundleAdjustmentConfig()
    for img_id, img in rec.images.items():
        config.add_image(img_id)
    for k, (img_id, _) in enumerate(img_pairs):
        if k % pin_every == 0:
            config.set_constant_rig_from_world_pose(img_id)
    for p3d_id in rec.points3D:
        config.add_variable_point(p3d_id)

    # Diagnostic: confirm the mutated pinned poses really sit on top of their GPS targets.
    pinned_err_pre = []
    for k, (img_id, img) in enumerate(img_pairs):
        if k % pin_every != 0 or img.name not in gps_priors:
            continue
        pose = img.cam_from_world()
        c = -pose.rotation.matrix().T @ pose.translation
        pinned_err_pre.append(float(np.linalg.norm(c - gps_priors[img.name][0])))
    print(
        f"[densify] GPS-anchored BA: pinned {pinned} frames (every {pin_every}th); "
        f"pre-BA pinned residual mean = {np.mean(pinned_err_pre):.3f} m"
    )

    ba_opts = pycolmap.BundleAdjustmentOptions()
    ba_opts.ceres.solver_options.minimizer_progress_to_stdout = False
    ba_opts.print_summary = False
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(ba_opts, config, rec)
    adjuster.solve()

    pinned_err_post = []
    for k, (img_id, img) in enumerate(img_pairs):
        if k % pin_every != 0 or img.name not in gps_priors:
            continue
        pose = img.cam_from_world()
        c = -pose.rotation.matrix().T @ pose.translation
        pinned_err_post.append(float(np.linalg.norm(c - gps_priors[img.name][0])))
    print(f"[densify] post-BA pinned residual mean = {np.mean(pinned_err_post):.3f} m")


def colmap_densify(
    image_dir: Path,
    workspace: Path,
    scene: Scene,
    overlap: int = 20,
    matcher: str = "sift",
    gps_priors: dict[str, tuple[np.ndarray, float]] | None = None,
    gps_pin_every: int = 0,
    guide_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    epi_threshold_px: float = 4.0,
    min_tri_angle_deg: float = 1.0,
    fix_first_pose: bool = False,
    perception_cache: Path | None = None,
) -> pycolmap.Reconstruction:
    """Run feature extract+match, write VO poses as text recon, triangulate, BA.

    `matcher` selects the feature/matcher backend:
        - "sift": COLMAP's built-in SIFT pipeline (CPU).
        - "disk": DISK + LightGlue on CUDA.

    GPS is already applied as a *soft global constraint* upstream of this
    function by Umeyama-anchoring the VO trajectory to GPS. If
    ``gps_pin_every > 0`` and ``gps_priors`` are supplied, the final BA
    additionally pins every Nth camera to its GPS position. That's only
    worth doing when GPS accuracy is better than the post-BA-without-pin
    residual; on noisy consumer GPS (~few-metre accuracy) pinning forces the
    trajectory to fit noise and degrades local smoothness, so leave it off
    unless you have RTK-grade GPS.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "database.db"
    sparse_init = workspace / "sparse_init"
    sparse_out = workspace / "sparse"

    for p in (db_path, sparse_init, sparse_out):
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    sparse_out.mkdir(parents=True)

    if matcher == "sift":
        if guide_poses is not None:
            raise ValueError("guide_poses only supported with matcher='disk'")
        if perception_cache is not None:
            raise ValueError("perception_cache only supported with matcher='disk'")
        _sift_extract_match(image_dir, db_path, overlap)
    elif matcher == "disk":
        _disk_lightglue_match(
            image_dir, db_path, overlap,
            guide_poses=guide_poses, epi_threshold_px=epi_threshold_px,
            min_tri_angle_deg=min_tri_angle_deg,
            perception_cache=perception_cache,
        )
    else:
        raise ValueError(f"unknown matcher: {matcher!r} (expected 'sift' or 'disk')")

    print("[densify] reading database camera + image table…")
    db = pycolmap.Database.open(str(db_path))
    try:
        cameras = db.read_all_cameras()
        images_db = db.read_all_images()
        name_to_image_id = {img.name: img.image_id for img in images_db}
    finally:
        db.close()
    print(
        f"[densify] db has {len(cameras)} camera(s), {len(images_db)} images, "
        f"focal = {cameras[0].params[0]:.1f} px"
    )

    print("[densify] writing text recon with VO poses…")
    _write_text_reconstruction(sparse_init, scene, cameras, name_to_image_id)

    rec = pycolmap.Reconstruction()
    rec.read_text(str(sparse_init))
    print(f"[densify] loaded recon: {rec.num_reg_images()} registered images")

    print("[densify] triangulating from VO poses…")
    rec = pycolmap.triangulate_points(
        reconstruction=rec,
        database_path=str(db_path),
        image_path=str(image_dir),
        output_path=str(sparse_out),
        refine_intrinsics=True,
    )
    print(f"[densify] after triangulation: {rec.num_points3D()} 3D points")

    if rec.num_points3D() > 0:
        if gps_priors and gps_pin_every > 0:
            _gps_anchored_bundle_adjustment(rec, gps_priors, pin_every=gps_pin_every)
        elif fix_first_pose:
            # Gauge fix: pin the first image (by name order). Without GPS this
            # is what keeps consecutive refinement runs comparable and kills
            # one Sim(3) degree of freedom from the BA cost surface, which
            # also tends to reduce per-step pose wobble.
            img_pairs = sorted(rec.images.items(), key=lambda kv: kv[1].name)
            first_image_id = img_pairs[0][0]
            config = pycolmap.BundleAdjustmentConfig()
            for img_id in rec.images:
                config.add_image(img_id)
            config.set_constant_rig_from_world_pose(first_image_id)
            for p3d_id in rec.points3D:
                config.add_variable_point(p3d_id)
            print(
                f"[densify] global BA, first image pinned "
                f"({img_pairs[0][1].name})…"
            )
            ba_opts = pycolmap.BundleAdjustmentOptions()
            ba_opts.ceres.solver_options.minimizer_progress_to_stdout = False
            ba_opts.print_summary = False
            adjuster = pycolmap.create_default_ceres_bundle_adjuster(
                ba_opts, config, rec
            )
            adjuster.solve()
        else:
            print("[densify] global bundle adjustment (unconstrained)…")
            ba_opts = pycolmap.BundleAdjustmentOptions()
            ba_opts.ceres.solver_options.minimizer_progress_to_stdout = False
            ba_opts.print_summary = False
            pycolmap.bundle_adjustment(rec, ba_opts)
        # triangulate_points wrote the pre-BA reconstruction to sparse_out;
        # overwrite with the BA-refined poses + points so downstream tools
        # (pose export, AR overlay, dense MVS) see the optimised state.
        rec.write(str(sparse_out))
        print(f"[densify] after BA: {rec.num_points3D()} 3D points (written to {sparse_out})")

    return rec
