"""End-to-end pipelines composed from the building blocks in this package."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import typer

from .camera import Intrinsics
from .colmap_runner import run_sfm
from .features import detect_orb, match_mutual
from .gps import frame_timestamps_ms, load_track
from .io import dump_video_frames, iter_frames
from .render import render_orbit_video
from .sfm import estimate_two_view
from .viz import draw_matches, save_point_cloud_ply
from .triangulate import colmap_densify
from .vo import anchor_scene_to_gps, run_vo


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def _build_position_priors(
    video: Path, gps_path: Path, step: int
) -> dict[str, tuple[np.ndarray, float]]:
    """Map each extracted frame filename to (ENU position, accuracy_metres)."""
    track = load_track(gps_path)

    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    timestamps = frame_timestamps_ms(
        n_frames=n_frames, fps=fps, start_time_ms=track.start_time_ms, step=step
    )
    positions, accuracies = track.interp_at(timestamps)

    priors: dict[str, tuple[np.ndarray, float]] = {}
    for kept, (xyz, acc) in enumerate(zip(positions, accuracies)):
        priors[f"frame_{kept:06d}.jpg"] = (xyz, float(acc))
    return priors


def pose_guided_refine_pipeline(
    existing_recon_dir: Path,
    image_dir: Path,
    out_workspace: Path,
    overlap: int = 50,
    epi_threshold_px: float = 4.0,
    min_tri_angle_deg: float = 1.0,
    gps: Path | None = None,
    video: Path | None = None,
    step: int = 3,
    anchor_gps: bool = False,
    fix_first_pose: bool = True,
    perception_cache: Path | None = None,
) -> Path:
    """Second SfM pass: use existing recon's poses to drive richer matching,
    then re-triangulate + BA in a fresh workspace.

    Builds a `Scene` from the existing reconstruction, passes its poses to
    the DISK + LightGlue matcher (which pre-filters matches by epipolar
    distance under those poses), then triangulates against the same poses
    and runs an unconstrained BA on the new correspondence graph.

    Returns the output workspace path (containing sparse/ with the refined
    recon).
    """
    rec = pycolmap.Reconstruction(str(existing_recon_dir))
    typer.echo(f"[refine] guide recon: {rec.num_reg_images()} images, "
               f"{rec.num_points3D()} points")

    # Build Scene from the recon (names, R, t in image-name order).
    img_pairs = sorted(rec.images.items(), key=lambda kv: kv[1].name)
    image_names = [img.name.replace(".jpg", "") for _, img in img_pairs]
    cam_R = np.stack([img.cam_from_world().rotation.matrix() for _, img in img_pairs])
    cam_t = np.stack([
        np.asarray(img.cam_from_world().translation, dtype=np.float64)
        for _, img in img_pairs
    ])
    # Points are not used by colmap_densify (only image_names + cam_R + cam_t
    # via _write_text_reconstruction), so fill in placeholders.
    from .vo import Scene
    scene = Scene(
        points_xyz=np.zeros((0, 3)),
        points_rgb=np.zeros((0, 3)),
        cam_R=cam_R, cam_t=cam_t,
        image_names=image_names,
    )
    guide_poses = {
        img.name: (img.cam_from_world().rotation.matrix(),
                   np.asarray(img.cam_from_world().translation, dtype=np.float64))
        for _, img in img_pairs
    }

    # GPS re-anchoring is opt-in. Default off because on consumer 1 Hz / ~10 m
    # GPS the Umeyama anchor drags the recon away from a feature-consistent
    # gauge without delivering useful accuracy. Pass anchor_gps=True only when
    # you actually want the recon in world (ENU) metres.
    if anchor_gps and gps is None and video is not None:
        candidate = video.with_suffix(".json")
        if candidate.exists():
            gps = candidate
    if anchor_gps and gps is not None and video is not None:
        from .gps import load_track
        track = load_track(gps)
        timestamps = frame_timestamps_ms(
            n_frames=_video_frame_count(video),
            fps=_video_fps(video),
            start_time_ms=track.start_time_ms,
            step=step,
        )
        gps_positions, _ = track.interp_at(timestamps[: len(scene.camera_centers)])
        typer.echo(f"[refine] re-anchoring guide poses to {gps}")
        scene = anchor_scene_to_gps(scene, gps_positions)
        # Update guide_poses too — anchoring may have changed R, t.
        for i, name in enumerate(scene.image_names):
            guide_poses[name + ".jpg"] = (scene.cam_R[i], scene.cam_t[i])

    msg = (f"[refine] guided densify: overlap={overlap}, "
           f"epi<{epi_threshold_px:.1f} px, "
           f"tri-angle≥{min_tri_angle_deg:.2f}°")
    if perception_cache is not None:
        msg += f", perception_cache={perception_cache}"
    typer.echo(msg)
    refined = colmap_densify(
        image_dir=image_dir,
        workspace=out_workspace,
        scene=scene,
        overlap=overlap,
        matcher="disk",
        guide_poses=guide_poses,
        epi_threshold_px=epi_threshold_px,
        min_tri_angle_deg=min_tri_angle_deg,
        fix_first_pose=fix_first_pose,
        perception_cache=perception_cache,
    )
    typer.echo(f"[refine] refined recon: {refined.num_reg_images()} images, "
               f"{refined.num_points3D()} points → {out_workspace / 'sparse'}")
    return out_workspace


def two_frame_demo(source: Path, out_dir: Path, step: int = 5, hfov_deg: float = 60.0) -> None:
    """Take the first two sampled frames, match features, recover pose, triangulate."""
    frames = list(islice(iter_frames(source, step=step), 2))
    if len(frames) < 2:
        raise typer.BadParameter(f"Need at least 2 frames from {source}, got {len(frames)}")

    (name_a, img_a), (name_b, img_b) = frames
    h, w = img_a.shape[:2]
    intr = Intrinsics.guess_from_hfov(width=w, height=h, hfov_deg=hfov_deg)

    feats_a = detect_orb(img_a)
    feats_b = detect_orb(img_b)
    pairs = match_mutual(feats_a, feats_b)
    typer.echo(f"frames: {name_a} ↔ {name_b}  |  matches: {len(pairs)}")

    if len(pairs) < 8:
        raise typer.Exit(code=2)

    pts_a = feats_a.xy[pairs[:, 0]]
    pts_b = feats_b.xy[pairs[:, 1]]
    result = estimate_two_view(pts_a, pts_b, intr)
    if result is None:
        typer.echo("Pose estimation failed — try a larger frame gap or richer scene.")
        raise typer.Exit(code=3)

    typer.echo(f"inliers: {int(result.mask.sum())}  |  3D points: {len(result.points_3d)}")
    typer.echo("R =")
    typer.echo(result.R)
    typer.echo(f"t (unit) = {result.t.ravel()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    draw_matches(img_a, img_b, result.pts_a, result.pts_b, out_dir / "matches.png")
    save_point_cloud_ply(result.points_3d, out_dir / "points.ply")
    typer.echo(f"wrote {out_dir / 'matches.png'} and {out_dir / 'points.ply'}")


def pointcloud_video_pipeline(
    source: Path,
    out_dir: Path,
    backend: str = "vo",
    step: int = 1,
    overlap: int = 10,
    matcher: str = "sift",
    width: int = 1280,
    height: int = 720,
    orbit_frames: int = 240,
    fps: int = 30,
    skip_sfm: bool = False,
    gps: Path | None = None,
) -> Path:
    """Extract frames → SfM (COLMAP or VO) → render orbiting flythrough video.

    `backend` is "vo" (chained two-view VO, reliable for forward driving) or
    "colmap" (full incremental SfM with bundle adjustment, fragile on pure
    forward motion but produces higher-quality clouds when it works).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    workspace = out_dir / "sfm"
    sparse_dir = workspace / "sparse"
    out_mp4 = out_dir / "pointcloud.mp4"

    if source.is_file():
        typer.echo(f"[1/3] extracting frames from {source} (step={step})…")
        n = dump_video_frames(source, frames_dir, step=step)
        typer.echo(f"      {n} frames → {frames_dir}")
        image_dir = frames_dir
    else:
        image_dir = source

    if backend == "vo":
        typer.echo(f"[2/3] chained-VO triangulation (stride=1, {n if source.is_file() else 'N'} frames)…")
        scene = run_vo(image_dir=image_dir)
        typer.echo(
            f"      {len(scene.camera_centers)} cameras, "
            f"{len(scene.points_xyz)} 3D points"
        )

        # Auto-detect <video>.json sidecar and anchor VO to GPS.
        gps_path = gps
        if gps_path is None and source.is_file():
            candidate = source.with_suffix(".json")
            if candidate.exists():
                gps_path = candidate
        if gps_path is not None and source.is_file():
            from .gps import load_track
            track = load_track(gps_path)
            timestamps = frame_timestamps_ms(
                n_frames=int(_video_frame_count(source)),
                fps=_video_fps(source),
                start_time_ms=track.start_time_ms,
                step=step,
            )
            gps_positions, _ = track.interp_at(timestamps[: len(scene.camera_centers)])
            typer.echo(f"      anchoring VO trajectory to GPS ({gps_path})")
            scene = anchor_scene_to_gps(scene, gps_positions)

        render_source = scene
    elif backend == "colmap-from-vo":
        typer.echo(f"[2/3] hybrid: VO + GPS anchor → SIFT triangulation + BA…")
        scene = run_vo(image_dir=image_dir)
        typer.echo(
            f"      VO seed: {len(scene.camera_centers)} cameras, "
            f"{len(scene.points_xyz)} 3D points (ORB)"
        )

        # GPS sidecar autodetect + anchor.
        gps_path = gps
        if gps_path is None and source.is_file():
            candidate = source.with_suffix(".json")
            if candidate.exists():
                gps_path = candidate
        if gps_path is None or not source.is_file():
            raise typer.BadParameter(
                "--backend colmap-from-vo requires a GPS sidecar (auto-detect "
                "failed and --gps not given)"
            )
        from .gps import load_track
        track = load_track(gps_path)
        timestamps = frame_timestamps_ms(
            n_frames=_video_frame_count(source),
            fps=_video_fps(source),
            start_time_ms=track.start_time_ms,
            step=step,
        )
        gps_positions, gps_acc = track.interp_at(timestamps[: len(scene.camera_centers)])
        typer.echo(f"      anchoring VO trajectory to {gps_path}")
        scene = anchor_scene_to_gps(scene, gps_positions)

        # Build per-image GPS priors so densify-BA treats GPS as a soft
        # constraint (same metric frame as the anchored VO poses).
        gps_priors_map: dict[str, tuple[np.ndarray, float]] = {}
        for stem, xyz, acc in zip(scene.image_names, gps_positions, gps_acc):
            gps_priors_map[stem + ".jpg"] = (np.asarray(xyz, dtype=np.float64), float(acc))

        render_source = colmap_densify(
            image_dir=image_dir, workspace=workspace, scene=scene,
            overlap=overlap, matcher=matcher, gps_priors=gps_priors_map,
        )
        typer.echo(
            f"      densified: {render_source.num_reg_images()} images, "
            f"{render_source.num_points3D()} 3D points"
        )
    elif backend == "colmap":
        if skip_sfm:
            sub_dirs = (
                [p for p in sparse_dir.iterdir() if p.is_dir()]
                if sparse_dir.exists() else []
            )
            if not sub_dirs:
                raise typer.BadParameter(f"--skip-sfm but no reconstruction in {sparse_dir}")
            recs = [pycolmap.Reconstruction(str(p)) for p in sub_dirs]
            render_source = max(recs, key=lambda r: r.num_reg_images())
            typer.echo(
                f"[2/3] reused COLMAP reconstruction: "
                f"{render_source.num_reg_images()} images, "
                f"{render_source.num_points3D()} 3D points"
            )
        else:
            # Auto-detect <video>.json GPS sidecar if --gps not given.
            gps_path = gps
            if gps_path is None and source.is_file():
                candidate = source.with_suffix(".json")
                if candidate.exists():
                    gps_path = candidate

            priors = None
            if gps_path is not None:
                if not source.is_file():
                    raise typer.BadParameter(
                        "GPS priors require a video file source (not a frame dir)"
                    )
                typer.echo(f"      using GPS priors from {gps_path}")
                priors = _build_position_priors(source, gps_path, step=step)

            typer.echo(f"[2/3] COLMAP SfM (sequential overlap={overlap})…")
            render_source = run_sfm(
                image_dir=image_dir,
                workspace=workspace,
                overlap=overlap,
                position_priors=priors,
            )
            typer.echo(
                f"      registered {render_source.num_reg_images()} images, "
                f"{render_source.num_points3D()} 3D points"
            )
    else:
        raise typer.BadParameter(f"unknown backend: {backend!r} (expected 'vo' or 'colmap')")

    typer.echo(f"[3/3] rendering orbit video → {out_mp4}…")
    render_orbit_video(
        source=render_source, out_mp4=out_mp4,
        width=width, height=height, n_frames=orbit_frames, fps=fps,
    )
    typer.echo(f"done → {out_mp4}")
    return out_mp4
