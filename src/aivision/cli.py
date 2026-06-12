"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from . import __version__, model_cache
from .pipeline import pointcloud_video_pipeline, pose_guided_refine_pipeline, two_frame_demo
from .map_viz import render_gps_map, render_pose_map
from .poses import (
    export_poses,
    render_ar_overlay,
    render_camera_balls_overlay,
    render_camera_trail_overlay,
    render_checker_overlay,
    render_point_cloud_overlay,
)
from .custom_ba import (
    compare_reconstructions as compare_custom_ba_recons,
    run_custom_ba,
    run_pycolmap_ba,
)
from .depth_analysis import write_depth_report
from .depth_aware_ba import run_depth_aware_ba
from .flow import run_flow_poc, run_flow_video
from .perception import (
    CacheManifest,
    MANIFEST_NAME,
    build_perception_cache,
    histogram_recon_keypoints,
    id2label_from_manifest,
    render_keypoint_tag_overlay,
    render_perception_overlay,
)
from .pose_smooth import smooth_reconstruction
from .track_viz import render_track_overlay

app = typer.Typer(add_completion=False, help="aivision — SfM for automotive front cameras")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command("two-view")
def two_view(
    source: Path = typer.Argument(..., help="Image directory or video file"),
    out: Path = typer.Option(Path("outputs/two_view"), help="Output directory"),
    step: int = typer.Option(5, help="When SOURCE is a video, keep every Nth frame"),
    hfov: float = typer.Option(60.0, help="Camera horizontal FOV in degrees (placeholder)"),
) -> None:
    """Run the two-frame SfM demo on the first two sampled frames."""
    two_frame_demo(source=source, out_dir=out, step=step, hfov_deg=hfov)


@app.command("download-models")
def download_models() -> None:
    """Prefetch learned-feature models (DISK + LightGlue) into ./models/."""
    target = model_cache.models_dir()
    typer.echo(f"Cache root: {target}")

    import torch
    from kornia.feature import DISK, LightGlueMatcher

    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Device: {device}")

    typer.echo("→ DISK (depth checkpoint)…")
    DISK.from_pretrained("depth", device=torch.device(device)).eval()

    typer.echo("→ LightGlue matcher (DISK features)…")
    LightGlueMatcher("disk").to(device).eval()

    typer.echo(f"done — see {target}")


@app.command("pointcloud-video")
def pointcloud_video(
    source: Path = typer.Argument(..., help="Input video file"),
    out: Path = typer.Option(Path("outputs/pointcloud_video"), help="Output directory"),
    backend: str = typer.Option(
        "vo",
        help="'vo' = chained two-view (robust to pure forward motion); "
             "'colmap' = incremental SfM with bundle adjustment (fragile on dashcam); "
             "'colmap-from-vo' = VO+GPS seed poses → SIFT triangulation + BA (recommended when GPS is available)",
    ),
    step: int = typer.Option(
        3,
        help="Keep every Nth video frame (adjacent dashcam frames have too little parallax)",
    ),
    overlap: int = typer.Option(20, help="COLMAP sequential matcher overlap window"),
    matcher: str = typer.Option(
        "sift",
        help="Feature/matcher backend for --backend colmap-from-vo: "
             "'sift' (COLMAP CPU SIFT) or 'disk' (DISK + LightGlue on CUDA)",
    ),
    width: int = typer.Option(1280),
    height: int = typer.Option(720),
    orbit_frames: int = typer.Option(240, help="Frames in the output orbit video"),
    fps: int = typer.Option(30),
    skip_sfm: bool = typer.Option(False, help="Reuse existing COLMAP reconstruction"),
    gps: Path = typer.Option(
        None,
        help="GPS sidecar JSON (auto-detected as <video>.json if omitted). "
             "Only used by --backend colmap.",
    ),
) -> None:
    """Extract frames → SfM/VO → orbit video of the point cloud."""
    pointcloud_video_pipeline(
        source=source, out_dir=out, backend=backend, step=step, overlap=overlap,
        matcher=matcher, width=width, height=height, orbit_frames=orbit_frames,
        fps=fps, skip_sfm=skip_sfm, gps=gps,
    )


@app.command("gps-map")
def gps_map_cmd(
    gps_json: Path = typer.Argument(..., help="GPS sidecar JSON"),
    out: Path = typer.Option(Path("outputs/gps_map.png")),
) -> None:
    """Render GPS samples + accuracy circles on an OpenStreetMap basemap."""
    render_gps_map(gps_json=gps_json, out_png=out)


@app.command("pose-map")
def pose_map_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    gps_json: Path = typer.Argument(..., help="GPS sidecar JSON"),
    out: Path = typer.Option(Path("outputs/pose_map.png")),
) -> None:
    """Render reconstructed camera trajectory on an OSM basemap."""
    render_pose_map(reconstruction_dir=reconstruction, gps_json=gps_json, out_png=out)


@app.command("export-poses")
def export_poses_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    out: Path = typer.Option(Path("outputs/poses.json"), help="Output JSON path"),
) -> None:
    """Dump per-frame camera poses + intrinsics to JSON for AR use."""
    export_poses(reconstruction_dir=reconstruction, out_json=out)


@app.command("checker-overlay")
def checker_overlay_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/checker_overlay.mp4")),
    plane_z: float = typer.Option(3.5, help="Plane height in the first-camera frame, in metres (+z = up)"),
    cell: float = typer.Option(2.0, help="Checker cell size (m)"),
    x_min: float = typer.Option(-10.0, help="Forward-axis min (m)"),
    x_max: float = typer.Option(80.0, help="Forward-axis max (m)"),
    y_min: float = typer.Option(-15.0, help="Left-axis min (m)"),
    y_max: float = typer.Option(15.0, help="Left-axis max (m)"),
    fps: int = typer.Option(30),
) -> None:
    """Project a static translucent blue checkerboard onto every frame.

    The world frame is fixed by the first frame's camera pose:
    +x = forward, +y = left, +z = up. The checker lies in the plane
    z = ``plane_z`` and stays static while the camera moves through it."""
    render_checker_overlay(
        reconstruction_dir=reconstruction, frames_dir=frames, out_mp4=out,
        fps=fps, plane_z=plane_z, cell=cell,
        x_range=(x_min, x_max), y_range=(y_min, y_max),
    )


@app.command("camera-balls-overlay")
def camera_balls_overlay_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/camera_balls_overlay.mp4")),
    diameter: float = typer.Option(0.20, help="Ball diameter (m)"),
    near_clip: float = typer.Option(5.0, help="Skip balls closer than this (m)"),
    fps: int = typer.Option(30),
) -> None:
    """Render every reconstructed camera centre as a static yellow ball."""
    render_camera_balls_overlay(
        reconstruction_dir=reconstruction, frames_dir=frames, out_mp4=out,
        fps=fps, ball_diameter=diameter, near_clip=near_clip,
    )


@app.command("point-cloud-overlay")
def point_cloud_overlay_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/point_cloud_overlay.mp4")),
    fps: int = typer.Option(30),
    near_clip: float = typer.Option(1.0, help="Skip points closer than this (m)"),
    far_clip: float = typer.Option(300.0, help="Skip points farther than this (m)"),
    radius: int = typer.Option(2, help="Drawn dot radius (px)"),
    alpha: float = typer.Option(0.75, help="Overlay opacity (1.0 = opaque dots)"),
    colour: str = typer.Option("rgb", help="'rgb' = COLMAP per-point colour, 'depth' = TURBO on first-frame depth"),
    min_track_len: int = typer.Option(1, help="Only render points seen in ≥N images"),
    max_reproj_err: float = typer.Option(float("inf"), help="Only render points with reprojection error ≤ this (px)"),
    depth_band_lo: float = typer.Option(0.0, help="Only render points with median observed depth ≥ this (m)"),
    depth_band_hi: float = typer.Option(float("inf"), help="Only render points with median observed depth ≤ this (m)"),
) -> None:
    """Project every 3D point of the recon onto each frame."""
    band = None
    if depth_band_lo > 0 or depth_band_hi < float("inf"):
        band = (depth_band_lo, depth_band_hi)
    render_point_cloud_overlay(
        reconstruction_dir=reconstruction, frames_dir=frames, out_mp4=out,
        fps=fps, near_clip=near_clip, far_clip=far_clip,
        radius_px=radius, alpha=alpha, colour=colour,
        min_track_len=min_track_len, max_reproj_err=max_reproj_err,
        depth_band=band,
    )


@app.command("camera-trail")
def camera_trail_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/camera_trail.mp4")),
    fps: int = typer.Option(30),
    ground_offset: float = typer.Option(1.5, help="Drop the camera positions down by this much (m) to land on the road"),
    diameter: float = typer.Option(0.20, help="Trail marker diameter (m)"),
) -> None:
    """Project the camera positions of every frame onto the road surface,
    rendered onto every frame. Past frames blue, future green, current
    white. The trail should form a smooth recede-into-distance arc."""
    render_camera_trail_overlay(
        reconstruction_dir=reconstruction, frames_dir=frames, out_mp4=out,
        fps=fps, ground_offset_m=ground_offset, radius_world_m=diameter * 0.5,
    )


@app.command("ar-overlay")
def ar_overlay_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/ar_overlay.mp4")),
    fps: int = typer.Option(30),
    grid_extent: float = typer.Option(40.0, help="Half-extent of the ground grid (m)"),
    grid_cell: float = typer.Option(2.0, help="Ground grid cell size (m)"),
) -> None:
    """Render an AR overlay: ground grid + 3D axes drawn via the extracted poses."""
    render_ar_overlay(
        reconstruction_dir=reconstruction, frames_dir=frames, out_mp4=out,
        fps=fps, grid_extent=grid_extent, grid_cell=grid_cell,
    )


@app.command("track-depth-analysis")
def track_depth_analysis_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    out: Path = typer.Option(Path("outputs/depth_analysis"), help="Output directory"),
) -> None:
    """Compute per-track median depth + inverse depth distribution."""
    write_depth_report(reconstruction_dir=reconstruction, out_dir=out)


@app.command("pose-guided-refine")
def pose_guided_refine_cmd(
    reconstruction: Path = typer.Argument(..., help="Existing COLMAP sparse reconstruction dir (poses used as guide)"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/pose_guided_refine"), help="Output workspace"),
    overlap: int = typer.Option(50, help="Sequential matching window (widened vs default 20)"),
    epi_threshold: float = typer.Option(4.0, help="Sampson epipolar distance threshold (px)"),
    min_tri_angle: float = typer.Option(1.0, help="Minimum triangulation angle (deg) for guided matches"),
    gps: Path = typer.Option(None, help="GPS sidecar JSON (auto-detected from --video if omitted; only used with --anchor-gps)"),
    video: Path = typer.Option(None, help="Source video (for frame timestamps when re-anchoring to GPS)"),
    step: int = typer.Option(3, help="Frame step used originally (for GPS timestamp alignment)"),
    anchor_gps: bool = typer.Option(False, "--anchor-gps/--no-anchor-gps", help="Re-Umeyama recon onto GPS positions (off by default; consumer GPS is too noisy to anchor)"),
    fix_first_pose: bool = typer.Option(True, "--fix-first-pose/--no-fix-first-pose", help="Hold the first image's pose constant during BA (gauge fix; default on)"),
    perception_cache: Path = typer.Option(
        None,
        "--perception-cache",
        help="Phase-A perception cache (semantic + depth) to enable B2 sky-drop "
             "and B3 candidate-dynamic flagging at the DISK+LightGlue stage",
    ),
) -> None:
    """Run a second SfM pass guided by an existing recon's poses.

    Uses the existing poses to epipolar-filter DISK+LightGlue matches over a
    widened sequential window, then re-triangulates + BA in a fresh workspace.
    """
    pose_guided_refine_pipeline(
        existing_recon_dir=reconstruction,
        image_dir=frames,
        out_workspace=out,
        overlap=overlap,
        epi_threshold_px=epi_threshold,
        min_tri_angle_deg=min_tri_angle,
        gps=gps,
        video=video,
        step=step,
        anchor_gps=anchor_gps,
        fix_first_pose=fix_first_pose,
        perception_cache=perception_cache,
    )


@app.command("depth-aware-ba")
def depth_aware_ba_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    database: Path = typer.Argument(..., help="COLMAP database.db"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/depth_aware_ba"), help="Output dir"),
    far_depth: float = typer.Option(50.0, help="Tracks with median depth >= this (m) are excluded from BA cost"),
    no_retriangulate: bool = typer.Option(False, help="Skip re-triangulation after BA (faster, but cloud is stale)"),
) -> None:
    """Run near-only BA on a recon: drop far tracks from the cost,
    refine poses + intrinsics + near points, then re-triangulate full cloud."""
    run_depth_aware_ba(
        reconstruction_dir=reconstruction,
        database_path=database,
        image_dir=frames,
        out_dir=out,
        far_depth_m=far_depth,
        retriangulate_after=not no_retriangulate,
    )


@app.command("custom-ba")
def custom_ba_cmd(
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    out: Path = typer.Option(Path("outputs/custom_ba"), help="Output directory"),
    max_iter: int = typer.Option(500, help="Max LM outer iterations"),
    ftol: float = typer.Option(1e-10, help="Cost-reduction convergence tolerance"),
    compare_pycolmap: bool = typer.Option(
        True, help="Also run pycolmap BA on the same input and report a comparison"
    ),
    perception_cache: Path = typer.Option(
        None,
        help="Phase-A perception cache dir; when set the C2 mono-depth soft prior is added "
             "to the BA cost. First image's (s, b) held to (1, 0) as metric-scale anchor.",
    ),
    depth_alpha: float = typer.Option(
        0.0,
        help="C2 depth-prior weight α. 0 = auto-calibrate so depth cost ≈ "
             "--depth-target-share of reproj cost at LS-fit baseline.",
    ),
    depth_target_share: float = typer.Option(
        0.1, help="Target fraction of reproj cost the depth prior should carry"
    ),
    accel_weight: float = typer.Option(
        0.0,
        help="C3 acceleration-prior weight w. 0 = off (unless --accel-target-share>0). "
             "Residual is sqrt(w)·(c_{k+1} - 2 c_k + c_{k-1}) on cam centres.",
    ),
    accel_target_share: float = typer.Option(
        0.0,
        help="Auto-calibrate the C3 accel weight to this fraction of reproj cost at "
             "the warm-start state. 0 = no auto-calibration; together with --accel-weight=0 "
             "leaves the accel prior off.",
    ),
    step_priors: Path = typer.Option(
        None,
        help="C4: flow-video JSON with per-pair homographies (e.g. "
             "outputs/c4_flow_video.json). Per-pair metric step length is "
             "extracted via Song-Chandraker decomposition and used as a soft prior.",
    ),
    step_weight: float = typer.Option(
        0.0,
        help="C4 step-prior global weight w. 0 = off unless --step-target-share>0. "
             "Per-pair residual sqrt(w·conf_k)·(||Δc|| - s_obs).",
    ),
    step_target_share: float = typer.Option(
        0.0,
        help="Auto-calibrate the C4 step weight to this fraction of reproj cost at "
             "the (rescaled) warm-start state. 0 = off.",
    ),
    h_cam: float = typer.Option(
        1.4, help="Assumed camera height above road (m) for the C4 step prior"
    ),
    nonholonomic: bool = typer.Option(
        False, "--nonholonomic/--no-nonholonomic",
        help="C5: add a non-holonomic motion prior on consecutive image pairs. "
             "Residual zeroes the lateral (x) and vertical (y) components of "
             "R_a·(c_b - c_a), pinning per-pair steps to the camera z-axis.",
    ),
    nh_weight: float = typer.Option(
        0.0,
        help="C5 non-holonomic-prior weight w_nh. 0 = off unless "
             "--nh-target-share>0. Residual sqrt(w_nh)·δ_cam[0:2].",
    ),
    nh_target_share: float = typer.Option(
        0.0,
        help="Auto-calibrate the C5 nh weight to this fraction of reproj cost "
             "at the warm-start state. 0 = off.",
    ),
) -> None:
    """Custom Python bundle adjustment (Phase C1 + C2 + C3 + C4 + C5).

    Roll-your-own scipy.optimize.least_squares BA with an analytical sparse
    Jacobian. Refines poses + 3D points; first image's pose is held constant
    (gauge fix). Intrinsics are held constant — matches pycolmap if it's
    also run with refine_focal_length=False, refine_extra_params=False.

    With --perception-cache the C2 mono-depth soft prior is added to the
    BA cost (CodeSLAM/DeepFactors-style), with per-frame (s_i, b_i) scale+
    shift co-optimised. The first image's (s, b) is held at (1, 0) so the
    cloud is anchored to Depth Pro's metric scale on the gauge frame.

    With --accel-weight>0 or --accel-target-share>0 the C3 acceleration
    soft prior is added on consecutive camera centres. Either prior (or
    both) triggers a C1 warm-start before weight calibration.

    With --nonholonomic the C5 non-holonomic prior is added on consecutive
    image pairs (zeroes the lateral + vertical components of the
    cam-frame step). Stacks freely with C3/C4.
    """
    out.mkdir(parents=True, exist_ok=True)
    rec_cu, stats_cu = run_custom_ba(
        reconstruction,
        out / "custom_sparse",
        max_iter=max_iter,
        ftol=ftol,
        perception_cache=perception_cache,
        depth_alpha=depth_alpha,
        depth_target_share=depth_target_share,
        accel_weight=accel_weight,
        accel_target_share=accel_target_share,
        step_priors_json=step_priors,
        step_weight=step_weight,
        step_target_share=step_target_share,
        h_cam=h_cam,
        with_nonholonomic_prior=nonholonomic,
        nh_weight=nh_weight,
        nh_target_share=nh_target_share,
    )
    payload: dict = {"custom": stats_cu}
    if compare_pycolmap:
        rec_pc, stats_pc = run_pycolmap_ba(
            reconstruction, out / "pycolmap_sparse", fix_first=True
        )
        cmp = compare_custom_ba_recons(rec_pc, rec_cu, "pycolmap", "custom")
        payload["pycolmap"] = stats_pc
        payload["compare"] = cmp
        typer.echo(
            f"[custom-ba] custom mean reproj (final): "
            f"{rec_cu.compute_mean_reprojection_error():.6f} px"
        )
        typer.echo(
            f"[custom-ba] pycolmap mean reproj (final): "
            f"{rec_pc.compute_mean_reprojection_error():.6f} px"
        )
        typer.echo(
            f"[custom-ba] cam-centre mean Δ: {cmp['cam_centre_mean_m']*1000:.3f} mm "
            f"(max {cmp['cam_centre_max_m']*1000:.3f} mm); "
            f"rot mean Δ: {cmp['rot_mean_deg']*60:.3f} arcmin"
        )
    (out / "stats.json").write_text(json.dumps(payload, indent=2))


@app.command("b4-verdict-overlay")
def b4_verdict_overlay_cmd(
    frames: Path = typer.Argument(..., help="Frame directory (frame_NNNNNN.jpg)"),
    perception_cache: Path = typer.Argument(..., help="Phase-A perception cache (semantic + instance + depth)"),
    guide_recon: Path = typer.Argument(..., help="Guide reconstruction (cam_from_world poses for the pair)"),
    frame_a: int = typer.Argument(..., help="Index of first frame (0-based)"),
    frame_b: int = typer.Argument(..., help="Index of second frame (0-based)"),
    out: Path = typer.Option(Path("outputs/b4_verdict.png"), help="Output PNG"),
    n_keypoints: int = typer.Option(8192, help="DISK keypoint budget per frame"),
) -> None:
    """B4 diagnostic — render one frame pair with kept vs B4-dropped matches colour-coded.

    Static kept matches are drawn as grey dots; dynamic kept (passed B4) as
    green lines; B4-flagged moving as red lines; sky-drop as cyan dots.
    """
    from .b4_verdict import b4_verdict_pair, render_b4_verdict

    pv = b4_verdict_pair(
        frames_dir=frames,
        perception_cache=perception_cache,
        guide_recon=guide_recon,
        frame_a_idx=frame_a,
        frame_b_idx=frame_b,
        n_keypoints=n_keypoints,
    )
    out_path = render_b4_verdict(pv, out)
    typer.echo(
        f"[b4-verdict] {pv.name_a} -> {pv.name_b}: "
        f"matches={len(pv.matches)}, sky={(pv.verdict == 1).sum()}, "
        f"b4_dropped={(pv.verdict == 2).sum()}, "
        f"moving_inst={pv.n_b4_moving_instances}, static_inst={pv.n_b4_static_instances} "
        f"-> {out_path}"
    )


@app.command("pose-smooth")
def pose_smooth_cmd(
    reconstruction: Path = typer.Argument(..., help="Input COLMAP sparse reconstruction dir"),
    out: Path = typer.Argument(..., help="Output sparse dir (will be created/overwritten)"),
    window: int = typer.Option(11, help="Savitzky-Golay window length (must be odd)"),
    poly: int = typer.Option(3, help="Polynomial order of the local fit"),
) -> None:
    """Post-hoc Savitzky-Golay smoothing of camera centres (diagnostic).

    Rewrites camera translations from a smoothed centre curve while leaving
    rotations and 3D points untouched. Used to verify the recon's *mean*
    motion is correct and only per-step direction is noisy.
    """
    smooth_reconstruction(
        reconstruction_dir=reconstruction, out_dir=out,
        window=window, poly=poly,
    )


@app.command("perception-cache")
def perception_cache_cmd(
    source: Path = typer.Argument(..., help="Input video file or frames directory"),
    out: Path = typer.Argument(..., help="Cache output directory"),
    step: int = typer.Option(3, help="Keep every Nth video frame"),
    segment: bool = typer.Option(False, "--segment/--no-segment", help="Run panoptic seg (A2)"),
    depth: bool = typer.Option(False, "--depth/--no-depth", help="Run mono depth (A3)"),
) -> None:
    """Build the per-frame perception cache (Phase A scaffold).

    With no flags this only iterates frames and writes a manifest. A2 / A3
    will plug their model output into the same cache directory via
    ``--segment`` / ``--depth``.
    """
    build_perception_cache(
        source=source, out_dir=out, step=step,
        do_segment=segment, do_depth=depth,
    )


@app.command("perception-video")
def perception_video_cmd(
    cache: Path = typer.Argument(..., help="Perception cache directory (built by perception-cache)"),
    out: Path = typer.Option(Path("outputs/perception_cache.mp4")),
    fps: int = typer.Option(30),
    seg_alpha: float = typer.Option(0.45, help="Semantic-mask opacity when present"),
) -> None:
    """Render a video showing the perception cache: source frames + manifest
    banner + (where populated) semantic mask + depth heatmap inset.

    For an A1-only cache (no fields populated yet) the output is the source
    frames with a banner that documents the cache schedule. The same command
    lights up automatically as A2 (segmentation) and A3 (depth) land.
    """
    render_perception_overlay(cache_dir=cache, out_mp4=out, fps=fps, seg_alpha=seg_alpha)


@app.command("perception-tag-kps")
def perception_tag_kps_cmd(
    cache: Path = typer.Argument(..., help="Perception cache directory (built by perception-cache)"),
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    top: int = typer.Option(20, help="Number of classes to print"),
    out_json: Path = typer.Option(None, help="Optional JSON dump of the full per-class histogram"),
) -> None:
    """A4 smoke check: tag every 2D keypoint in *reconstruction* via the
    perception cache and print the class-distribution histogram.

    Sky should be a small fraction (matchers already filter most of it via
    texture); large `255 (unknown)` is a red flag that the segmenter has
    coverage holes the cache hasn't recorded.
    """
    counts, n_total = histogram_recon_keypoints(cache, reconstruction)
    manifest = CacheManifest.load(cache / MANIFEST_NAME)
    id2label = id2label_from_manifest(manifest)
    id2label.setdefault(255, "unknown")

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    typer.echo(
        f"[perception-tag-kps] {n_total} keypoint observations across "
        f"{manifest.n_frames} frames, {len(counts)} distinct classes"
    )
    for cid, c in ranked[:top]:
        name = id2label.get(cid, f"id_{cid}")
        pct = 100.0 * c / n_total if n_total else 0.0
        typer.echo(f"  {cid:4d}  {name:40s}  {c:>10d}  {pct:6.2f}%")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_total": n_total,
            "counts": {int(cid): int(c) for cid, c in counts.items()},
            "labels": {int(cid): id2label.get(int(cid), f"id_{cid}") for cid in counts},
            "seg_model": manifest.seg_model,
            "depth_model": manifest.depth_model,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        typer.echo(f"[perception-tag-kps] wrote {out_json}")


@app.command("perception-tag-kps-video")
def perception_tag_kps_video_cmd(
    cache: Path = typer.Argument(..., help="Perception cache directory"),
    reconstruction: Path = typer.Argument(..., help="COLMAP sparse reconstruction dir"),
    frames: Path = typer.Argument(..., help="Frame directory used by reconstruction"),
    out: Path = typer.Option(Path("outputs/a4_tag_kps.mp4")),
    fps: int = typer.Option(30),
    radius: int = typer.Option(3, help="Keypoint dot radius (px)"),
    colour: str = typer.Option("class", help="'class' = palette by tagged class id; 'depth' = turbo by tagged mono depth"),
) -> None:
    """A4 visualization: draw every recon keypoint coloured by its tagged
    class (or by tagged mono depth) over the source frames.

    The class palette matches `perception-video`'s semantic-mask palette, so
    eyeballing the two side-by-side confirms the tagger is indexing the
    cache correctly.
    """
    render_keypoint_tag_overlay(
        cache_dir=cache, reconstruction_dir=reconstruction, frames_dir=frames,
        out_mp4=out, fps=fps, radius_px=radius, colour=colour,
    )


@app.command("flow-poc")
def flow_poc_cmd(
    cache: Path = typer.Argument(..., help="Perception cache (must have semantic populated)"),
    frames: Path = typer.Argument(..., help="Directory of frame_NNNNNN.jpg matching the cache"),
    out: Path = typer.Option(Path("outputs/c4_flow_poc"), help="Output directory"),
    pairs: str = typer.Option(
        "",
        help="Comma-separated source-frame indices (e.g. '0,49,98'); empty → uniform sampling",
    ),
    n_pairs: int = typer.Option(3, help="Number of evenly-spaced pairs when --pairs is empty"),
    grid_stride: int = typer.Option(12, help="Sampling stride (px) for the candidate grid"),
    ransac_thresh: float = typer.Option(3.0, help="RANSAC reprojection threshold (px)"),
    arrow_stride: int = typer.Option(4, help="Sub-sample arrows for the bottom panels"),
    backend: str = typer.Option("raft_large", help="Flow model backend"),
) -> None:
    """Round-3 C4 POC: dense flow → road mask → per-pair homography.

    Writes one PNG panel per pair (frame_a + road mask | flow magnitude |
    masked flow vectors | RANSAC inliers/outliers) and a summary JSON
    with inlier counts and Sampson residuals — the green-light gate for
    BA integration is >500 inliers/pair with low residual.
    """
    pair_list = [int(x) for x in pairs.split(",") if x.strip()] or None
    run_flow_poc(
        perception_cache=cache,
        frames_dir=frames,
        out_dir=out,
        pair_indices=pair_list,
        n_pairs=n_pairs,
        grid_stride=grid_stride,
        ransac_thresh=ransac_thresh,
        arrow_stride=arrow_stride,
        backend=backend,
    )


@app.command("flow-video")
def flow_video_cmd(
    cache: Path = typer.Argument(..., help="Perception cache (must have semantic populated)"),
    frames: Path = typer.Argument(..., help="Directory of frame_NNNNNN.jpg matching the cache"),
    out: Path = typer.Option(Path("outputs/c4_flow_video.mp4"), help="Output mp4"),
    fps: int = typer.Option(10, help="Playback fps (n_pairs/fps = video duration)"),
    grid_stride: int = typer.Option(12, help="Sampling stride (px) for the candidate grid"),
    ransac_thresh: float = typer.Option(3.0, help="RANSAC reprojection threshold (px)"),
    arrow_stride: int = typer.Option(6, help="Sub-sample arrows in the overlay"),
    backend: str = typer.Option("sea_raft", help="Flow model backend"),
) -> None:
    """Run SEA-RAFT on every consecutive frame pair, mask to road, fit
    homography, and write a single mp4 visualizing flow + road mask +
    RANSAC inliers + per-pair stats HUD.

    Companion JSON beside the mp4 stores per-pair (inliers, sampson,
    homography). Default backend is SEA-RAFT (spring-M from HF); pass
    --backend raft_large for the torchvision fallback.
    """
    run_flow_video(
        perception_cache=cache,
        frames_dir=frames,
        out_mp4=out,
        fps=fps,
        grid_stride=grid_stride,
        ransac_thresh=ransac_thresh,
        arrow_stride=arrow_stride,
        backend=backend,
    )


@app.command("tracks-video")
def tracks_video(
    reconstruction: Path = typer.Argument(..., help="Path to a COLMAP sparse reconstruction (cameras/images/points3D)"),
    frames: Path = typer.Argument(..., help="Directory of frame_NNNNNN.jpg images used by the reconstruction"),
    out: Path = typer.Option(Path("outputs/tracks.mp4"), help="Output MP4"),
    history: int = typer.Option(5, help="How many previous frames to trail per track"),
    fps: int = typer.Option(30),
) -> None:
    """Render an overlay video: per-frame keypoint observations + history tails."""
    render_track_overlay(
        reconstruction_dir=reconstruction, frames_dir=frames,
        out_mp4=out, history=history, fps=fps,
    )


if __name__ == "__main__":
    app()
