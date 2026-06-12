"""Render a COLMAP reconstruction as an orbiting flythrough video.

Pipeline: Reconstruction → Open3D geometry → offscreen renderer → PNG frames
→ ffmpeg encode to mp4.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm

from .vo import Scene


def reconstruction_to_geometry(
    rec: pycolmap.Reconstruction,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Extract the sparse point cloud (with colors) and camera centers."""
    if rec.num_points3D() == 0:
        raise RuntimeError("Reconstruction has no 3D points")

    xyz = np.zeros((rec.num_points3D(), 3), dtype=np.float64)
    rgb = np.zeros((rec.num_points3D(), 3), dtype=np.float64)
    for i, p in enumerate(rec.points3D.values()):
        xyz[i] = p.xyz
        rgb[i] = np.asarray(p.color, dtype=np.float64) / 255.0

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)

    posed = [img for img in rec.images.values() if img.has_pose]
    posed.sort(key=lambda im: im.name)  # chronological for trajectory line
    centers = np.array([img.projection_center() for img in posed], dtype=np.float64)
    return pc, centers


def scene_to_geometry(scene: Scene) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Convert a VO Scene into an Open3D point cloud + camera centers."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(scene.points_xyz)
    pc.colors = o3d.utility.Vector3dVector(scene.points_rgb)
    return pc, scene.camera_centers


def trajectory_lineset(centers: np.ndarray, color=(1.0, 0.8, 0.0)) -> o3d.geometry.LineSet:
    n = len(centers)
    lines = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def _orbit_lookat(
    center: np.ndarray, radius: float, angle: float, elevation: float
):
    """Camera eye on a tilted circle around `center`, looking at `center`.

    `elevation` is the height above `center` in scene units (negative because
    -Y is up in our camera convention).
    """
    eye = center + np.array(
        [radius * np.cos(angle), elevation, radius * np.sin(angle)]
    )
    return eye, center, np.array([0.0, -1.0, 0.0])


def render_orbit_video(
    source: pycolmap.Reconstruction | Scene,
    out_mp4: Path,
    width: int = 1280,
    height: int = 720,
    n_frames: int = 240,
    fps: int = 30,
    point_size: float = 4.0,
) -> Path:
    """Render an orbiting camera around the reconstructed point cloud.

    Accepts either a pycolmap.Reconstruction or a vo.Scene.
    """
    if isinstance(source, Scene):
        pc, centers = scene_to_geometry(source)
    else:
        pc, centers = reconstruction_to_geometry(source)
    traj = trajectory_lineset(centers)

    # Robust scene metrics — clip top/bottom 2% to ignore outliers.
    xyz = np.asarray(pc.points)
    lo, hi = np.percentile(xyz, [2, 98], axis=0)
    scene_center = 0.5 * (lo + hi)
    extent = np.linalg.norm(hi - lo)
    # For an elongated dashcam cloud, orbit at ~half the extent and tilt the
    # camera high so the trajectory is visible from a bird's-eye angle.
    radius = 0.5 * extent
    cam_height = -0.5 * extent  # -Y is up

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0.04, 0.05, 0.08, 1.0])

    mat_pc = o3d.visualization.rendering.MaterialRecord()
    mat_pc.shader = "defaultUnlit"
    mat_pc.point_size = point_size
    scene.add_geometry("points", pc, mat_pc)

    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0
    scene.add_geometry("trajectory", traj, mat_line)

    fov_deg = 60.0
    near, far = max(extent * 1e-3, 0.01), extent * 10.0

    frames_dir = out_mp4.parent / (out_mp4.stem + "_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    for i in tqdm(range(n_frames), desc="render"):
        angle = 2.0 * np.pi * i / n_frames
        eye, look, up = _orbit_lookat(scene_center, radius, angle, cam_height)
        renderer.setup_camera(fov_deg, look, eye, up)
        scene.camera.set_projection(
            fov_deg, width / height, near, far,
            o3d.visualization.rendering.Camera.FovType.Vertical,
        )
        img = renderer.render_to_image()
        o3d.io.write_image(str(frames_dir / f"f_{i:05d}.png"), img)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(frames_dir)
    return out_mp4
