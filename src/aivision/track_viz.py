"""Overlay per-frame keypoint observations + N-frame history tails on the
original video, for visually comparing feature/matcher backends."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from tqdm import tqdm


_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def _frame_index(name: str) -> int:
    m = _FRAME_RE.search(name)
    if not m:
        raise ValueError(f"unexpected image name (no frame_NNNNNN.jpg): {name}")
    return int(m.group(1))


def render_track_overlay(
    reconstruction_dir: Path,
    frames_dir: Path,
    out_mp4: Path,
    history: int = 5,
    fps: int = 30,
    point_radius: int = 2,
    color_head: tuple[int, int, int] = (0, 255, 255),  # BGR — yellow
    color_tail: tuple[int, int, int] = (0, 165, 255),  # BGR — orange
) -> Path:
    """For each frame, draw every track that observes it as a head dot plus a
    tail of line segments back through the previous `history` frames where the
    same 3D point was observed."""
    rec = pycolmap.Reconstruction(str(reconstruction_dir))

    # image_id → frame_idx; frame_idx → image_id; frame_idx → image_name
    img_frame: dict[int, int] = {
        img_id: _frame_index(img.name) for img_id, img in rec.images.items()
    }
    frame_to_image: dict[int, int] = {f: i for i, f in img_frame.items()}
    frame_to_name: dict[int, str] = {
        f: rec.images[i].name for i, f in img_frame.items()
    }

    # For each track (== point3D), collect {frame_idx → (x, y)}.
    # Index per-frame the (track_id, xy) observations for fast lookup.
    obs_per_frame: dict[int, list[tuple[int, np.ndarray]]] = {
        f: [] for f in frame_to_image
    }
    track_history: dict[int, dict[int, np.ndarray]] = {}
    for p3d_id, p3d in rec.points3D.items():
        per_track: dict[int, np.ndarray] = {}
        for el in p3d.track.elements:
            frame = img_frame[el.image_id]
            xy = np.asarray(rec.images[el.image_id].points2D[el.point2D_idx].xy)
            per_track[frame] = xy
            obs_per_frame[frame].append((p3d_id, xy))
        track_history[p3d_id] = per_track

    sorted_frames = sorted(frame_to_image.keys())
    sample = cv2.imread(str(frames_dir / frame_to_name[sorted_frames[0]]))
    h, w = sample.shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

    n_obs_total = 0
    for f in tqdm(sorted_frames, desc=f"render {out_mp4.name}"):
        img = cv2.imread(str(frames_dir / frame_to_name[f]))

        for track_id, xy in obs_per_frame[f]:
            x, y = int(round(float(xy[0]))), int(round(float(xy[1])))

            # Draw tail: line from frame f-1 back to f-history (only where observed).
            tail = track_history[track_id]
            prev_x, prev_y = x, y
            for k in range(1, history + 1):
                fk = f - k
                if fk not in tail:
                    continue
                tx, ty = tail[fk]
                tx, ty = int(round(float(tx))), int(round(float(ty)))
                # Fade tail by drawing later (older) segments thinner / dimmer
                fade = 1.0 - (k - 1) / float(history)
                col = tuple(int(round(c * fade)) for c in color_tail)
                cv2.line(img, (prev_x, prev_y), (tx, ty), col, 1, cv2.LINE_AA)
                prev_x, prev_y = tx, ty
            cv2.circle(img, (x, y), point_radius, color_head, -1, cv2.LINE_AA)
        n_obs_total += len(obs_per_frame[f])

        # HUD: frame number + per-frame observation count
        cv2.putText(
            img, f"{frame_to_name[f]}  obs={len(obs_per_frame[f])}",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(img)
    writer.release()

    print(
        f"[tracks] wrote {out_mp4} — {len(sorted_frames)} frames, "
        f"{rec.num_points3D()} tracks, {n_obs_total} total observations drawn"
    )
    return out_mp4
