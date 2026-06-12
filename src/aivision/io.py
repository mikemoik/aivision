"""Frame I/O: read images from a directory or sample a video at a fixed rate."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_image_dir(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, BGR frame) for each image in a directory, sorted by filename."""
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield f.stem, img


def iter_video(path: Path, step: int = 1) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, BGR frame) from a video, keeping every `step`-th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield f"frame_{kept:06d}", frame
                kept += 1
            idx += 1
    finally:
        cap.release()


def iter_frames(source: Path, step: int = 1) -> Iterator[tuple[str, np.ndarray]]:
    """Auto-dispatch: directory of images or a video file."""
    source = Path(source)
    if source.is_dir():
        yield from iter_image_dir(source)
    else:
        yield from iter_video(source, step=step)


def dump_video_frames(video: Path, out_dir: Path, step: int = 1, quality: int = 92) -> int:
    """Extract every Nth frame of a video to JPGs in `out_dir`. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, frame in iter_video(video, step=step):
        cv2.imwrite(
            str(out_dir / f"{name}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        n += 1
    return n
