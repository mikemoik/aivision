"""Composite the static-environment 'lift-and-return' cinematic and assemble the mp4.

INTENT & DESIGN: docs/tasks/static_env_blender_cinematic.md

Consumes three per-frame layers (produced by scripts/blender_static_env_render.py + the dataset):
    real  = data/kitti/04/image_2/{f:06d}.png        (original footage, frames 0..270)
    full  = /tmp/se_k2/f{f:04d}.png                  (full RGB render along the lift path)
    obj   = /tmp/se_o2/f{f:04d}.png                  (objects-only RGBA, blend frames only)

The SEQUENCE SPLIT (forward reveal, reverse on the way back) -- these frame boundaries are the
saved design. Markings+objects = arrows, white border curbs, signs, traffic lights, guideposts
(NOT the grey lane surface / ground / sky, which only appear in the 'full' layer):
    0-19    real footage
    20-29   fade IN markings + objects onto the real footage
    30-59   real footage + markings/objects
    60-79   cross-dissolve (real+objects) -> full 3D render
    80-214  full render: dashcam drive (80-110) then the lift excursion (110-215)
    215-229 reverse cross-dissolve: full -> (real+objects)   [camera back on the drive]
    230-239 fade OUT markings/objects -> real footage
    240-270 real footage (tail)
Output: outputs/static_env/blender/reveal_lift_return.mp4 (10 fps, = KITTI rate).
"""
from __future__ import annotations
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
REAL = REPO / "data/kitti/04/image_2"
FULL = Path("/tmp/se_k2"); OBJ = Path("/tmp/se_o2")
OUT_FRAMES = Path("/tmp/se_final")
OUT_MP4 = REPO / "outputs/static_env/blender/reveal_lift_return.mp4"
W, H, NF, FPS = 1226, 370, 271, 10

# saved sequence-split boundaries (frame indices)
S = dict(real_in=20, mark_in=30, hold=60, dissolve=80, full=215, rdissolve=230, mark_out=240)


def _real(f):
    im = cv2.imread(str(REAL / f"{f:06d}.png"))
    return None if im is None else (cv2.resize(im, (W, H)) if im.shape[:2] != (H, W) else im)


def _full(f):
    im = cv2.imread(str(FULL / f"f{f:04d}.png"))
    return cv2.resize(im, (W, H)) if im.shape[:2] != (H, W) else im


def _obj(f):
    o = cv2.imread(str(OBJ / f"f{f:04d}.png"), cv2.IMREAD_UNCHANGED)
    if o is None or o.shape[2] < 4:
        return None, None
    return o[:, :, :3].astype(np.float32), o[:, :, 3:4].astype(np.float32) / 255.0


def _over(real, f, a):
    """Alpha-composite the objects layer (fraction a) onto the real frame."""
    rgb, A = _obj(f)
    if rgb is None:
        return real.astype(np.float32)
    m = A * a
    return real.astype(np.float32) * (1 - m) + rgb * m


def main():
    OUT_FRAMES.mkdir(parents=True, exist_ok=True)
    for fn in OUT_FRAMES.glob("*.png"):
        fn.unlink()
    for f in range(NF):
        real = _real(f); full = _full(f)
        if f < S["real_in"]:
            out = real.astype(np.float32)
        elif f < S["mark_in"]:
            out = _over(real, f, (f - S["real_in"]) / (S["mark_in"] - S["real_in"]))
        elif f < S["hold"]:
            out = _over(real, f, 1.0)
        elif f < S["dissolve"]:
            base = _over(real, f, 1.0); a = (f - S["hold"]) / (S["dissolve"] - S["hold"])
            out = base * (1 - a) + full.astype(np.float32) * a
        elif f < S["full"]:
            out = full.astype(np.float32)
        elif f < S["rdissolve"]:
            base = _over(real, f, 1.0); a = (f - S["full"]) / (S["rdissolve"] - S["full"])
            out = full.astype(np.float32) * (1 - a) + base * a
        elif f < S["mark_out"]:
            out = _over(real, f, 1.0 - (f - S["rdissolve"]) / (S["mark_out"] - S["rdissolve"]))
        else:
            out = real.astype(np.float32)
        cv2.imwrite(str(OUT_FRAMES / f"f{f:04d}.png"), np.clip(out, 0, 255).astype(np.uint8))

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(OUT_FRAMES / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT_MP4)], check=True)
    print(f"wrote {OUT_MP4}")


if __name__ == "__main__":
    main()
