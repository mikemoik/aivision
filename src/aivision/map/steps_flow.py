"""Flow step (dense optical flow cache) — preprocessing for the dynamic role.

Lives outside ``mapgen.py`` so importing the framework never drags in cv2 / the
SEA-RAFT (torch) toolchain; the heavy imports stay inside the builder, which is
only called from :meth:`FlowStep.process`. Registered as ``(flow, sea-raft)``.

Optical flow is its own role because more than one consumer wants it: the
``dynamic`` motion-segmentation step (forward + backward flow drive the
motion-disparity residuals and the temporal vote), and potential future steps
(far-field rotation prior, flow-guided temporal label fusion). Computing it once
and caching keeps those cheap — RAFT inference is the expensive part.

Frames are read from ``image_0`` (gray-left, replicated to BGR) so the flow lives
in the same frame as the SLAM pose and the stereo rho — exactly the consistency
the motion-disparity geometry needs. For each consecutive pair ``(t, t+1)`` both
directions are computed:

  * ``fwd`` : flow t → t+1   (the observed flow ``f_obs``)
  * ``bwd`` : flow t+1 → t   (``f_ba``, for the forward/backward occlusion gate)

Out contract (``map-gen/flow/out/``):
  * ``flow_NNNNNN.npz``  — one per source frame ``t`` (pairs 0 … N-2), each holding
                           ``fwd`` and ``bwd`` as (H, W, 2) float16 (dx, dy) px
  * ``manifest.json``    — {n_frames, n_pairs, h, w, camera, backend, n_iters,
                           dtype, direction encoding}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mapgen import MapGenContext, MapGenStep, register_step

_DTYPES = {"float16": np.float16, "float32": np.float32}


def build_flow_cache(
    image_paths: list[Path],
    out_dir: Path,
    *,
    backend: str = "sea_raft",
    n_iters: int | None = None,
    dtype: str = "float16",
    camera: str = "image_0",
    log=None,
) -> dict:
    """Compute forward+backward dense flow for every consecutive pair, cache as npz.

    ``image_paths`` are read as grayscale and replicated to 3-channel BGR (the
    flow models expect 3 channels; image_0 is gray). Returns the manifest dict.
    """
    import cv2

    from ..flow import compute_flow_pair

    def _say(msg: str) -> None:
        (log.info if log is not None else print)(msg)

    np_dtype = _DTYPES[dtype]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    if n < 2:
        raise ValueError(f"need ≥2 frames for flow, got {n}")

    def _bgr(p: Path) -> np.ndarray:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"could not read frame: {p}")
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    h, w = _bgr(image_paths[0]).shape[:2]
    _say(f"[flow] backend={backend} camera={camera} {n} frames {w}x{h} "
         f"→ {n - 1} pairs (fwd+bwd, {dtype})")

    a = _bgr(image_paths[0])
    mags = []
    for t in range(n - 1):
        b = _bgr(image_paths[t + 1])
        fwd = compute_flow_pair(a, b, backend=backend, n_iters=n_iters)
        bwd = compute_flow_pair(b, a, backend=backend, n_iters=n_iters)
        np.savez_compressed(out_dir / f"flow_{t:06d}.npz",
                            fwd=fwd.astype(np_dtype), bwd=bwd.astype(np_dtype))
        mags.append(float(np.linalg.norm(fwd, axis=-1).mean()))
        if t % 50 == 0:
            _say(f"[flow] pair {t}/{n - 1}  |fwd|mean={mags[-1]:.2f}px")
        a = b

    manifest = {
        "n_frames": n,
        "n_pairs": n - 1,
        "h": h,
        "w": w,
        "camera": camera,
        "backend": backend,
        "n_iters": n_iters,
        "dtype": dtype,
        "keys": {"fwd": "flow t -> t+1 (f_obs)", "bwd": "flow t+1 -> t (f_ba)"},
        "index": "flow_{t:06d}.npz holds the pair (t, t+1); t in 0..n_pairs-1",
        "mean_fwd_mag_px": round(float(np.mean(mags)), 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _say(f"[flow] wrote {n - 1} npz + manifest (mean |fwd|={manifest['mean_fwd_mag_px']}px)")
    return manifest


@register_step
class FlowStep(MapGenStep):
    """Dense optical-flow cache (SEA-RAFT) in the image_0 / SLAM-pose frame.

    Consumes only the odometry image slice; emits a per-pair forward+backward
    flow cache that the ``dynamic`` step (and future flow consumers) read.
    """

    role = "flow"
    type = "sea-raft"
    inputs = ["input:odometry"]

    def process(self, ctx: MapGenContext) -> list[str]:
        from ..kitti import load_sequence

        seq_dir = ctx.input("odometry")
        camera = self.params.get("camera", "image_0")
        seq = load_sequence(seq_dir, camera=camera)
        ctx.log.info("flow: seq=%s camera=%s n=%d", seq_dir, camera, seq.n_frames)

        manifest = build_flow_cache(
            seq.image_paths, ctx.out("flow"),
            backend=self.params.get("backend", "sea_raft"),
            n_iters=self.params.get("n_iters"),
            dtype=self.params.get("dtype", "float16"),
            camera=camera, log=ctx.log,
        )
        ctx.log.info("flow: %d pairs cached", manifest["n_pairs"])
        return [
            "map-gen/flow/out/manifest.json",
            f"map-gen/flow/out/flow_NNNNNN.npz ({manifest['n_pairs']} pairs)",
        ]
