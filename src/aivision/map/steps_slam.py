"""SLAM step (ORB-SLAM3 stereo) — T1 of the map-gen workflow.

Lives outside ``mapgen.py`` so importing the framework never drags in cv2 / the
ORB toolchain (the heavy imports stay inside :meth:`SlamStep.process`). Approach
**B** with a subprocess tail: ``process`` calls the in-process wrapper
``aivision.pocs.orb_slam3``, which shells out to the ``stereo_kitti`` binary, then
folds the raw ``CameraTrajectory.txt`` into the role's ``out/trajectory.json``
contract. Registered as ``(slam, ORB3)``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .mapgen import MapGenContext, MapGenStep, register_step

_TRAJ_CONVENTION = "Twc 3x4 row-major [R|t], camera-to-world, metric metres"


def _resolve_orb(ctx: MapGenContext, orb_root: str) -> tuple[Path, Path]:
    """Locate (stereo_kitti binary, ORBvoc.txt), preferring generator bindata.

    The generator declares ``bin/stereo_kitti`` in config ``bindata`` (url=null,
    built locally); if it hasn't been staged there we fall back to the build tree
    under ``orb_root`` and extract the vocab tarball on demand.
    """
    binary = ctx.bindata("bin", "stereo_kitti")
    if not binary.exists():
        binary = Path(orb_root) / "Examples" / "Stereo" / "stereo_kitti"
    if not binary.exists():
        raise FileNotFoundError(f"stereo_kitti binary not found in bindata or {orb_root}")

    vocab = ctx.bindata("bin", "ORBvoc.txt")
    if not vocab.exists():
        vocab = Path(orb_root) / "Vocabulary" / "ORBvoc.txt"
        tar = vocab.with_name("ORBvoc.txt.tar.gz")
        if not vocab.exists() and tar.exists():
            ctx.log.info("extracting ORB vocabulary %s", tar)
            subprocess.run(["tar", "-xzf", str(tar)], cwd=tar.parent, check=True)
    if not vocab.exists():
        raise FileNotFoundError(f"ORBvoc.txt not found in bindata or {orb_root}")
    return binary, vocab


@register_step
class SlamStep(MapGenStep):
    """Stereo ORB-SLAM3 on the odometry slice → camera-to-world trajectory.

    Non-deterministic run-to-run by design; the role contract (the trajectory)
    is what stays stable, not the method (see provenance.py).
    """

    role = "slam"
    type = "ORB3"
    inputs = ["input:odometry"]

    def process(self, ctx: MapGenContext) -> list[str]:
        from ..kitti import load_sequence
        from ..pocs.orb_slam3 import run_stereo_kitti, write_kitti_yaml

        seq_dir = ctx.input("odometry")
        orb_root = self.params.get("orb_root", "external/orb_slam3")
        binary, vocab = _resolve_orb(ctx, orb_root)
        ctx.log.info("slam: seq=%s binary=%s", seq_dir, binary)

        s = load_sequence(seq_dir, camera="image_0")
        work = ctx.work("slam")
        yaml = write_kitti_yaml(
            work / "orbslam_stereo.yaml", s.K, s.width, s.height,
            baseline_m=float(self.params.get("baseline_m", 0.5372)),
        )
        cam_traj = run_stereo_kitti(seq_dir, binary, vocab, yaml, work)

        rows = np.loadtxt(cam_traj)
        if rows.ndim != 2 or rows.shape[1] != 12:
            raise ValueError(f"unexpected trajectory shape {rows.shape} from {cam_traj}")
        if len(rows) != s.n_frames:
            ctx.log.warning("slam: %d poses for %d frames", len(rows), s.n_frames)
        (ctx.out("slam") / "trajectory.json").write_text(json.dumps({
            "frame_id": "M",
            "convention": _TRAJ_CONVENTION,
            "n_frames": len(rows),
            "poses": rows.tolist(),
        }, indent=2))
        return ["map-gen/slam/out/trajectory.json"]
