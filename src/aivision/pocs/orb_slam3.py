"""ORB-SLAM3 stereo wrapper for KITTI seq 04 — POC1 Phase A.

Two responsibilities:
    write_kitti_yaml  - synthesise the OpenCV-FileStorage YAML config that
                        ORB-SLAM3 demands (intrinsics + baseline + tuning).
    run_stereo_kitti  - subprocess-invoke the `stereo_kitti` example binary
                        the upstream build produces under
                        external/orb_slam3/Examples/Stereo/.

Trajectory is parsed by `common.load_tum_trajectory` from the
`CameraTrajectory.txt` file ORB-SLAM3 writes into its CWD.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np


# ORB-SLAM3 ships KITTI04-12.yaml with these ORB defaults — we keep them so
# our YAML is a drop-in replacement (only intrinsics/baseline change).
_DEFAULT_ORB = dict(
    n_features=2000,
    scale_factor=1.2,
    n_levels=8,
    ini_th_fast=20,
    min_th_fast=7,
)


def write_kitti_yaml(
    out_path: Path,
    K: np.ndarray,
    width: int,
    height: int,
    baseline_m: float,
    fps: float = 10.0,
    orb: dict | None = None,
) -> Path:
    """Generate an ORB-SLAM3 stereo YAML config from rectified KITTI intrinsics.

    `baseline_m` is the *physical* stereo baseline in metres; ORB-SLAM3 wants
    `Camera.bf` = baseline * fx (i.e. the right cam's translation in px-units),
    not the metre value. For KITTI seq 04 this is 379.81 (~= 0.537 m * 707.09).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    bf = baseline_m * fx
    orb = {**_DEFAULT_ORB, **(orb or {})}

    body = textwrap.dedent(
        f"""\
        %YAML:1.0

        #--------------------------------------------------------------------------------------------
        # Camera Parameters. Rectified stereo (KITTI grayscale).
        #--------------------------------------------------------------------------------------------
        File.version: "1.0"

        Camera.type: "PinHole"

        Camera1.fx: {fx:.6f}
        Camera1.fy: {fy:.6f}
        Camera1.cx: {cx:.6f}
        Camera1.cy: {cy:.6f}

        Camera1.k1: 0.0
        Camera1.k2: 0.0
        Camera1.p1: 0.0
        Camera1.p2: 0.0

        Camera2.fx: {fx:.6f}
        Camera2.fy: {fy:.6f}
        Camera2.cx: {cx:.6f}
        Camera2.cy: {cy:.6f}

        Camera2.k1: 0.0
        Camera2.k2: 0.0
        Camera2.p1: 0.0
        Camera2.p2: 0.0

        Camera.width: {int(width)}
        Camera.height: {int(height)}

        Camera.fps: {int(round(fps))}
        Camera.RGB: 0

        Stereo.ThDepth: 35.0
        Stereo.b: {baseline_m:.6f}

        # bf = baseline (m) * fx (px) — ORB-SLAM3 reads disparity in this unit.
        Camera.bf: {bf:.6f}

        # Identity right-cam extrinsic — KITTI imagery is already rectified.
        Stereo.T_c1_c2: !!opencv-matrix
            rows: 4
            cols: 4
            dt: f
            data: [1.0, 0.0, 0.0, {baseline_m:.6f},
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 1.0]

        #--------------------------------------------------------------------------------------------
        # ORB Parameters
        #--------------------------------------------------------------------------------------------
        ORBextractor.nFeatures: {orb["n_features"]}
        ORBextractor.scaleFactor: {orb["scale_factor"]}
        ORBextractor.nLevels: {orb["n_levels"]}
        ORBextractor.iniThFAST: {orb["ini_th_fast"]}
        ORBextractor.minThFAST: {orb["min_th_fast"]}

        #--------------------------------------------------------------------------------------------
        # Viewer Parameters — disabled, we run headless.
        #--------------------------------------------------------------------------------------------
        Viewer.KeyFrameSize: 0.6
        Viewer.KeyFrameLineWidth: 2.0
        Viewer.GraphLineWidth: 1.0
        Viewer.PointSize: 2.0
        Viewer.CameraSize: 0.7
        Viewer.CameraLineWidth: 3.0
        Viewer.ViewpointX: 0.0
        Viewer.ViewpointY: -100.0
        Viewer.ViewpointZ: -0.1
        Viewer.ViewpointF: 2000.0
        Viewer.imageViewScale: 1.0
        """
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    return out_path


def run_stereo_kitti(
    seq_dir: Path,
    binary: Path,
    vocab: Path,
    yaml_path: Path,
    out_dir: Path,
    timeout_s: float = 1800.0,
) -> Path:
    """Subprocess-invoke `stereo_kitti` and return the CameraTrajectory.txt path.

    `seq_dir`  - flat per-sequence dir holding image_0/, image_1/, times.txt
    `binary`   - path to Examples/Stereo/stereo_kitti
    `vocab`    - path to Vocabulary/ORBvoc.txt
    `yaml_path`- output of `write_kitti_yaml`
    `out_dir`  - cwd for the subprocess; CameraTrajectory.txt lands here.

    The example binary takes positional args:
        stereo_kitti <vocab> <settings> <kitti_seq_dir>
    `kitti_seq_dir` is the *flat* layout (image_0/, image_1/, times.txt),
    which matches `data/kitti/<seq>/` — no nesting fixup needed.
    """
    seq_dir = Path(seq_dir).resolve()
    binary = Path(binary).resolve()
    vocab = Path(vocab).resolve()
    yaml_path = Path(yaml_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not binary.exists():
        raise FileNotFoundError(f"stereo_kitti binary missing: {binary}")
    if not vocab.exists():
        raise FileNotFoundError(f"ORBvoc.txt missing: {vocab}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"settings YAML missing: {yaml_path}")
    for required in ("image_0", "image_1", "times.txt"):
        if not (seq_dir / required).exists():
            raise FileNotFoundError(f"{seq_dir} missing {required}")

    log_path = out_dir / "stereo_kitti.log"
    cam_traj = out_dir / "CameraTrajectory.txt"
    cam_traj.unlink(missing_ok=True)
    cmd = [str(binary), str(vocab), str(yaml_path), str(seq_dir)]
    with log_path.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=out_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )

    # stereo_kitti commonly segfaults (signal 11 -> exit -11) during shutdown
    # AFTER writing the trajectory; trust the file's presence over the rc.
    if not cam_traj.exists():
        tail = "\n".join(log_path.read_text().splitlines()[-40:])
        raise RuntimeError(
            f"stereo_kitti exited {proc.returncode} and wrote no CameraTrajectory.txt; "
            f"see {log_path}\n--- tail ---\n{tail}"
        )
    if proc.returncode != 0:
        print(
            f"[warn] stereo_kitti exited {proc.returncode} but "
            f"CameraTrajectory.txt was written; continuing"
        )
    # KeyFrameTrajectory.txt is also written; we keep both in out_dir.
    return cam_traj


def find_paths(root: Path = Path("external/orb_slam3")) -> dict[str, Path]:
    """Locate the binary + vocab inside an ORB-SLAM3 build tree.

    Convenience for drivers; raises FileNotFoundError early if the build is
    missing pieces.
    """
    root = Path(root).resolve()
    binary = root / "Examples" / "Stereo" / "stereo_kitti"
    vocab_txt = root / "Vocabulary" / "ORBvoc.txt"
    vocab_tar = root / "Vocabulary" / "ORBvoc.txt.tar.gz"

    if not binary.exists():
        raise FileNotFoundError(
            f"{binary} not built — run external/orb_slam3/build.sh first"
        )
    if not vocab_txt.exists():
        if vocab_tar.exists():
            subprocess.run(["tar", "-xzf", str(vocab_tar)], cwd=vocab_tar.parent, check=True)
        else:
            raise FileNotFoundError(f"missing both {vocab_txt} and {vocab_tar}")
    if shutil.which("ffmpeg") is None:
        # Not strictly needed by ORB-SLAM3, but the render harness uses it.
        raise RuntimeError("ffmpeg not on PATH — needed for render_debug videos")
    return {"binary": binary, "vocab": vocab_txt}
