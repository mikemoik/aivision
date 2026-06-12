# aivision

Structure-from-Motion (SfM) for automotive front-camera footage. Classical
pipeline built on OpenCV + pycolmap, with PyTorch/Kornia available for
learning-based extensions.

## Setup

```bash
uv sync                              # creates .venv and installs everything
uv run aivision download-models      # prefetch DISK + LightGlue into ./models/
source .venv/bin/activate            # optional — or prefix commands with `uv run`
```

All model weights are cached under `./models/` (gitignored). `aivision`
redirects `TORCH_HOME` / `HF_HOME` there automatically, so nothing leaks into
`~/.cache/`.

## Quickstart

Drop a short driving clip (or a folder of sequential JPGs) into `data/`, then:

```bash
uv run aivision two-view data/your_clip.mp4 --step 5
# or
uv run aivision two-view data/frames_dir/
```

Outputs land in `outputs/two_view/`:
- `matches.png` — feature-match overlay between the two sampled frames
- `points.ply` — triangulated 3D points (open in MeshLab, CloudCompare, or Open3D)

The translation vector `t` is unit-length — monocular two-view geometry leaves
absolute scale unobservable. Adding more frames and a scale source (wheel
odometry, IMU, or a known object) is the next step.

## Layout

```
src/aivision/
  io.py          # read frames from a video file or image directory
  camera.py      # pinhole intrinsics (placeholder — calibrate per camera)
  features.py    # ORB detection + mutual-NN matching with ratio test
  sfm.py         # essential matrix → pose recovery → triangulation
  viz.py         # match overlays + PLY export
  pipeline.py    # end-to-end demos composed from the above
  cli.py         # typer CLI (entry point: `aivision`)

data/            # raw videos / frame directories (gitignored)
outputs/         # results (gitignored)
models/          # cached model weights (gitignored; see models/README.md)
notebooks/       # jupyter experiments
```

## GPS sidecars

If a video has a sibling `<name>.json` file with per-second GPS samples
(schema: `locations[]` with `timestamp/latitude/longitude/course/speed/accuracy`,
plus `startTime/endTime`), it is auto-detected and used as prior positions for
the COLMAP backend. This:

- Fixes COLMAP's initial-pair-selection failure on pure forward motion
- Anchors the reconstruction in real-world ENU metres (origin = first GPS sample)
- Acts as a soft constraint during bundle adjustment, killing long-term drift

```bash
uv run aivision pointcloud-video data/1.mp4 --backend colmap        # auto-detects 1.json
uv run aivision pointcloud-video data/1.mp4 --backend colmap --gps elsewhere.json
```

## Roadmap

1. Two-view geometry (done — `aivision two-view`)
2. Multi-view SfM via `pycolmap` with GPS priors (done — `aivision pointcloud-video`)
3. Chained VO fallback when COLMAP fails (done)
4. Per-camera calibration helper (chessboard or driving-scene self-calibration)
5. Dense depth via learned monocular models (Kornia / torch hub)
6. Switch ORB → DISK + LightGlue in the VO matcher for richer correspondences

## Hardware

CUDA-enabled PyTorch is installed by default. Verify the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
