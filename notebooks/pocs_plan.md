# POC plan — ORB-SLAM3, BASALT, DROID-SLAM on KITTI seq 04

Goal: get a GT-anchored ATE/RPE number from each of the three SOTA non-LiDAR
methods on our benchmark, so we have a reference to compare the current
pipeline + C3/C4/C5 BA priors against.

## Ground rules
- **No edits to existing `src/aivision/*`.** POCs only import from it.
- All POC modules live under `src/aivision/pocs/`.
- All POC artefacts (trajectories, videos, logs) under `outputs/pocs/<method>/`.
- External sources / binaries clone into `external/<method>/` (gitignored).
- Each POC method gets its own uv venv if its deps risk conflicting with the
  main project (DROID-SLAM definitely; ORB-SLAM3 / BASALT are C++ so no risk).

## Shared eval harness — `src/aivision/pocs/common.py`
Built once, reused by all three. New module, not a refactor of existing code.

- `load_tum_trajectory(path) -> (N,4,4) cam_to_world, list[float] timestamps`
- `load_kitti_trajectory(path) -> (N,4,4) cam_to_world`  (12-float-per-row format)
- `evaluate(est_cam_to_world, seq, delta_rpe=10) -> dict`
  - reuses `aivision.kitti.umeyama_similarity` + `absolute_trajectory_error` + `relative_pose_error`
  - also reports per-axis lateral / vertical / longitudinal residuals after Sim(3) align
- `render_debug(traj, points_xyz, points_rgb, K, frame_paths, out_dir, gt_xyz)`
  - lifted from `outputs/kitti04_5s/render_debug.py` pattern
  - produces `orbit.mp4`, `reproj.mp4`, `trajectory.png` per POC
  - `points_xyz=None` skips orbit + reproj (some methods don't return a cloud)

## POC1 — ORB-SLAM3 (start here)
The proven KITTI stereo baseline. Indirect (ORB), supports loop closure.

### Phase A — stereo only (no IMU)
Prep:
1. Extract `image_1/` (right grayscale) for seq 04 from the already-downloaded
   `data/kitti/raw/greyscale/data_odometry_gray.zip` → `data/kitti/04/image_1/`.
   (~67 MB, same procedure as image_0 earlier.)
2. Clone + build ORB-SLAM3 at `external/orb_slam3/`. Deps: OpenCV ≥4, Eigen3,
   Pangolin, DBoW2 (vendored), g2o (vendored). Vocab file `ORBvoc.txt.tar.gz`
   ships with the repo (~145 MB extracted).

Module `src/aivision/pocs/orb_slam3.py`:
- `write_kitti_yaml(out_path, K, width, height, baseline_m)` — synthesise the
  YAML config ORB-SLAM3 demands; for seq 04 the stereo baseline is
  `bf / fx = -P1[0,3] / P0[0,0] = 379.81 / 707.09 ≈ 0.537 m`.
- `run_stereo_kitti(seq_dir, binary, vocab, yaml, out_dir) -> Path`
  - Subprocess: `<binary>/Examples/Stereo/stereo_kitti <vocab> <yaml> <seq_dir>`
  - ORB-SLAM3 writes `CameraTrajectory.txt` (TUM format) and
    `KeyFrameTrajectory.txt` to its CWD; we cd to `out_dir` first.
- (Trajectory loaded via `common.load_tum_trajectory`.)
- ORB-SLAM3 does not export a point cloud through the example binary; we'll
  pass `points_xyz=None` to `render_debug` for Phase A and only get the
  trajectory plot.

Driver `outputs/pocs/orb_slam3/run_phaseA.py`:
- run → load → evaluate against full 271-frame `seq.gt_positions()` → render
- Acceptance: ATE rmse < 0.5 m on seq 04 (their paper reports ~0.16 m).

### Phase B — stereo-inertial (deferred until Phase A passes)
- Pull the KITTI raw drive that backs odometry seq 04: `2011_09_30_drive_0016`,
  for the 100 Hz OXTS IMU stream. KITTI raw drives are downloadable per-drive,
  ~1 GB.
- Convert OXTS `.txt` rows → timestamps + 6-axis IMU in ORB-SLAM3's
  `stereo_inertial_kitti` expected layout.
- Run `stereo_inertial_kitti`, repeat eval. Compare Phase A vs Phase B ATE to
  isolate the IMU contribution — directly informs whether a phone IMU would
  help the dashcam pipeline.

## POC2 — BASALT (after POC1 lands)
TUM stereo-VIO with an offline BA backend — structurally closest to our
"VO + BA" stack, so the most relevant reference for the C3/C4/C5 work.

- Build at `external/basalt/` (CMake; deps: OpenCV, Eigen3, TBB, Boost, fmt,
  CLI11, Pangolin). Upstream provides Arch instructions.
- Calibration in BASALT's JSON format. Their `basalt_kitti_eval` script
  consumes the standard KITTI layout we already have on disk.
- Wrapper `src/aivision/pocs/basalt.py`:
  - `run_kitti_vio(seq_dir, calib_json, out_dir)` subprocess
  - Output: TUM trajectory + (optional) sparse cloud
- IMU needed for full VIO; stereo-only mode exists but is documented as
  degraded — only run it if Phase B of POC1 confirms IMU adds value.

## POC3 — DROID-SLAM (after POC1 lands)
Deep-learned dense SLAM. No IMU support; uses mono or stereo. The interesting
test is whether its visual front-end stays stable on the straight-line case
where classical front-ends struggle.

- Separate uv venv `external/droid_slam/.venv/` — its torch pin (2.0.x) and
  CUDA extensions will clash with our `torch>=2.4`.
- Clone + `pip install -e .` + download weights from their Google Drive.
- GPU: user has Nvidia (saw `10de:2b85` device ID in EGL output earlier).
- Wrapper `src/aivision/pocs/droid_slam.py`:
  - `run_stereo(seq_dir, weights, out_dir)` subprocess via their `demo.py`
  - Output: poses + dense map (.ply)
- ~minutes per sequence — offline-grade, but seq 04 at 271 frames should fit
  in well under a minute on consumer Nvidia.

## Layout summary
```
src/aivision/pocs/
    __init__.py
    common.py        # shared eval + render harness
    orb_slam3.py     # POC1 wrapper
    basalt.py        # POC2 wrapper
    droid_slam.py    # POC3 wrapper
external/            # gitignored
    orb_slam3/
    basalt/
    droid_slam/
outputs/pocs/
    orb_slam3/       # per-POC trajectories, videos, logs
    basalt/
    droid_slam/
```

## Ordering
1. **`common.py`** — build the eval+render harness first. Reuses
   `aivision.kitti` + `aivision.render` + the `render_debug.py` pattern.
   No external deps yet.
2. **Extract `image_1/`** for seq 04 from the gray zip already on disk.
3. **POC1 Phase A** — build ORB-SLAM3, run stereo, get first GT-anchored ATE
   from a SOTA reference. This unblocks the comparison the C5 verdict said we
   needed.
4. **POC1 Phase B** OR **POC2** OR **POC3** in any order after that — the
   harness is reusable so each subsequent POC is mostly the wrapper + a driver.

## Open questions (deferred to implementation)
- ORB-SLAM3's stereo baseline interpretation: `P1[0,3] / fx` vs `P0[0,3]−P1[0,3]`
  — verify against their KITTI yaml shipped in `Examples/Stereo/KITTI04-12.yaml`
  rather than computing from scratch.
- DROID-SLAM weights URL changes occasionally — pin a commit hash.
- For Phase B IMU, KITTI raw OXTS is 100 Hz; ORB-SLAM3 stereo-inertial wants
  IMU at ≥100 Hz with monotonic timestamps — should be a direct fit.
- LICENSE: ORB-SLAM3 = GPLv3 (binary call only, OK). BASALT = BSD. DROID-SLAM
  = research / non-commercial — fine for benchmarking.
