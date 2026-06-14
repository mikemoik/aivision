# Static-environment Blender cinematic — intent, design, and how to reproduce

A reproducible record of the "lift-and-return" reveal we built from the aivision static-environment
3D maps. Saved so it can be re-rendered or re-tuned later.

## Intent

Communicate the static-environment reconstruction by anchoring it to the *real* dashcam footage:
show reality, progressively reveal the reconstructed 3D scene registered to it, fly the camera up
to view the scene in 3D heading toward the signalised junction, then settle back onto the dashcam
drive and fade back to reality. The viewer experience is **reality → reconstruction → 3D view →
reality**, all locked to the original KITTI seq04 video so it could later be blended/composited.

## What is in the scene (`outputs/static_env/blender/static_env.blend`)

Built procedurally from the triangulated maps; **poles are excluded** (their poses were not accurate
enough — see the [[task_static_env_3d]] memory). Everything is in a re-runnable `StaticEnv` collection.

- **ego lane** — grey asphalt ribbon mesh between the densified (0.5 m Z-grid) left/right borders.
- **20 cm borders** — raised white curbs (0.20 m wide × 0.12 m tall) swept along each border.
- **road arrows** — flat white glyph polygons (`arrows_world.json`) ~2 cm above the surface.
- **traffic signs** — post + class plate: speed-limit = white disc + red rim, Yield = red inverted
  triangle, Ahead-only = blue disc (`poles_signs_world.json` → signs).
- **traffic lights** — post + dark housing + 3 emissive lamps (red/amber/green).
- **guideposts (Leitpfosten)** — white post of estimated height + black band.
- **ground** — flat green plane with the ego-lane footprint **cut out** (boolean) so there is no
  ground between the lane boundaries → no z-fighting with the lane surface.
- **sky** — large emissive blue sphere (infinite sky), seen from inside.

### Coordinate frame / alignment
The maps live in the cam0 world frame (X-right, Y-**down**, Z-forward). The scene was rigidly aligned
so the **ego lane lies on the XY plane (Z-up)**: best-fit plane of the densified lane via SVD
(normal n0 = smallest singular vector oriented up, forward f0 = largest, right = f0×n0), then
`R_align = [right; f0; n0]`, `t = -R_align·c0` (c0 = lane centroid). Applied as **one rigid transform**
to a parent empty `SE_root` (no bending, vertex positions preserved). Planarity residual after the
fit: **std 0.073 m, max 0.18 m** (= the road's real grade; cannot be removed without bending).

## Camera design (the part we iterated on)

The camera matches the **KITTI P2 intrinsics** (sensor_fit HORIZONTAL, sensor 36 mm, lens =
fx·36/W = 20.76 mm, shift_x = (W/2−cx)/W, shift_y = (cy−H/2)/W, square pixels, 1226×370). Because the
scene and camera were moved by the *same* rigid transform, the on-drive frames project identically to
the real camera → they register pixel-for-pixel with the original video (verified by a 50/50 overlay).

Design evolution (kept for context):
1. orbit overview — *"too much"*.
2. elevated forward dolly to the junction (+15 m up, 10 m left) — better.
3. heading locked after the elevation completes.
4. **final**: a **lift-and-return** — the camera follows the original ORB-SLAM3 drive poses
   *continuously*; the lift is a weighted **offset** (rise → apex → descend) that returns to zero, so
   it rejoins the drive *"as if there was no lift"* (NOT a rewind/palindrome — an earlier palindrome
   version was wrong). Heading eases onto the traffic-light junction at the apex via quaternion slerp.

`cam_matrix(f) = lerp(dashcam_pose(f), lift_pose(f), w(f))` with slerp on rotation, where
`lift_pose = (centerlineX(Y)−10, Y, 15)` aimed at the junction centroid, Y = the drive's own forward Y.

### Saved parameters (final)
- Frames: **NF = 271** (= drive poses 0..270), **10 fps**.
- Lift weight `w(f)` schedule: 0 before **110**; smoothstep up **110→145**; hold 1 **145→180**;
  smoothstep down **180→215**; 0 after. Offsets: **+15 m** up, **10 m** left (−X aligned).
- Intrinsics: W=1226, H=370, fx=707.0912, cx=601.8873, cy=183.1104.
- **Sequence split** (forward reveal / reverse on return). markings+objects = arrows + white border
  curbs + signs + lights + guideposts (NOT the grey lane surface / ground / sky):
  - 0–19 real · 20–29 fade-in markings+objects · 30–59 real+objects · 60–79 dissolve→full ·
    80–214 full (drive 80–110, lift 110–215) · 215–229 reverse dissolve full→real+objects ·
    230–239 markings fade-out · 240–270 real tail.
  - Tuning history: lift was shifted 25 frames earlier (drive-before-lift 80–135 → 80–110), keeping
    every subsequent duration, which lengthened the closing real tail by 25 frames.

## Files
- `outputs/static_env/blender/static_env.blend` — scene layout (canonical).
- `scripts/blender_static_env_render.py` — camera path + intrinsics; renders the full + objects layers.
- `scripts/static_env_cinematic_compose.py` — the sequence split; composites + assembles the mp4.
- `outputs/static_env/blender/reveal_lift_return.mp4` — the final cinematic (the hero shot).
- `outputs/static_env/blender/scene_kitti_intrinsics.mp4` — full render at KITTI intrinsics (blend-ready).
- Source maps: `outputs/ego_lanes/lane_map/lane_polylines_model.json`,
  `outputs/static_env/{arrows_lanemodel/arrows_world.json,poles_signs/poles_signs_world.json}`,
  poses `outputs/pocs/orb_slam3/CameraTrajectory.txt`.

## Reproduce
1. **Scene**: open `static_env.blend` in Blender 5.1 (or rebuild it from the maps — the build recipe
   is in the [[task_static_env_blender]] memory: align→lane surface→curbs→arrows→signs/lights/
   guideposts→ground hole→sky; cam0 frame, then the rigid XY alignment).
2. **Render layers**: `blender --background outputs/static_env/blender/static_env.blend --python
   scripts/blender_static_env_render.py` → writes `/tmp/se_k2` (full) and `/tmp/se_o2` (objects RGBA).
3. **Composite + assemble**: `uv run python scripts/static_env_cinematic_compose.py` →
   `outputs/static_env/blender/reveal_lift_return.mp4`.
4. **Cuts/variants**: `ffmpeg -framerate 10 -start_number A -i /tmp/se_final/f%04d.png -frames:v N ...`.

Tweak knobs: the `LIFT` dict (frame schedule / height / left) in the render script for the camera;
the `S` dict in the compose script for the sequence split.
