# T0 — Map-gen workflow contract audit (gate before T1)

**Goal of T0:** confirm every role in the `mapgen/kitti-seq04` DAG has a fully
specified I/O contract — input interface, output interface (the `out/` artifact
schema), method `type`, and `params` — and that every handoff is unambiguous.
Only proceed to T1 (implement steps) once each step is well-defined.

Audited against: `mapgen/kitti-seq04/config.json`, the backfill run
(`map/kitti-seq04/bindata/mapgen/kitti-seq04-backfill/map-gen/<role>/{step.json,out/}`),
`src/aivision/map/{mapgen,provenance,ingest,layout}.py`, and the real producer
scripts (`scripts/{run_orbslam3,build_lane_map,fit_lane_model,lift_poles_signs,
lift_arrows_lanemodel}.py`, `src/aivision/perception.py`).

## DAG (declared inputs → out artifact)

```
input:odometry ─┬─▶ slam ──────────────▶ out/trajectory.json ─┐
                ├─▶ depth ─────────────▶ out/<inv-depth>  ?    │
                └─▶ segmentation ──────▶ out/<perception>  ?   │
input:odometry ─┐                                              │
        slam ───┴─▶ lanes ─▶ out/lane_polylines.json           │ (→ assemble)
                            └ out/lane_polylines_model.json ────┤ (→ objects, markings)
 slam, depth, segmentation, lanes ─▶ objects ─▶ out/poles_signs_world.json ─┐
              slam, lanes ─────────▶ markings ─▶ out/arrows_world.json ──────┤
        slam, input:oxts ─────────▶ georef ───▶ out/georef.json ─────────────┤
                                                                              ▼
   lanes, objects, markings, georef ─▶ assemble ─▶ map_adas.json   (DONE/validated)
```

## Per-role verdict

| role | type (config) | input iface | out schema | params | verdict |
|---|---|---|---|---|---|
| **slam** | `ORB3` | odometry slice (image_0/1, times.txt, calib) + baseline | `out/trajectory.json` `{frame_id, convention, n_frames, poses:[3×4 Twc row-major]}` (producer emits KITTI `CameraTrajectory.txt` → wrap to JSON; raw goes to `work/`) | `baseline_m`, `sensor` | ✅ well-defined |
| **depth** | `inv-depth-fused-mono-rho-gf` | odometry (stereo + mono rho fusion) | ❌ **undeclared** — consumer reads a 16-bit `inv_depth_fused_mono_rho_gf_16bit.mkv` (`rho=(pix-1)/65534`, `Z=1/rho`) | none | ⚠️ **GAP** |
| **segmentation** | `mapillary-vistas` | odometry (image_2 color) | ❌ **undeclared** — producer exists (`perception.build_perception_cache`): cache dir of `frame_NNNNNN.npz {semantic uint8 H×W, instance uint16 H×W}` + `manifest.json` | none | ⚠️ **GAP** |
| **lanes** | `egolanes+gray-stereo` | slam traj + odometry (image_2, gray stereo) | `out/lane_polylines.json` (border polylines) **and** `out/lane_polylines_model.json` (coupled lane model = road plane) | `z_max`, `dz` | ⚠️ two-artifact handoff to disambiguate |
| **objects** | `vistas+gtsrb+delineator` | slam, depth(rho mkv), segmentation(cache), lanes(model) | `out/poles_signs_world.json` `{poles,signs,traffic_lights,guideposts}` each `{center,height/size,res_m,n_frames,...}` | `pole_eps`, `sign_conf`, `gp_conf` (subset of producer knobs) | ✅ (depends on depth+seg gaps) |
| **markings** | `rlmd+lanemodel` | slam, lanes(model) | `out/arrows_world.json` list of `{cls,centroid,forward,length,width,yaw_to_lane_deg,n_frames,glyph_world}` | `conf`, `cluster_eps` (subset) | ✅ well-defined |
| **georef** | `umeyama+gnss` | slam traj + input:oxts | `out/georef.json` `{T_M_to_ENU 4×4, datum[lat,lon,alt], enu_offset}` | `with_scale`, `enu_offset_m` | ✅ DONE + validated |
| **assemble** | `ingest-v1` | lanes, objects, markings, georef | `map_adas.json` (+ BuildManifest) | — | ✅ DONE + validated |

## Gaps — fix these before the step they block

1. **`depth` out contract undeclared + producer not located (blocks T5/objects).**
   `objects` consumes the *fused-rho* stream `inv_depth_fused_mono_rho_gf_16bit.mkv`,
   not the perception cache's `mono_depth` field — there are **two** depth
   representations in the tree and the contract must name the rho mkv explicitly.
   Action: declare `out/inv_depth_rho16.mkv` (rho encoding `rho=(pix-1)/65534`,
   `Z=1/rho`, N×H×W gray16) and find/codify the `_rho_gf` producer (it is **not**
   among `scripts/lift_*`).

2. **`segmentation` out contract undeclared though producer exists (blocks T5).**
   Declare `out/` = perception cache (`frame_NNNNNN.npz {semantic uint8, instance
   uint16}` + `manifest.json`), produced by `perception.build_perception_cache`.

3. **`lanes` emits two artifacts — name which consumer reads which.**
   `assemble` reads `lane_polylines.json` (the border polylines that become
   `LaneMarking`s); `objects` **and** `markings` read `lane_polylines_model.json`
   (the coupled lane model used as the ground plane). The step must run **both**
   producers (`build_lane_map.py` → polylines, `fit_lane_model.py` → model). Make
   the per-consumer artifact explicit in the contract.

## Consistency nits (cheap to fix now, costly later)

- **`type` dispatch is exact-match and fails silently.** `MapGen.run` keys
  `STEP_REGISTRY[(role, type)]`; a config `type` that doesn't equal the class's
  `type` attribute falls through to `_passthrough` (reuses old `step.json`) with
  no error. config vs backfill step.json already disagree on casing
  (`mapillary-vistas` vs `Mapillary-Vistas`, `egolanes+gray-stereo` vs
  `EgoLanes+gray-stereo`, etc.). Pin one canonical string per role and make the
  step class match it; consider an assert that every config step resolves to a
  registered class under `force`.
- **Standardize the `inputs` ref vocabulary.** Mix of role names (`"slam"`),
  `"input:oxts"` (GeorefStep code), and raw paths (`"bindata/input/odometry"` in
  backfill step.json). Settle on: upstream → role name; config input →
  `"input:<key>"`. Never a raw path.
- **`params` are a curated subset of producer knobs** (e.g. objects exposes
  `pole_eps/sign_conf/gp_conf` of ~15 CLI args). Acceptable, but when wrapping
  each `lift_*` core into a lib fn, the un-exposed knobs become step-class
  defaults — document them so the contract is the full signature, not just config.

## Verdict

**Conditional GO.** 6 of 8 roles (slam, lanes, markings, georef, assemble — plus
objects modulo its upstreams) have specified, DAG-consistent contracts. georef +
assemble are already implemented and reproduce GT exactly. **Two contracts must
be written before their consumer step (T5/objects):** `depth` (declare out +
locate producer) and `segmentation` (declare out). `lanes` needs its two-artifact
handoff named. None of these block **T1 (slam)** — start there, and close gaps
1–3 before T5. Adopt the two consistency fixes (canonical `type`, input vocab)
during T1 so the framework is sound before steps accumulate.
