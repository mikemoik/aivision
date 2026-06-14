"""Backfill the generation record of the kitti-seq04 ADAS map.

The seq04 map was produced by an iterative sequence of scripts. This records
that process after the fact as a ``map-gen/`` tree + a BuildManifest stored in
``map_adas.json`` — one step per role, each with its method ``type``, the
artifact creation timestamp (from file mtime), input refs, and the output
interface contract copied into ``out/``. Heavy method-private artifacts are
referenced (and left in their legacy ``outputs/`` location) rather than moved;
those steps carry a ``note``. This is a one-time backfill, not the orchestrator.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aivision.map import MapLayout, StepRecord
from aivision.map.ingest import SEQ04_ENU_OFFSET, build_seq04_map
from aivision.map.provenance import utc_now

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"


def mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def write_step(layout: MapLayout, rec: StepRecord) -> None:
    layout.ensure_step(rec.role)
    layout.step_record(rec.role).write_text(json.dumps(rec.to_dict(), indent=2))


def trajectory_to_json(src: Path, dst: Path) -> None:
    """ORB CameraTrajectory.txt (Nx12, Twc 3x4 row-major) -> canonical contract."""
    rows = np.loadtxt(src)
    dst.write_text(json.dumps({
        "frame_id": "M",
        "convention": "Twc 3x4 row-major [R|t], camera-to-world, metric metres",
        "n_frames": len(rows),
        "poses": rows.tolist(),
    }, indent=2))


def main() -> None:
    layout = MapLayout("kitti-seq04").ensure()
    m = build_seq04_map(created=utc_now())  # objects + georef, as before

    traj = OUT / "pocs/orb_slam3/CameraTrajectory.txt"
    lanes_poly = OUT / "ego_lanes/lane_map/lane_polylines.json"
    lanes_model = OUT / "ego_lanes/lane_map/lane_polylines_model.json"
    poles = OUT / "static_env/poles_signs/poles_signs_world.json"
    arrows = OUT / "static_env/arrows_lanemodel/arrows_world.json"
    rho = OUT / "pocs/orb_slam3/inv_depth_fused_mono_rho_gf_16bit.mkv"
    perception = OUT / "pocs/orb_slam3/perception"

    # --- slam: ORB3 -> canonical trajectory contract; raw kept in work/ ---
    write_step(layout, StepRecord(
        role="slam", type="ORB3", created=mtime(traj),
        params={"baseline_m": 0.5372, "sensor": "stereo"},
        inputs=["bindata/input/odometry"],
        outputs=["map-gen/slam/out/trajectory.json"],
        note="ORB-SLAM3 stereo; non-deterministic run-to-run, frozen as input artifact.",
    ))
    trajectory_to_json(traj, layout.step_out("slam") / "trajectory.json")
    shutil.copy2(traj, layout.step_work("slam") / "CameraTrajectory.txt")

    # --- depth + segmentation: perception caches (heavy, referenced not moved) ---
    write_step(layout, StepRecord(
        role="depth", type="inv-depth-fused-mono-rho-gf", created=mtime(rho),
        inputs=["bindata/input/odometry"],
        note=f"artifact not migrated: {rho.relative_to(REPO)}",
    ))
    write_step(layout, StepRecord(
        role="segmentation", type="Mapillary-Vistas", created=mtime(perception),
        inputs=["bindata/input/odometry"],
        note=f"per-frame cache not migrated: {perception.relative_to(REPO)}",
    ))

    # --- lanes: EgoLanes + gray stereo -> polylines + coupled lane model ---
    write_step(layout, StepRecord(
        role="lanes", type="EgoLanes+gray-stereo", created=mtime(lanes_poly),
        inputs=["slam", "bindata/input/odometry"],
        outputs=["map-gen/lanes/out/lane_polylines.json",
                 "map-gen/lanes/out/lane_polylines_model.json"],
    ))
    shutil.copy2(lanes_poly, layout.step_out("lanes") / "lane_polylines.json")
    shutil.copy2(lanes_model, layout.step_out("lanes") / "lane_polylines_model.json")

    # --- objects: poles/signs/lights/guideposts ---
    write_step(layout, StepRecord(
        role="objects", type="Vistas-seg+GTSRB-ViT+delineator", created=mtime(poles),
        inputs=["slam", "depth", "segmentation", "lanes"],
        outputs=["map-gen/objects/out/poles_signs_world.json"],
    ))
    shutil.copy2(poles, layout.step_out("objects") / "poles_signs_world.json")

    # --- markings: road arrows on the lane-model plane ---
    write_step(layout, StepRecord(
        role="markings", type="RLMD+lanemodel-plane", created=mtime(arrows),
        inputs=["slam", "lanes"],
        outputs=["map-gen/markings/out/arrows_world.json"],
    ))
    shutil.copy2(arrows, layout.step_out("markings") / "arrows_world.json")

    # --- georef: Umeyama(ORB centres vs GNSS) + stored ENU offset ---
    write_step(layout, StepRecord(
        role="georef", type="Umeyama+GNSS", created=utc_now(),
        params={"with_scale": True, "enu_offset_m": SEQ04_ENU_OFFSET.tolist()},
        inputs=["slam", "bindata/input/oxts"],
        outputs=["map-gen/georef/out/georef.json"],
    ))
    (layout.step_out("georef") / "georef.json").write_text(json.dumps(m.georef.to_dict(), indent=2))

    # --- assemble: ingest -> the map DB (the product) ---
    write_step(layout, StepRecord(
        role="assemble", type="ingest-v1", created=utc_now(),
        inputs=["lanes", "objects", "markings", "georef"],
        outputs=["map_adas.json"],
    ))

    # attach the manifest (DAG order) and persist
    for role in ("slam", "depth", "segmentation", "lanes", "objects", "markings", "georef", "assemble"):
        m.provenance.add(StepRecord.from_dict(json.loads(layout.step_record(role).read_text())))
    m.save(layout.db)
    print(f"backfilled {len(m.provenance.steps)} steps; map has {len(m)} objects -> {layout.db}")


if __name__ == "__main__":
    main()
