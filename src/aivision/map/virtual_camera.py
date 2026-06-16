"""Virtual camera sensor — replay the per-frame static-object detections.

This is the reader half of the virtual sensor. The producing half lives in the
map-generation steps, which persist per-frame detections (previously computed then
discarded) as ``*_by_frame.json`` artifacts — one per perception role:

  * ``objects``  → ``detections_by_frame.json``       (poles / signs / lights / guideposts)
  * ``lanes``    → ``lane_detections_by_frame.json``  (ego-left / ego-right borders)
  * ``markings`` → ``arrow_detections_by_frame.json`` (ego-lane road arrows)

Each frame's detections are :class:`~aivision.map.objects.MapObject` instances — the
SAME data structures as the aggregated ``map_adas.json`` — only their geometry lives
in THIS frame's CAMERA coordinates (``X``-right, ``Y``-down, ``Z``-forward, metric).
:meth:`from_map` loads and MERGES whichever of the three are present.

DESIGN PRINCIPLE — the sensor knows nothing about the map. The output is the raw
single-frame perception (real depth noise, FOV limits, FP/FN), NOT a reprojection of
the finished, aggregated map. The map is built *from* this sensor stream, so the
sensor cannot depend on it: no cluster ids, no kept flags, no world poses. Image-space
detector outputs (pixel box, centroid, score, range, …) ride in each object's
``attributes``.

Usage::

    sensor = VirtualCameraSensor.from_map("kitti-seq04")
    for t, objs in sensor:
        for obj in objs:               # MapObject in camera coords
            print(obj.cls.value, sensor.center(obj), obj.attributes.get("range_m"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import GaussianPoint, GaussianPose, LineSegment, OrientedBox, Polyline
from .layout import MapLayout
from .map_adas import _representative_point
from .objects import MapObject

SCHEMA = "virtual_camera_detections_v3"

# Per-frame detections live in the MOVING vehicle frame f_veh(t): the f_veh body
# frame at frame t (x-fwd, y-left, z-up). Producers build geometry in camera (cam0)
# coords then reframe by ``sensor_T_fveh_cam`` (the static cam0->f_veh extrinsic).
SENSOR_FRAME = "f_veh(t): vehicle frame at frame t (X-forward, Y-left, Z-up), metric meters"


def sensor_T_fveh_cam() -> np.ndarray:
    """Static cam0 -> f_veh transform used to reframe per-frame detections into f_veh(t)."""
    from ..calibration import KittiCalibration
    return KittiCalibration.load().T_fveh("cam0")

# (role, artifact filename) for every per-frame sensor stream a run may carry.
ARTIFACTS = [
    ("objects", "detections_by_frame.json"),
    ("lanes", "lane_detections_by_frame.json"),
    ("markings", "arrow_detections_by_frame.json"),
]


def make_artifact(frames: dict, *, n_frames: int, image_size, K, camera: str = "image_2",
                  note: str | None = None) -> dict:
    """Build a self-describing per-frame sensor artifact (the producer side).

    ``frames`` maps ``str(frame_idx) -> [MapObject.to_dict(), ...]`` in CAMERA coords.
    Single schema definition shared by every role's step so the reader stays generic.
    """
    cls_values = sorted({d["cls"] for fr in frames.values() for d in fr})
    return {
        "schema": SCHEMA,
        "frame": SENSOR_FRAME,
        "note": note or ("per-frame detections as MapObject.to_dict() in the MOVING vehicle "
                         "frame f_veh(t); same data structures as map_adas.json "
                         "(read via VirtualCameraSensor)"),
        "camera": camera,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "K": [[float(v) for v in row] for row in np.asarray(K)],
        "n_frames": int(n_frames),
        "classes": cls_values,
        "frames": frames,
    }


@dataclass
class PixelNoise:
    """Image-sensor pixel-noise model applied to the sensor's reported detections.

    Models a detector whose image-plane localization is uncertain to ``sigma_px``
    pixels (1-sigma, per measured point, both axes). Each reported 3D point is the
    back-projection of a pixel; adding Gaussian pixel noise and re-projecting at the
    SAME depth (pixel noise carries no range information) perturbs the point laterally
    and vertically by ``Z * sigma_px / f`` while leaving the forward range untouched.
    The same scale inflates the point's covariance in the bearing plane.

    The draw is deterministic per (``seed``, frame): one frame always yields the same
    noisy measurement no matter how many times it is read (so EM re-association and
    scoring see a single, fixed realization of the sensor).
    """

    sigma_px: float = 10.0
    seed: int = 0
    perturb_mean: bool = True   # shift each measured point's mean by the pixel noise
    inflate_cov: bool = True    # add the matching bearing-plane covariance


def _noisy_point(mean, cov, *, fx, fy, T_cam_fveh, T_fveh_cam, sigma_px, rng,
                 perturb: bool, inflate: bool):
    """Apply image-plane pixel noise to one f_veh(t) point (mean, 3x3 cov).

    Projects ``mean`` into the camera, jitters the pixel by N(0, sigma_px^2) on each
    axis, back-projects at the unchanged depth, and rotates the induced bearing-plane
    covariance back into f_veh. Points at/behind the image plane are left untouched.
    """
    p_cam = T_cam_fveh[:3, :3] @ np.asarray(mean, float) + T_cam_fveh[:3, 3]
    z = float(p_cam[2])
    if z <= 1e-3:                                   # not in front of the camera
        return np.asarray(mean, float), np.asarray(cov, float)
    sx, sy = z * sigma_px / fx, z * sigma_px / fy   # metric lateral/vertical scale
    out_mean, out_cov = np.asarray(mean, float), np.asarray(cov, float)
    if perturb:
        du, dv = rng.normal(0.0, sigma_px, size=2)
        p_cam_n = np.array([p_cam[0] + du * z / fx, p_cam[1] + dv * z / fy, z])
        out_mean = T_fveh_cam[:3, :3] @ p_cam_n + T_fveh_cam[:3, 3]
    if inflate:
        R = T_fveh_cam[:3, :3]
        out_cov = out_cov + R @ np.diag([sx ** 2, sy ** 2, 0.0]) @ R.T
    return out_mean, out_cov


def _apply_pixel_noise(geom, **kw):
    """Return a copy of a geometry primitive with pixel noise on every measured point."""
    if isinstance(geom, GaussianPoint):
        m, c = _noisy_point(geom.mean, geom.cov, **kw)
        return GaussianPoint(m, c)
    if isinstance(geom, GaussianPose):
        m, c = _noisy_point(geom.translation, geom.cov[:3, :3], **kw)
        T = geom.T.copy(); T[:3, 3] = m
        cov = geom.cov.copy(); cov[:3, :3] = c
        return GaussianPose(T=T, cov=cov)
    if isinstance(geom, LineSegment):
        return LineSegment(_apply_pixel_noise(geom.a, **kw), _apply_pixel_noise(geom.b, **kw))
    if isinstance(geom, Polyline):
        return Polyline([_apply_pixel_noise(v, **kw) for v in geom.verts], closed=geom.closed)
    if isinstance(geom, OrientedBox):
        return OrientedBox(_apply_pixel_noise(geom.pose, **kw),
                           geom.extents.copy(), geom.extents_cov.copy())
    return geom


class VirtualCameraSensor:
    """Reader over one or more ``*_by_frame.json`` artifacts (merged per frame).

    Construct from explicit artifact path(s), or via :meth:`from_map` to find and
    merge a map run's object / lane / marking streams. :meth:`detections` returns the
    frame's objects as :class:`MapObject` instances (camera coordinates). Pass
    ``noise=PixelNoise(...)`` to model an imprecise image sensor: each reported
    detection is jittered (mean + covariance) by the pixel-noise model.
    """

    def __init__(self, *paths: str | Path, noise: PixelNoise | None = None):
        if not paths:
            raise ValueError("VirtualCameraSensor needs at least one artifact path")
        self.paths = [Path(p) for p in paths]
        self.noise = noise
        self._frames: dict[str, list] = {}
        self.n_frames = 0
        self.K = None
        self.image_size = (0, 0)
        self.camera = ""
        self.classes: list[str] = []
        for p in self.paths:
            self._merge(json.loads(p.read_text()), p)
        self.classes = sorted({d["cls"] for fr in self._frames.values() for d in fr})
        self._T_fveh_cam = sensor_T_fveh_cam()
        self._T_cam_fveh = np.linalg.inv(self._T_fveh_cam)

    def _merge(self, data: dict, src: Path) -> None:
        if self.K is None:                       # first artifact fixes the header
            self.K = np.asarray(data["K"], dtype=float)
            self.image_size = tuple(data.get("image_size", (0, 0)))
            self.camera = data.get("camera", "")
            self.schema = data.get("schema", "")
        elif not np.allclose(self.K, np.asarray(data["K"], dtype=float)):
            raise ValueError(f"intrinsics mismatch merging {src} into the sensor")
        self.n_frames = max(self.n_frames, int(data.get("n_frames", 0)))
        for t, dets in data.get("frames", {}).items():
            self._frames.setdefault(t, []).extend(dets)

    @classmethod
    def from_map(cls, name: str, run_id: str | None = None,
                 maps_root: Path | None = None,
                 noise: PixelNoise | None = None) -> "VirtualCameraSensor":
        """Find + merge a map run's per-frame sensor artifacts.

        ``run_id`` defaults to the run that produced the map (``map_adas.json``'s
        ``provenance.run_id``), mirroring ``render_mapgen_overlay.resolve_run`` — so
        the sensor stream is the one that built the live DB.
        """
        layout = MapLayout(name) if maps_root is None else MapLayout(name, maps_root)
        if run_id is None:
            db = json.loads(layout.db.read_text())
            run_id = (db.get("provenance") or {}).get("run_id")
            if not run_id:
                raise ValueError(
                    f"map '{name}' has no provenance.run_id; pass run_id explicitly")
        run = layout.run(run_id)
        paths = [run.step_out(role) / fn for role, fn in ARTIFACTS
                 if (run.step_out(role) / fn).exists()]
        if not paths:
            raise FileNotFoundError(
                f"no virtual-sensor artifacts in run {run_id} — re-run the objects / "
                f"lanes / markings steps to produce the *_by_frame.json files")
        return cls(*paths, noise=noise)

    def detections(self, frame: int) -> list[MapObject]:
        """Static-object detections the camera saw at ``frame`` (camera coords).

        With a :class:`PixelNoise` model configured, each detection is jittered
        (mean + covariance) by a per-frame-deterministic realization of the noise.
        """
        objs = [MapObject.from_dict(d) for d in self._frames.get(str(frame), [])]
        if self.noise is None:
            return objs
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        rng = np.random.default_rng([self.noise.seed, int(frame)])
        kw = dict(fx=fx, fy=fy, T_cam_fveh=self._T_cam_fveh, T_fveh_cam=self._T_fveh_cam,
                  sigma_px=self.noise.sigma_px, rng=rng,
                  perturb=self.noise.perturb_mean, inflate=self.noise.inflate_cov)
        out = []
        for o in objs:
            o.geometry = {name: _apply_pixel_noise(g, **kw) for name, g in o.geometry.items()}
            out.append(o)
        return out

    @staticmethod
    def center(obj: MapObject) -> np.ndarray:
        """Representative 3D point (camera coords) of a detection — same accessor the
        map uses for proximity queries."""
        return _representative_point(obj, None)

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self):
        """Yield ``(frame, detections)`` for every frame in order."""
        for t in range(self.n_frames):
            yield t, self.detections(t)

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._frames.values())
        return (f"VirtualCameraSensor(camera={self.camera!r}, n_frames={self.n_frames}, "
                f"detections={total}, classes={self.classes}, sources={len(self.paths)})")
