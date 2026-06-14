"""The ADAS map container and its on-disk DB.

:class:`MapAdas` is the belief store: a collection of static ADAS objects in
the constant map frame ``M``, plus the georeferencing transform needed to put
them on the Earth. It is written by detectors/estimators (online per-object
fusion, stubbed for now) and read by consumers (GTSAM batch refinement,
viz, georef export). It deliberately holds no observations, trajectory,
depth, segmentation, or point clouds.

The DB is a single human-diffable JSON file (object counts are tiny). The
schema is designed to migrate to SQLite/GeoPackage later without touching
the object model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .frame import FrameInfo, GeoReference
from .geometry import Geometry
from .objects import MapClass, MapObject

SCHEMA_VERSION = 1


class MapAdas:
    """In-memory ADAS map with JSON persistence."""

    def __init__(self, frame: FrameInfo | None = None, georef: GeoReference | None = None):
        self.frame = frame or FrameInfo()
        self.georef = georef
        self.objects: dict[str, MapObject] = {}

    # -- mutation -------------------------------------------------------
    def add(self, obj: MapObject) -> str:
        if obj.id in self.objects:
            raise KeyError(f"object id already present: {obj.id}")
        self.objects[obj.id] = obj
        return obj.id

    def get(self, obj_id: str) -> MapObject:
        return self.objects[obj_id]

    def remove(self, obj_id: str) -> None:
        del self.objects[obj_id]

    def update(self, obj_id: str, slot: str, measurement: Geometry, frame: int | None = None) -> None:
        """Fold a new measurement into one object's geometry slot."""
        self.objects[obj_id].update(slot, measurement, frame=frame)

    # -- query ----------------------------------------------------------
    def of_class(self, cls: MapClass) -> list[MapObject]:
        return [o for o in self.objects.values() if o.cls == cls]

    def query(
        self,
        cls: MapClass | None = None,
        near: np.ndarray | None = None,
        radius: float | None = None,
        anchor_slot: str | None = None,
    ) -> list[MapObject]:
        """Filter objects by class and/or proximity to a point in frame ``M``.

        Proximity uses the first geometry slot's representative point unless
        ``anchor_slot`` names one explicitly.
        """
        out = list(self.objects.values())
        if cls is not None:
            out = [o for o in out if o.cls == cls]
        if near is not None and radius is not None:
            near = np.asarray(near, dtype=float)
            out = [o for o in out if _representative_point(o, anchor_slot) is not None
                   and np.linalg.norm(_representative_point(o, anchor_slot) - near) <= radius]
        return out

    # -- georef passthrough --------------------------------------------
    def to_enu(self, p_M: np.ndarray) -> np.ndarray:
        if self.georef is None:
            raise RuntimeError("map has no georeference")
        return self.georef.to_enu(p_M)

    def to_wgs84(self, p_M: np.ndarray) -> np.ndarray:
        if self.georef is None:
            raise RuntimeError("map has no georeference")
        return self.georef.to_wgs84(p_M)

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "frame": self.frame.to_dict(),
            "georef": self.georef.to_dict() if self.georef else None,
            "objects": [o.to_dict() for o in self.objects.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapAdas":
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {d.get('schema_version')}")
        m = cls(
            frame=FrameInfo.from_dict(d["frame"]),
            georef=GeoReference.from_dict(d["georef"]) if d.get("georef") else None,
        )
        for od in d["objects"]:
            m.add(MapObject.from_dict(od))
        return m

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MapAdas":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __len__(self) -> int:
        return len(self.objects)

    def __repr__(self) -> str:
        counts = {c.value: len(self.of_class(c)) for c in MapClass}
        counts = {k: v for k, v in counts.items() if v}
        return f"MapAdas(frame={self.frame.name!r}, n={len(self)}, {counts}, georef={self.georef is not None})"


def _representative_point(obj: MapObject, anchor_slot: str | None) -> np.ndarray | None:
    """A single point standing in for an object's location, for proximity."""
    from .geometry import GaussianPoint, GaussianPose, LineSegment, OrientedBox, Polyline

    slot = anchor_slot or next(iter(obj.geometry))
    g = obj.geometry[slot]
    if isinstance(g, GaussianPoint):
        return g.mean
    if isinstance(g, GaussianPose):
        return g.translation
    if isinstance(g, OrientedBox):
        return g.pose.translation
    if isinstance(g, LineSegment):
        return 0.5 * (g.a.mean + g.b.mean)
    if isinstance(g, Polyline) and g.verts:
        return np.mean([v.mean for v in g.verts], axis=0)
    return None
