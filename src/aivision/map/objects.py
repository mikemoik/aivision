"""ADAS map objects.

A :class:`MapObject` is the unit of the map's belief store: it holds the
*current best estimate + uncertainty* of one static, ADAS-relevant
environment object, never the observations that produced it. Geometry is
composed from the reusable primitives in :mod:`aivision.map.geometry`;
each subclass declares its named geometry ``SLOTS`` (their "set of data
structures"), so serialization stays generic while structure varies by class.

Out of scope by design: dynamic objects, ego trajectory, depth, segmentation,
point clouds (see the ADAS-map architecture notes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from .geometry import (
    Geometry,
    GaussianPoint,
    GaussianPose,
    LineSegment,
    OrientedBox,
    Polyline,
)


class MapClass(str, Enum):
    POLE = "pole"
    SIGN = "sign"
    TRAFFIC_LIGHT = "traffic_light"
    GUIDE_POST = "guide_post"
    LANE_MARKING = "lane_marking"
    ROAD_ARROW = "road_arrow"


@dataclass
class MapObject:
    """Base map object: id, semantic class, geometry slots, bookkeeping.

    ``geometry`` maps slot name -> primitive. Subclasses fix ``cls`` and the
    expected ``SLOTS`` spec ``{name: PrimitiveType}`` used for validation and
    convenience accessors.
    """

    cls: ClassVar[MapClass]
    SLOTS: ClassVar[dict[str, type[Geometry]]] = {}

    id: str
    geometry: dict[str, Geometry]
    attributes: dict[str, Any] = field(default_factory=dict)
    n_obs: int = 1
    created_frame: int | None = None
    last_update_frame: int | None = None

    def __post_init__(self) -> None:
        missing = set(self.SLOTS) - set(self.geometry)
        if missing:
            raise ValueError(f"{type(self).__name__} missing geometry slots: {missing}")
        for name, prim in self.geometry.items():
            want = self.SLOTS.get(name)
            if want and not isinstance(prim, want):
                raise TypeError(f"slot '{name}' expects {want.__name__}, got {type(prim).__name__}")

    def update(self, slot: str, measurement: Geometry, frame: int | None = None) -> None:
        """Fold a new measurement into one geometry slot (fusion stubbed)."""
        self.geometry[slot].fuse(measurement)
        self.n_obs += 1
        if frame is not None:
            self.last_update_frame = frame

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cls": self.cls.value,
            "attributes": self.attributes,
            "n_obs": self.n_obs,
            "created_frame": self.created_frame,
            "last_update_frame": self.last_update_frame,
            "geometry": {name: prim.to_dict() for name, prim in self.geometry.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapObject":
        sub = _OBJECT_REGISTRY[MapClass(d["cls"])]
        geometry = {name: Geometry.from_dict(g) for name, g in d["geometry"].items()}
        return sub(
            id=d["id"],
            geometry=geometry,
            attributes=d.get("attributes", {}),
            n_obs=d.get("n_obs", 1),
            created_frame=d.get("created_frame"),
            last_update_frame=d.get("last_update_frame"),
        )


@dataclass
class Pole(MapObject):
    cls: ClassVar[MapClass] = MapClass.POLE
    SLOTS: ClassVar[dict[str, type[Geometry]]] = {"shaft": LineSegment}


@dataclass
class GuidePost(MapObject):
    cls: ClassVar[MapClass] = MapClass.GUIDE_POST
    SLOTS: ClassVar[dict[str, type[Geometry]]] = {"point": GaussianPoint}


@dataclass
class TrafficSign(MapObject):
    cls: ClassVar[MapClass] = MapClass.SIGN
    SLOTS: ClassVar[dict[str, type[Geometry]]] = {"mount": GaussianPose, "face": OrientedBox}


@dataclass
class TrafficLight(MapObject):
    cls: ClassVar[MapClass] = MapClass.TRAFFIC_LIGHT
    SLOTS: ClassVar[dict[str, type[Geometry]]] = {"mount": GaussianPose, "box": OrientedBox}


@dataclass
class LaneMarking(MapObject):
    cls: ClassVar[MapClass] = MapClass.LANE_MARKING
    SLOTS: ClassVar[dict[str, type[Geometry]]] = {"line": Polyline}


@dataclass
class RoadArrow(MapObject):
    cls: ClassVar[MapClass] = MapClass.ROAD_ARROW
    SLOTS: ClassVar[dict[str, type[Geometry]]] = {"pose": GaussianPose, "glyph": Polyline}


_OBJECT_REGISTRY: dict[MapClass, type[MapObject]] = {
    sub.cls: sub
    for sub in (Pole, GuidePost, TrafficSign, TrafficLight, LaneMarking, RoadArrow)
}
