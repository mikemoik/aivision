"""ADAS map: a belief store of static, ADAS-relevant environment objects.

The map holds the current best estimate + uncertainty of each object (poles,
signs, traffic lights, guide posts, lane markings, road arrows, ...) in a
constant Euclidean frame ``M``, updatable as new data arrives but retaining
no observations. Dynamic objects, ego trajectory, depth, segmentation and
point clouds are out of scope.
"""

from __future__ import annotations

from .frame import FrameInfo, GeoReference
from .geometry import (
    GaussianPoint,
    GaussianPose,
    Geometry,
    LineSegment,
    OrientedBox,
    Polyline,
)
from .layout import MapLayout, MapRun
from .map_adas import MapAdas
from .mapgen import MapGen, MapGenContext, MapGenStep, STEP_REGISTRY
from .provenance import BuildManifest, StepRecord
from .objects import (
    GuidePost,
    LaneMarking,
    MapClass,
    MapObject,
    Pole,
    RoadArrow,
    TrafficLight,
    TrafficSign,
)

__all__ = [
    "MapAdas",
    "MapLayout",
    "MapRun",
    "MapGen",
    "MapGenStep",
    "MapGenContext",
    "STEP_REGISTRY",
    "BuildManifest",
    "StepRecord",
    "FrameInfo",
    "GeoReference",
    "Geometry",
    "GaussianPoint",
    "GaussianPose",
    "LineSegment",
    "Polyline",
    "OrientedBox",
    "MapClass",
    "MapObject",
    "Pole",
    "GuidePost",
    "TrafficSign",
    "TrafficLight",
    "LaneMarking",
    "RoadArrow",
]
