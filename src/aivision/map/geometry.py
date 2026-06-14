"""Reusable geometry primitives for the ADAS map.

Every primitive carries its own Gaussian uncertainty and is the shared
vocabulary that map objects compose from (a pole is a :class:`LineSegment`,
a lane marking is a :class:`Polyline`, a sign is a :class:`GaussianPose`
plus an :class:`OrientedBox`, ...). Reuse happens here, at the primitive
level; structure varies at the object level (see :mod:`aivision.map.objects`).

All coordinates are expressed in the constant Euclidean map frame ``M``
(metres). Poses are stored as 4x4 homogeneous matrices ``T_M_obj`` with a
6x6 covariance in the se(3) tangent space (translation-then-rotation order).

Fusion is stubbed in v1: :meth:`fuse` records the latest measurement and
bumps bookkeeping but does not yet run the information-form update. The
interface is in place so the recursive estimator (or a GTSAM batch pass)
can be wired in without touching callers. Search ``TODO(fuse)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation

Array = np.ndarray


def _arr(x: Any, shape: tuple[int, ...]) -> Array:
    a = np.asarray(x, dtype=float)
    if a.shape != shape:
        raise ValueError(f"expected shape {shape}, got {a.shape}")
    return a


class Geometry:
    """Base class: serialization dispatch + the (stubbed) fuse contract."""

    kind: ClassVar[str] = "geometry"

    def to_dict(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return _GEOMETRY_REGISTRY[d["kind"]]._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> "Geometry":  # pragma: no cover - overridden
        raise NotImplementedError

    def fuse(self, measurement: "Geometry") -> None:
        """Fold a new measurement of this primitive into the belief.

        TODO(fuse): replace the latest-wins stub with the information-form
        update (Λ' = Λ + R⁻¹, η' = η + R⁻¹z; poses in se(3) tangent space).
        """
        raise NotImplementedError


@dataclass
class GaussianPoint(Geometry):
    """A 3D point with full 3x3 covariance."""

    kind: ClassVar[str] = "point"
    mean: Array  # (3,)
    cov: Array = field(default_factory=lambda: np.eye(3))  # (3, 3)

    def __post_init__(self) -> None:
        self.mean = _arr(self.mean, (3,))
        self.cov = _arr(self.cov, (3, 3))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "mean": self.mean.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def _from_dict(cls, d: dict) -> "GaussianPoint":
        return cls(mean=d["mean"], cov=d["cov"])

    def fuse(self, measurement: "GaussianPoint") -> None:
        # TODO(fuse): information-form combine. Stub: latest measurement wins.
        self.mean = measurement.mean.copy()
        self.cov = measurement.cov.copy()


@dataclass
class GaussianPose(Geometry):
    """An SE(3) pose ``T_M_obj`` (4x4) with a 6x6 se(3) tangent covariance."""

    kind: ClassVar[str] = "pose"
    T: Array  # (4, 4)
    cov: Array = field(default_factory=lambda: np.eye(6))  # (6, 6) [t, rot]

    def __post_init__(self) -> None:
        self.T = _arr(self.T, (4, 4))
        self.cov = _arr(self.cov, (6, 6))

    @classmethod
    def from_Rt(cls, R: Array, t: Array, cov: Array | None = None) -> "GaussianPose":
        T = np.eye(4)
        T[:3, :3] = _arr(R, (3, 3))
        T[:3, 3] = _arr(t, (3,))
        return cls(T=T, cov=np.eye(6) if cov is None else cov)

    @property
    def translation(self) -> Array:
        return self.T[:3, 3]

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_matrix(self.T[:3, :3])

    def to_dict(self) -> dict:
        return {"kind": self.kind, "T": self.T.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def _from_dict(cls, d: dict) -> "GaussianPose":
        return cls(T=d["T"], cov=d["cov"])

    def fuse(self, measurement: "GaussianPose") -> None:
        # TODO(fuse): tangent-space information-form combine. Stub: latest wins.
        self.T = measurement.T.copy()
        self.cov = measurement.cov.copy()


@dataclass
class LineSegment(Geometry):
    """A segment between two Gaussian endpoints (poles, guide posts)."""

    kind: ClassVar[str] = "line_segment"
    a: GaussianPoint
    b: GaussianPoint

    def to_dict(self) -> dict:
        return {"kind": self.kind, "a": self.a.to_dict(), "b": self.b.to_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> "LineSegment":
        return cls(a=GaussianPoint.from_dict(d["a"]), b=GaussianPoint.from_dict(d["b"]))

    def fuse(self, measurement: "LineSegment") -> None:
        self.a.fuse(measurement.a)
        self.b.fuse(measurement.b)


@dataclass
class Polyline(Geometry):
    """An ordered chain of Gaussian vertices; ``closed=True`` is a polygon.

    Per-vertex covariance (decision: richer than a single pooled cov).
    """

    kind: ClassVar[str] = "polyline"
    verts: list[GaussianPoint]
    closed: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "closed": self.closed,
            "verts": [v.to_dict() for v in self.verts],
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "Polyline":
        return cls(
            verts=[GaussianPoint.from_dict(v) for v in d["verts"]],
            closed=bool(d.get("closed", False)),
        )

    def fuse(self, measurement: "Polyline") -> None:
        # TODO(fuse): vertex association is non-trivial for polylines; stub
        # assumes matched vertex counts and fuses pairwise.
        if len(self.verts) != len(measurement.verts):
            self.verts = [GaussianPoint(v.mean, v.cov) for v in measurement.verts]
            return
        for v, m in zip(self.verts, measurement.verts):
            v.fuse(m)


@dataclass
class OrientedBox(Geometry):
    """A box: center pose ``T_M_box`` + 3 half-extents with their covariance."""

    kind: ClassVar[str] = "oriented_box"
    pose: GaussianPose
    extents: Array  # (3,) half-extents
    extents_cov: Array = field(default_factory=lambda: np.eye(3))  # (3, 3)

    def __post_init__(self) -> None:
        self.extents = _arr(self.extents, (3,))
        self.extents_cov = _arr(self.extents_cov, (3, 3))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "pose": self.pose.to_dict(),
            "extents": self.extents.tolist(),
            "extents_cov": self.extents_cov.tolist(),
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "OrientedBox":
        return cls(
            pose=GaussianPose.from_dict(d["pose"]),
            extents=d["extents"],
            extents_cov=d["extents_cov"],
        )

    def fuse(self, measurement: "OrientedBox") -> None:
        self.pose.fuse(measurement.pose)
        # TODO(fuse): information-form on extents. Stub: latest wins.
        self.extents = measurement.extents.copy()
        self.extents_cov = measurement.extents_cov.copy()


_GEOMETRY_REGISTRY: dict[str, type[Geometry]] = {
    g.kind: g
    for g in (GaussianPoint, GaussianPose, LineSegment, Polyline, OrientedBox)
}
