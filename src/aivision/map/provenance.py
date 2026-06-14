"""Provenance for ADAS-map generation: named, swappable steps + a manifest.

Map creation is a DAG of *roles* — stable pipeline stages with fixed I/O
contracts (``slam``, ``depth``, ``segmentation``, ``lanes``, ``objects``,
``markings``, ``georef``, ``assemble``). Each role is filled by a swappable
*method*, recorded as the step's ``type`` (e.g. ``"ORB3"`` for ``slam``). A run
emits a :class:`StepRecord` (type + creation timestamp + params + input refs);
the ordered :class:`BuildManifest` is stored in the map so a ``map_adas.json``
always knows how it was made.

The goal is **traceability, not bit-reproducibility**: a non-deterministic
method like ORB3 is fine as long as we record what produced the artifact and
keep the role's output contract stable so the method stays swappable.

On disk each step owns ``map/<name>/map-gen/<role>/`` with ``step.json`` (this
record), ``out/`` (the output interface artifact), and ``work/`` (method-private
intermediates + debug). See :class:`~aivision.map.layout.MapLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now() -> str:
    """ISO-8601 UTC timestamp for 'created' fields."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StepRecord:
    """Provenance for one map-generation step."""

    role: str  # stable stage, e.g. "slam"
    type: str  # swappable method, e.g. "ORB3"
    created: str = field(default_factory=utc_now)  # ISO-8601 UTC
    params: dict = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)  # upstream role names / refs
    outputs: list[str] = field(default_factory=list)  # artifact paths rel. to map root
    note: str = ""

    def to_dict(self) -> dict:
        d = {"role": self.role, "type": self.type, "created": self.created}
        if self.params:
            d["params"] = self.params
        if self.inputs:
            d["inputs"] = self.inputs
        if self.outputs:
            d["outputs"] = self.outputs
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StepRecord":
        return cls(
            role=d["role"],
            type=d["type"],
            created=d.get("created", ""),
            params=d.get("params", {}),
            inputs=d.get("inputs", []),
            outputs=d.get("outputs", []),
            note=d.get("note", ""),
        )


@dataclass
class BuildManifest:
    """Ordered record of the steps that produced a map, + the run that made it."""

    steps: list[StepRecord] = field(default_factory=list)
    run_id: str = ""  # "<gen>-<timestamp>"; locates the run dir under bindata/mapgen/

    def add(self, rec: StepRecord) -> StepRecord:
        self.steps.append(rec)
        return rec

    def of_role(self, role: str) -> StepRecord | None:
        return next((s for s in self.steps if s.role == role), None)

    def to_dict(self) -> dict:
        d: dict = {}
        if self.run_id:
            d["run_id"] = self.run_id
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BuildManifest":
        return cls(steps=[StepRecord.from_dict(s) for s in d.get("steps", [])],
                   run_id=d.get("run_id", ""))
