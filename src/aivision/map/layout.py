"""On-disk layout: stateless generator vs self-contained map, with per-run dirs.

``mapgen/<gen>/`` — the STATELESS, map-instance-less generator: everything
needed to run any map and nothing tied to one::

    mapgen/<gen>/
      config.json     recipe + step params + bindata download metadata (tracked)
      bindata/        models + binaries (reusable deps, downloadable; gitignored)

``map/<map>/`` — the MAP: self-contained in its own state + result::

    map/<map>/
      map_adas.json                              the result (TRACKED; carries BuildManifest)
      bindata/                                   heavy local state (gitignored):
        input/                                   input dataset slice
        mapgen/<gen>-<timestamp>/                one generation run's byproducts:
          map-gen/<role>/{step.json, out/, work/}    provenance + contracts + intermediates
          logs/                                  run logs
      output/                                    rendered media (gitignored)

Only ``map_adas.json`` (+ the generator's ``config.json``) is git-tracked;
everything under ``bindata/`` and ``output/`` is local. :class:`MapLayout`
resolves map-scoped paths; :class:`MapRun` resolves run-scoped paths within one
generation. This module is the single source of truth so nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAPS_ROOT = Path("map")
MAPGEN_ROOT = Path("mapgen")


@dataclass(frozen=True)
class MapLayout:
    """Map-scoped paths for ``map/<name>/``."""

    name: str
    maps_root: Path = MAPS_ROOT

    @property
    def root(self) -> Path:
        return self.maps_root / self.name

    @property
    def db(self) -> Path:
        return self.root / "map_adas.json"

    @property
    def gt(self) -> Path:
        return self.root / "map_adas.gt.json"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def bindata(self) -> Path:
        return self.root / "bindata"

    @property
    def input(self) -> Path:
        """The input dataset slice (map state)."""
        return self.bindata / "input"

    @property
    def runs_root(self) -> Path:
        """Where per-generation run byproducts live."""
        return self.bindata / "mapgen"

    def run(self, run_id: str) -> "MapRun":
        return MapRun(self, run_id)

    def ensure(self) -> "MapLayout":
        for d in (self.output, self.input, self.runs_root):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class MapRun:
    """Run-scoped paths for one generation: ``map/<name>/bindata/mapgen/<run_id>/``."""

    layout: MapLayout
    run_id: str  # "<gen>-<timestamp>"

    @property
    def root(self) -> Path:
        return self.layout.runs_root / self.run_id

    @property
    def mapgen(self) -> Path:
        return self.root / "map-gen"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def step(self, role: str) -> Path:
        return self.mapgen / role

    def step_record(self, role: str) -> Path:
        return self.step(role) / "step.json"

    def step_out(self, role: str) -> Path:
        return self.step(role) / "out"

    def step_work(self, role: str) -> Path:
        return self.step(role) / "work"

    def ensure(self) -> "MapRun":
        self.logs.mkdir(parents=True, exist_ok=True)
        self.mapgen.mkdir(parents=True, exist_ok=True)
        return self

    def ensure_step(self, role: str) -> "MapRun":
        for d in (self.step_out(role), self.step_work(role)):
            d.mkdir(parents=True, exist_ok=True)
        return self
