"""On-disk layout of a self-contained map directory.

A map owns everything it needs under ``map/<name>/`` so it can be archived or
deleted as one unit, with no external dependencies::

    map/<name>/
      map_adas.json     the map DB (git-tracked)
      output/           rendered media: videos, images (gitignored)
      bindata/          heavy binaries (gitignored)
        input/          copied dataset slice this map was built from
        proc/           processed/derived per-frame data (depth, seg, flow, ...),
                        keyed  proc/<modality>/<method>/<frame>.<ext>

Only ``map_adas.json`` is tracked; ``output/`` and ``bindata/`` contents are
ignored (see .gitignore). This module is the single source of truth for those
paths so scripts never hardcode them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAPS_ROOT = Path("map")


@dataclass(frozen=True)
class MapLayout:
    """Resolves the canonical paths inside one ``map/<name>/`` directory."""

    name: str
    maps_root: Path = MAPS_ROOT

    @property
    def root(self) -> Path:
        return self.maps_root / self.name

    @property
    def db(self) -> Path:
        return self.root / "map_adas.json"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def bindata(self) -> Path:
        return self.root / "bindata"

    @property
    def input(self) -> Path:
        return self.bindata / "input"

    @property
    def proc(self) -> Path:
        return self.bindata / "proc"

    def proc_dir(self, modality: str, method: str) -> Path:
        """Directory for one processed modality+method, e.g. depth/sea_raft."""
        return self.proc / modality / method

    def proc_frame(self, modality: str, method: str, frame: int, ext: str = "npy") -> Path:
        """Per-frame processed file, e.g. proc/depth/sea_raft/000123.npy."""
        return self.proc_dir(modality, method) / f"{frame:06d}.{ext}"

    def ensure(self) -> "MapLayout":
        """Create the standard subdirectories if missing."""
        for d in (self.output, self.input, self.proc):
            d.mkdir(parents=True, exist_ok=True)
        return self
