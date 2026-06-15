"""MapGen — the JSON-configured, stateless ADAS-map generator.

``MapGen(instance_dir)`` loads ``mapgen/<gen>/config.json`` and ``run()``s the
role DAG, writing all byproducts into a fresh per-run dir inside the *product*
map (``map/<map>/bindata/mapgen/<gen>-<timestamp>/``) and the final
``map_adas.json`` at the map root. The generator itself stays stateless: it
holds only config + ``bindata/`` deps (models, binaries), fetched on demand.

Every step is a :class:`MapGenStep` that standardizes the lifecycle (ensure
dirs → stamp :class:`StepRecord` → run method → write ``step.json``); only
:meth:`MapGenStep.process` is method-specific. ``(role, type)`` selects the
implementation from :data:`STEP_REGISTRY`, so methods are swappable behind a
role's output contract. This module holds the framework + the pure-python steps
(``georef``, ``assemble``); heavy steps live elsewhere so loading never needs
torch/ORB.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .frame import FrameInfo, GeoReference
from .ingest import load_arrows, load_lane_polylines, load_oxts_enu, load_poles_signs
from .layout import MAPGEN_ROOT, MAPS_ROOT, MapLayout, MapRun
from .map_adas import MapAdas
from .provenance import BuildManifest, StepRecord

STEP_REGISTRY: dict[tuple[str, str], type["MapGenStep"]] = {}

# Heavy steps live in their own modules so importing this framework never drags
# in cv2 / torch / the ORB toolchain. They register on import; we import them
# lazily at run() time, tolerating a missing backend (e.g. torch) so a run that
# doesn't select that role still works.
_STEP_MODULES = ("steps_slam", "steps_depth", "steps_seg", "steps_lanes",
                 "steps_objects", "steps_markings")


def register_step(cls: type["MapGenStep"]) -> type["MapGenStep"]:
    STEP_REGISTRY[(cls.role, cls.type)] = cls
    return cls


def _load_step_modules(log: logging.Logger) -> None:
    for mod in _STEP_MODULES:
        try:
            importlib.import_module(f".{mod}", __package__)
        except Exception as exc:  # missing backend → role stays unregistered
            log.warning("step module %s not loaded: %s", mod, exc)


@dataclass
class MapGenContext:
    """Handed to every step: instance (deps), map + run (state), manifest, logger."""

    instance_dir: Path  # mapgen/<gen> — generator deps
    layout: MapLayout   # map/<map>    — map-scoped state/result
    run: MapRun         # map/<map>/bindata/mapgen/<run_id> — this run's byproducts
    config: dict
    manifest: BuildManifest
    log: logging.Logger

    def input(self, name: str) -> Path:
        """A map-state input path declared in config['input']."""
        return self.layout.root / self.config["input"][name]

    def bindata(self, *parts: str) -> Path:
        """A generator dep path under mapgen/<gen>/bindata/."""
        return self.instance_dir / "bindata" / Path(*parts)

    def upstream_out(self, role: str) -> Path:
        return self.run.step_out(role)

    def out(self, role: str) -> Path:
        return self.run.step_out(role)

    def work(self, role: str) -> Path:
        return self.run.step_work(role)


class MapGenStep:
    """Base step: standardizes the run lifecycle; subclasses implement process()."""

    role: str
    type: str
    inputs: list[str] = []

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def process(self, ctx: MapGenContext) -> list[str]:
        raise NotImplementedError

    def execute(self, ctx: MapGenContext) -> StepRecord:
        ctx.run.ensure_step(self.role)
        ctx.log.info("▶ %s (type=%s)", self.role, self.type)
        t0 = time.perf_counter()
        rec = StepRecord(role=self.role, type=self.type, params=self.params,
                         inputs=list(self.inputs))
        ctx.manifest.add(rec)  # before process so assemble sees the full DAG
        rec.outputs = self.process(ctx) or []
        ctx.run.step_record(self.role).write_text(json.dumps(rec.to_dict(), indent=2))
        ctx.log.info("✓ %s done in %.1fs → %s", self.role, time.perf_counter() - t0, rec.outputs)
        return rec


@register_step
class GeorefStep(MapGenStep):
    """Fit map(M) -> ENU from the SLAM trajectory vs GNSS (rigid Umeyama)."""

    role = "georef"
    type = "umeyama+gnss"
    inputs = ["slam", "input:oxts"]

    def process(self, ctx: MapGenContext) -> list[str]:
        traj = json.loads((ctx.upstream_out("slam") / "trajectory.json").read_text())
        cam_centres = np.asarray(traj["poses"], dtype=float)[:, [3, 7, 11]]
        enu, datum = load_oxts_enu(ctx.input("oxts") / "data", len(cam_centres))
        offset = np.asarray(self.params.get("enu_offset_m", [0.0, 0.0, 0.0]), dtype=float)
        gr = GeoReference.from_point_correspondences(
            cam_centres, enu, datum,
            with_scale=self.params.get("with_scale", True), enu_offset=offset,
        )
        (ctx.out("georef") / "georef.json").write_text(json.dumps(gr.to_dict(), indent=2))
        return ["map-gen/georef/out/georef.json"]


@register_step
class AssembleStep(MapGenStep):
    """Ingest the upstream contracts into the final MapAdas + provenance."""

    role = "assemble"
    type = "ingest-v1"
    inputs = ["lanes", "objects", "markings", "georef"]

    def process(self, ctx: MapGenContext) -> list[str]:
        objs = []
        objs += load_poles_signs(ctx.upstream_out("objects") / "poles_signs_world.json")
        objs += load_arrows(ctx.upstream_out("markings") / "arrows_world.json")
        # The map's lane belief is the FILTERED coupled lane model (RANSAC border
        # reject + Whittaker smooth, consistent width) — the same representation
        # the downstream consumers use — NOT the raw per-Z-bin border polylines
        # (lane_polylines.json), which remain available as a lanes-role artifact.
        objs += load_lane_polylines(ctx.upstream_out("lanes") / "lane_polylines_model.json")
        gr = GeoReference.from_dict(
            json.loads((ctx.upstream_out("georef") / "georef.json").read_text()))
        frame = FrameInfo(name="M", provenance=ctx.config.get("frame", {}).get("provenance", ""))
        m = MapAdas(frame=frame, georef=gr, provenance=ctx.manifest)
        for o in objs:
            m.add(o)
        m.save(ctx.layout.db)
        return ["map_adas.json"]


class MapGen:
    """Runs a map's generation DAG from a stateless generator instance."""

    def __init__(self, instance_dir: str | Path, maps_root: Path = MAPS_ROOT):
        self.instance_dir = Path(instance_dir)
        self.config = json.loads((self.instance_dir / "config.json").read_text())
        self.layout = MapLayout(self.config["map_name"], maps_root)

    @classmethod
    def from_name(cls, gen_name: str, mapgen_root: Path = MAPGEN_ROOT,
                  maps_root: Path = MAPS_ROOT) -> "MapGen":
        return cls(mapgen_root / gen_name, maps_root)

    def run(self, only: list[str] | None = None, force: bool = False,
            run_id: str | None = None) -> MapAdas:
        """Execute the configured steps into a per-run dir; return the map.

        A new ``run_id`` (``<gen>-<timestamp>``) is created unless one is given
        (resume). A step runs if registered, selected (``only``) and either
        ``force`` or its ``out/`` is empty; otherwise its existing ``step.json``
        is folded into the manifest (passthrough).
        """
        run_id = run_id or f"{self.instance_dir.name}-{datetime.now():%Y%m%dT%H%M%S}"
        run = self.layout.run(run_id)
        self.layout.ensure()
        run.ensure()
        log = self._setup_logging(run)
        log.info("MapGen '%s' → map '%s' run=%s", self.instance_dir.name,
                 self.config["map_name"], run_id)
        _load_step_modules(log)
        self._ensure_bindata(log)
        manifest = BuildManifest(run_id=run_id)
        ctx = MapGenContext(self.instance_dir, self.layout, run, self.config, manifest, log)
        for sc in self.config["steps"]:
            role, mtype = sc["role"], sc["type"]
            key = (role, mtype)
            out_dir = run.step_out(role)
            has_out = out_dir.exists() and any(out_dir.iterdir())
            selected = only is None or role in only
            # Fail loud instead of silently passing through: if a step is
            # explicitly selected+forced but its (role, type) has no registered
            # implementation, the config type likely disagrees with the class.
            if selected and force and key not in STEP_REGISTRY:
                raise KeyError(
                    f"step ({role}, {mtype}) selected for a forced run but no "
                    f"registered implementation; registered={sorted(STEP_REGISTRY)}")
            if key in STEP_REGISTRY and selected and (force or not has_out):
                STEP_REGISTRY[key](sc.get("params", {})).execute(ctx)
            else:
                self._passthrough(ctx, role)
        log.info("MapGen done: %d steps, map → %s", len(manifest.steps), self.layout.db)
        return MapAdas.load(self.layout.db)

    def _passthrough(self, ctx: MapGenContext, role: str) -> None:
        rec_path = ctx.run.step_record(role)
        if rec_path.exists():
            ctx.manifest.add(StepRecord.from_dict(json.loads(rec_path.read_text())))
            ctx.log.info("• %s passthrough (existing step.json)", role)

    def _setup_logging(self, run: MapRun) -> logging.Logger:
        log = logging.getLogger(f"mapgen.{self.config['map_name']}.{run.run_id}")
        log.setLevel(logging.INFO)
        log.handlers.clear()
        log.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.FileHandler(run.logs / "run.log"), logging.StreamHandler()):
            handler.setFormatter(fmt)
            log.addHandler(handler)
        return log

    def _ensure_bindata(self, log: logging.Logger) -> None:
        for asset in self.config.get("bindata", []):
            dest = self.instance_dir / "bindata" / asset["dest"]
            if dest.exists():
                log.info("bindata present: %s (%s)", asset["name"], asset["dest"])
            elif asset.get("url"):
                self._fetch(asset, dest, log)
            else:
                log.warning("bindata MISSING (no url): %s — %s",
                            asset["name"], asset.get("note", ""))

    def _fetch(self, asset: dict, dest: Path, log: logging.Logger) -> None:
        import hashlib
        import urllib.request

        log.info("bindata fetch: %s <- %s", asset["name"], asset["url"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(asset["url"], tmp)
        if asset.get("sha256"):
            got = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if got != asset["sha256"]:
                raise ValueError(f"sha256 mismatch for {asset['name']}: {got}")
        tmp.rename(dest)
