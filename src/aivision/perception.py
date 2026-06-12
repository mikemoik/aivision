"""Per-frame perception cache (Phase A of the round-3 perception-first SLAM plan).

Produces one `.npz` per kept frame plus a `manifest.json` at the cache root.
A1 lays the cache layout and CLI surface. A2 (panoptic segmentation) and A3
(mono depth) fill in the per-pixel arrays through the `segment_frame` /
`mono_depth_frame` hooks below.

Per-frame schema (`frame_<NNNNNN>.npz`, all optional; absence means
"not populated yet by the relevant phase"):

    semantic     uint8   H×W   Cityscapes-style class ids (255 = unknown)
    instance     uint16  H×W   per-class instance ids (0 = stuff / no instance)
    mono_depth   float16 H×W   metric depth in metres (Apple Depth Pro)
    depth_conf   float16 H×W   per-pixel depth confidence in [0, 1]

The manifest is the single source of truth for which fields *should* exist —
downstream stages should consult `manifest["populated"]` before assuming a
field is present.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import typer
from tqdm import tqdm

from .io import iter_frames


CACHE_VERSION = 1
MANIFEST_NAME = "manifest.json"
PERCEPTION_FIELDS: tuple[str, ...] = ("semantic", "instance", "mono_depth", "depth_conf")

# Mask2Former-Swin-L fine-tuned on Mapillary Vistas panoptic (65 classes:
# 28 stuff + 37 things). Vistas is geographically/lighting-diverse (~100
# countries) and labels finer dashcam-relevant classes than Cityscapes:
# car / truck / bus / motorcycle / bicycle / on-rails / trailer / caravan,
# plus lane markings, crosswalks, road dividers, and traffic-sign subtypes.
# 200M params — ~3× the Cityscapes-Small variant but worth it; computation
# time is not a constraint for this offline batch pipeline.
SEG_MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"

# Max-quality input resolution: 384×384 (the processor default) is the
# eval-time scale for cheap inference and visibly throws away detail. The
# Mask2Former Vistas recipe trains at ~1024-pixel crops, so feeding source
# frames upscaled to a 1024 shortest-edge (cap 2048 to keep 16:9 sane)
# matches the model's training-time scale and preserves small classes
# (street lights, lane markings, distant signs) that we want for Phase B's
# semantic filter.
SEG_INPUT_SIZE: dict[str, int] = {"shortest_edge": 1024, "longest_edge": 2048}

# Canonical Mapillary Vistas v1.2 "stuff" classes (instances=False in the
# dataset spec): everything that is not animal / human / movable-object /
# vehicle. `_get_seg_model` resolves these names against the loaded model's
# id2label and stores the integer ids on the cached state.
_VISTAS_STUFF_LABEL_NAMES: frozenset[str] = frozenset({
    "Curb", "Fence", "Guard Rail", "Barrier", "Wall",
    "Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking",
    "Pedestrian Area", "Rail Track", "Road", "Service Lane", "Sidewalk",
    "Bridge", "Building", "Tunnel",
    "Lane Marking - Crosswalk", "Lane Marking - General",
    "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water",
    "Car Mount", "Ego Vehicle",
})

# Apple Depth Pro — metric monocular depth (HuggingFace integration).
# Returns "predicted_depth" in metres at the source resolution, plus a
# "field_of_view" / "focal_length" prediction (unused here, but Phase C may
# feed it into pycolmap SIMPLE_RADIAL initialisation). Phase C still
# co-optimises a per-frame (scale, shift): metric scale on dashcam is not
# fully trustworthy, so we treat Depth Pro as a tight-but-soft prior.
DEPTH_MODEL_ID = "apple/DepthPro-hf"


@dataclass
class CacheManifest:
    """Top-level descriptor of a perception cache."""

    version: int
    source: str
    step: int
    n_frames: int
    height: int
    width: int
    populated: dict[str, bool] = field(default_factory=lambda: {k: False for k in PERCEPTION_FIELDS})
    seg_model: str | None = None
    depth_model: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls, path: Path) -> "CacheManifest":
        return cls(**json.loads(Path(path).read_text()))


def _frame_npz_path(cache_dir: Path, frame_idx: int) -> Path:
    return cache_dir / f"frame_{frame_idx:06d}.npz"


def write_frame_entry(
    cache_dir: Path,
    frame_idx: int,
    *,
    semantic: np.ndarray | None = None,
    instance: np.ndarray | None = None,
    mono_depth: np.ndarray | None = None,
    depth_conf: np.ndarray | None = None,
) -> Path:
    """Write or update a per-frame perception .npz.

    Fields left None are not touched: A2 can populate {semantic, instance}
    and A3 can later add {mono_depth, depth_conf} to the same file without
    clobbering each other.
    """
    out_path = _frame_npz_path(cache_dir, frame_idx)
    payload: dict[str, np.ndarray] = {}
    if out_path.exists():
        with np.load(out_path) as old:
            payload = {k: old[k] for k in old.files}

    if semantic is not None:
        payload["semantic"] = semantic.astype(np.uint8, copy=False)
    if instance is not None:
        payload["instance"] = instance.astype(np.uint16, copy=False)
    if mono_depth is not None:
        payload["mono_depth"] = mono_depth.astype(np.float16, copy=False)
    if depth_conf is not None:
        payload["depth_conf"] = depth_conf.astype(np.float16, copy=False)

    np.savez_compressed(out_path, **payload)
    return out_path


def load_frame_entry(cache_dir: Path, frame_idx: int) -> dict[str, np.ndarray]:
    with np.load(_frame_npz_path(cache_dir, frame_idx)) as f:
        return {k: f[k] for k in f.files}


_SEG_STATE: dict[str, object] = {}


def _get_seg_model():
    """Lazy-load Mask2Former-Swin-L Vistas panoptic; cache for the process.

    Resolves the canonical Vistas stuff-class names against the loaded
    model's `id2label` so we don't hardcode integer ids (and so a stale
    name set surfaces as a startup warning rather than as silent
    miscategorisation in the seg cache).
    """
    if "model" in _SEG_STATE:
        return _SEG_STATE

    import torch
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Mask2FormerImageProcessor.from_pretrained(SEG_MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL_ID).to(device).eval()

    id2label = model.config.id2label
    name_to_id = {v: int(k) for k, v in id2label.items()}
    stuff_ids = sorted(name_to_id[n] for n in _VISTAS_STUFF_LABEL_NAMES if n in name_to_id)
    missing = [n for n in _VISTAS_STUFF_LABEL_NAMES if n not in name_to_id]
    if missing:
        typer.echo(
            f"[perception] WARNING: Vistas stuff names not in model id2label: {missing}. "
            "Stuff fusion will skip them — `instance` field may contain spurious counters."
        )

    _SEG_STATE["processor"] = processor
    _SEG_STATE["model"] = model
    _SEG_STATE["device"] = device
    _SEG_STATE["stuff_label_ids"] = set(stuff_ids)
    return _SEG_STATE


def segment_frame(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Panoptic seg of one BGR frame → (semantic uint8 H×W, instance uint16 H×W).

    Stuff pixels get instance_id = 0; thing pixels get a 1-based per-class
    instance counter. Pixels with no segment get semantic = 255 (unknown)
    and instance = 0.
    """
    import torch

    state = _get_seg_model()
    processor = state["processor"]
    model = state["model"]
    device = state["device"]
    stuff_ids: set[int] = state["stuff_label_ids"]

    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Max-quality override: bypass the processor's 384×384 default and feed
    # at training-scale resolution (see SEG_INPUT_SIZE for rationale).
    inputs = processor(
        images=rgb, size=SEG_INPUT_SIZE, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[(H, W)],
        label_ids_to_fuse=stuff_ids,
    )[0]
    seg_map = result["segmentation"].to(torch.int64).cpu().numpy()

    semantic = np.full((H, W), 255, dtype=np.uint8)
    instance = np.zeros((H, W), dtype=np.uint16)

    thing_counters: dict[int, int] = {}
    for info in result["segments_info"]:
        seg_id = int(info["id"])
        label_id = int(info["label_id"])
        mask = seg_map == seg_id
        if not mask.any():
            continue
        semantic[mask] = label_id
        # was_fused == True ⇒ "stuff" class; one fused segment per class, instance=0.
        if not info.get("was_fused", False):
            thing_counters[label_id] = thing_counters.get(label_id, 0) + 1
            instance[mask] = thing_counters[label_id]

    return semantic, instance


_DEPTH_STATE: dict[str, object] = {}


def _get_depth_model():
    """Lazy-load Apple Depth Pro; cache for the process."""
    if "model" in _DEPTH_STATE:
        return _DEPTH_STATE["processor"], _DEPTH_STATE["model"], _DEPTH_STATE["device"]

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device).eval()

    _DEPTH_STATE["processor"] = processor
    _DEPTH_STATE["model"] = model
    _DEPTH_STATE["device"] = device
    return processor, model, device


def mono_depth_frame(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mono depth of one BGR frame → (metric_depth fp16, conf fp16), both H×W.

    Depth Pro emits metric depth in metres; the HF processor's post-process
    step resamples it back to the source resolution and casts to fp16. The
    extra `field_of_view` / `focal_length` predictions are dropped here —
    they're constant across a fixed-lens dashcam, and Phase C can read them
    off the model directly if a per-frame focal estimate is ever wanted.
    Depth Pro has no native per-pixel confidence; we cache a uniform 1.0 and
    let downstream stages do semantic-aware weighting from the segmentation
    cache (sky pixels' weight goes to zero via Phase B's class filter).
    """
    import torch

    processor, model, device = _get_depth_model()
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(H, W)]
    )[0]
    depth = post["predicted_depth"].detach().cpu().numpy().astype(np.float32)
    conf = np.ones_like(depth, dtype=np.float32)
    return depth.astype(np.float16), conf.astype(np.float16)


def build_perception_cache(
    source: Path,
    out_dir: Path,
    step: int = 3,
    *,
    do_segment: bool = False,
    do_depth: bool = False,
) -> Path:
    """Iterate frames of *source* at *step* and (optionally) run perception
    models, writing one .npz per frame plus a manifest.

    With no perception flags this only writes the manifest — useful for
    sanity-checking the frame schedule before A2/A3 are wired in.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-running with a different perception stage on an existing cache must
    # preserve the previous stage's manifest flags. Validate source/step
    # against the existing manifest so a typo can't silently mix two videos.
    manifest_path = out_dir / MANIFEST_NAME
    existing: CacheManifest | None = None
    if manifest_path.exists():
        existing = CacheManifest.load(manifest_path)
        if existing.source != str(source) or existing.step != step:
            raise typer.BadParameter(
                f"existing cache at {out_dir} was built from "
                f"source={existing.source!r} step={existing.step}; refusing to "
                f"merge with source={str(source)!r} step={step}"
            )

    n = 0
    height = width = 0
    for frame_idx, (_name, bgr) in enumerate(iter_frames(source, step=step)):
        if frame_idx == 0:
            height, width = bgr.shape[:2]
        if do_segment:
            sem, inst = segment_frame(bgr)
            write_frame_entry(out_dir, frame_idx, semantic=sem, instance=inst)
        if do_depth:
            depth, conf = mono_depth_frame(bgr)
            write_frame_entry(out_dir, frame_idx, mono_depth=depth, depth_conf=conf)
        n += 1

    if existing is not None:
        manifest = existing
        manifest.n_frames = n
        manifest.height = int(height)
        manifest.width = int(width)
    else:
        manifest = CacheManifest(
            version=CACHE_VERSION,
            source=str(source),
            step=step,
            n_frames=n,
            height=int(height),
            width=int(width),
        )
    if do_segment:
        manifest.populated["semantic"] = True
        manifest.populated["instance"] = True
        manifest.seg_model = SEG_MODEL_ID
    if do_depth:
        manifest.populated["mono_depth"] = True
        manifest.populated["depth_conf"] = True
        manifest.depth_model = DEPTH_MODEL_ID
    manifest_path.write_text(manifest.to_json())

    typer.echo(
        f"[perception] cache {out_dir} : {n} frames @ {width}x{height}, "
        f"step={step}, seg={'on' if do_segment else 'off'}, "
        f"depth={'on' if do_depth else 'off'}"
    )
    return out_dir


def _draw_manifest_banner(
    img: np.ndarray, manifest: CacheManifest, frame_idx: int, npz_present: bool
) -> None:
    """Top-left banner: cache version, frame index, populated-field summary.

    Honest about A1's empty state: a sky-blue dot means a field is populated
    in the cache, a grey dot means it's not. Until A2 / A3 land they're all
    grey — that's what "visualizing A1" looks like.
    """
    h, w = img.shape[:2]
    pad = 8
    line_h = 22
    n_lines = 3
    banner_h = pad * 2 + line_h * n_lines
    banner_w = min(w, 520)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (banner_w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

    fnt = cv2.FONT_HERSHEY_SIMPLEX
    white = (240, 240, 240)
    cv2.putText(img, f"perception cache v{manifest.version}  step={manifest.step}",
                (pad, pad + 16), fnt, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {frame_idx:06d} / {manifest.n_frames - 1:06d}   "
                f"npz: {'yes' if npz_present else 'no'}",
                (pad, pad + 16 + line_h), fnt, 0.5, white, 1, cv2.LINE_AA)

    # Populated-field dots — green when present, grey when absent.
    x = pad
    y = pad + 16 + line_h * 2
    for fld in PERCEPTION_FIELDS:
        on = manifest.populated.get(fld, False)
        color = (80, 220, 80) if on else (110, 110, 110)
        cv2.circle(img, (x + 6, y - 5), 5, color, -1)
        cv2.putText(img, fld, (x + 16, y), fnt, 0.42, white, 1, cv2.LINE_AA)
        x += 16 + 8 + len(fld) * 8 + 8


def _colorize_semantic(semantic: np.ndarray) -> np.ndarray:
    """Map a class-id array to a stable BGR colour image via a hashed palette."""
    # Stable per-class palette — deterministic, independent of class count.
    classes = np.arange(256, dtype=np.uint8)
    rng = np.random.default_rng(seed=0xC17C)
    palette = rng.integers(40, 230, size=(256, 3), dtype=np.uint8)
    palette[255] = (0, 0, 0)  # 255 = unknown → black
    return palette[semantic]


def _colorize_depth(mono_depth: np.ndarray, conf: np.ndarray | None) -> np.ndarray:
    d = mono_depth.astype(np.float32)
    lo, hi = float(np.nanpercentile(d, 2)), float(np.nanpercentile(d, 98))
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    if conf is not None:
        c = np.clip(conf.astype(np.float32), 0.0, 1.0)[..., None]
        colored = (colored.astype(np.float32) * c).astype(np.uint8)
    return colored


# ---------------------------------------------------------------------------
# A4 — Keypoint tagging utility
#
# Given (kpt_xy, frame_idx) and a perception cache, return per-keypoint
# (class, instance, mono_d) by indexing the cached arrays at the rounded
# pixel coords. Consumers: Phase B (match filtering) and Phase C (per-track
# depth-prior assembly).


@dataclass
class KeypointTags:
    """Per-keypoint perception tags from a single frame."""

    semantic: np.ndarray   # (N,) uint8   — class id (255 = unknown)
    instance: np.ndarray   # (N,) uint16  — per-class instance id (0 = stuff)
    mono_depth: np.ndarray  # (N,) float32 — metric depth (m); fp16 in cache


_REQUIRED_TAG_FIELDS: tuple[str, ...] = ("semantic", "instance", "mono_depth")


def _require_populated(manifest: CacheManifest, fields: tuple[str, ...]) -> None:
    missing = [f for f in fields if not manifest.populated.get(f, False)]
    if missing:
        raise RuntimeError(
            f"perception cache is missing required fields {missing}; "
            f"rerun `aivision perception-cache` with the relevant --segment/--depth flags"
        )


def tag_keypoints(
    cache_dir: Path,
    frame_idx: int,
    kpt_xy: np.ndarray,
) -> KeypointTags:
    """Tag DISK/SIFT/etc. keypoints with cached perception fields.

    *kpt_xy* is (N, 2) float pixel coords (x, y). Coords are rounded to
    nearest int and clipped to image bounds before indexing. Returns
    `KeypointTags(semantic, instance, mono_depth)` aligned with kpt_xy.

    Empty input returns empty arrays of the expected dtypes. Raises if any
    of {semantic, instance, mono_depth} is not populated in the manifest —
    soft fallbacks belong at the caller layer (e.g. Phase B's filter can
    early-out when the cache lacks segmentation).
    """
    cache_dir = Path(cache_dir)
    manifest = CacheManifest.load(cache_dir / MANIFEST_NAME)
    _require_populated(manifest, _REQUIRED_TAG_FIELDS)

    xy = np.asarray(kpt_xy, dtype=np.float64).reshape(-1, 2)
    n = xy.shape[0]
    if n == 0:
        return KeypointTags(
            semantic=np.empty((0,), dtype=np.uint8),
            instance=np.empty((0,), dtype=np.uint16),
            mono_depth=np.empty((0,), dtype=np.float32),
        )

    entry = load_frame_entry(cache_dir, frame_idx)
    sem = entry["semantic"]
    inst = entry["instance"]
    depth = entry["mono_depth"]
    H, W = sem.shape
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, W - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, H - 1)
    return KeypointTags(
        semantic=sem[y, x].astype(np.uint8, copy=False),
        instance=inst[y, x].astype(np.uint16, copy=False),
        mono_depth=depth[y, x].astype(np.float32),
    )


class KeypointTagger:
    """Reusable keypoint tagger with per-frame .npz caching.

    Phase B's pairwise matcher and Phase C's per-track depth-prior
    assembly hit the same frame_idx many times; reloading the .npz on
    every call is wasteful. This wrapper keeps each loaded entry in
    memory until the caller drops the instance.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = CacheManifest.load(self.cache_dir / MANIFEST_NAME)
        _require_populated(self.manifest, _REQUIRED_TAG_FIELDS)
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def _entry(self, frame_idx: int) -> dict[str, np.ndarray]:
        e = self._cache.get(frame_idx)
        if e is None:
            e = load_frame_entry(self.cache_dir, frame_idx)
            self._cache[frame_idx] = e
        return e

    def tag(self, frame_idx: int, kpt_xy: np.ndarray) -> KeypointTags:
        xy = np.asarray(kpt_xy, dtype=np.float64).reshape(-1, 2)
        n = xy.shape[0]
        if n == 0:
            return KeypointTags(
                semantic=np.empty((0,), dtype=np.uint8),
                instance=np.empty((0,), dtype=np.uint16),
                mono_depth=np.empty((0,), dtype=np.float32),
            )
        e = self._entry(frame_idx)
        sem, inst, depth = e["semantic"], e["instance"], e["mono_depth"]
        H, W = sem.shape
        x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, W - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, H - 1)
        return KeypointTags(
            semantic=sem[y, x].astype(np.uint8, copy=False),
            instance=inst[y, x].astype(np.uint16, copy=False),
            mono_depth=depth[y, x].astype(np.float32),
        )


_FRAME_NAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def id2label_from_manifest(manifest: CacheManifest) -> dict[int, str]:
    """Resolve seg-model class names without loading the full model weights.

    `AutoConfig.from_pretrained` pulls only config.json, so this is cheap
    even for the 200M Mask2Former-L checkpoint.
    """
    if manifest.seg_model is None:
        return {}
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(manifest.seg_model)
    return {int(k): str(v) for k, v in cfg.id2label.items()}


def histogram_recon_keypoints(
    cache_dir: Path,
    reconstruction_dir: Path,
) -> tuple[dict[int, int], int]:
    """A4 smoke check — tag every observed 2D keypoint in the recon and
    accumulate (class_id → count). Returns (counts, n_total).

    Aligns recon images to perception frames by parsing the `frame_NNNNNN`
    index out of the image name; assumes the recon and cache were built
    from the same source video at the same step (consistent with how
    `iter_video` names frames). 255 (unknown / no segment) is tracked
    like any other class so the histogram surfaces holes in the seg cache.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    tagger = KeypointTagger(cache_dir)

    counts: dict[int, int] = {}
    n_total = 0
    for _img_id, img in tqdm(rec.images.items(), desc="tag-kps"):
        m = _FRAME_NAME_RE.search(img.name)
        if not m:
            raise ValueError(f"image name does not match frame_NNNNNN.jpg: {img.name}")
        frame_idx = int(m.group(1))
        xy = np.asarray([p.xy for p in img.points2D], dtype=np.float64)
        if xy.size == 0:
            continue
        tags = tagger.tag(frame_idx, xy)
        ids, cs = np.unique(tags.semantic, return_counts=True)
        for cid, c in zip(ids.tolist(), cs.tolist()):
            counts[int(cid)] = counts.get(int(cid), 0) + int(c)
        n_total += int(xy.shape[0])
    return counts, n_total


def _semantic_palette() -> np.ndarray:
    """The same 256-entry BGR palette `_colorize_semantic` builds — split out
    so callers can colour individual keypoints without going through the
    full image-array path."""
    rng = np.random.default_rng(seed=0xC17C)
    palette = rng.integers(40, 230, size=(256, 3), dtype=np.uint8)
    palette[255] = (0, 0, 0)
    return palette


def render_keypoint_tag_overlay(
    cache_dir: Path,
    reconstruction_dir: Path,
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: int = 30,
    radius_px: int = 3,
    colour: str = "class",
    top_n_legend: int = 6,
) -> Path:
    """A4 visualization — render every recon image with its DISK keypoints
    drawn as filled dots coloured by tagged class (or by tagged mono_depth,
    depending on *colour*).

    The class palette is the same hashed BGR table `_colorize_semantic` uses,
    so a keypoint's dot colour matches the semantic-mask overlay colour for
    its class in the perception-cache video — eyeballing alignment is fast.

    Depth mode normalises across all tagged observations in the recon for a
    consistent video-wide turbo scale, so a near keypoint reads red on
    every frame.
    """
    import pycolmap

    if colour not in ("class", "depth"):
        raise ValueError(f"colour must be 'class' or 'depth'; got {colour!r}")

    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    tagger = KeypointTagger(cache_dir)
    id2label = id2label_from_manifest(tagger.manifest)
    id2label.setdefault(255, "unknown")
    palette = _semantic_palette()

    frame_to_img: dict[int, "pycolmap.Image"] = {}
    for img in rec.images.values():
        m = _FRAME_NAME_RE.search(img.name)
        if not m:
            raise ValueError(f"image name does not match frame_NNNNNN.jpg: {img.name}")
        frame_to_img[int(m.group(1))] = img
    sorted_frames = sorted(frame_to_img.keys())

    # Pre-compute per-frame tagged xys so the writer loop just iterates.
    per_frame: dict[int, tuple[np.ndarray, KeypointTags]] = {}
    all_depths: list[np.ndarray] = []
    for f in tqdm(sorted_frames, desc="tag-kps (cache)"):
        img = frame_to_img[f]
        xy = np.asarray([p.xy for p in img.points2D], dtype=np.float64)
        tags = tagger.tag(f, xy)
        per_frame[f] = (xy, tags)
        if colour == "depth" and tags.mono_depth.size:
            all_depths.append(tags.mono_depth)

    depth_lo = depth_hi = 0.0
    if colour == "depth" and all_depths:
        dcat = np.concatenate(all_depths)
        depth_lo = float(np.nanpercentile(dcat, 2))
        depth_hi = float(np.nanpercentile(dcat, 98))
        if depth_hi <= depth_lo:
            depth_hi = depth_lo + 1e-3

    sample = cv2.imread(str(frames_dir / frame_to_img[sorted_frames[0]].name))
    h, w = sample.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

    for f in tqdm(sorted_frames, desc=f"render {out_mp4.name}"):
        img_obj = frame_to_img[f]
        img = cv2.imread(str(frames_dir / img_obj.name))
        xy, tags = per_frame[f]
        if xy.size:
            ix = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
            iy = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
            if colour == "class":
                colours = palette[tags.semantic]
            else:
                d = np.clip((tags.mono_depth - depth_lo) / (depth_hi - depth_lo), 0.0, 1.0)
                u8 = (d * 255).astype(np.uint8)
                colours = cv2.applyColorMap(u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
            for px, py, bgr in zip(ix.tolist(), iy.tolist(), colours.tolist()):
                cv2.circle(img, (px, py), radius_px, (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1, cv2.LINE_AA)

        # HUD: frame name + kpt count + top-N classes for this frame (class mode
        # only — in depth mode the colour bar is more informative than the list).
        cv2.rectangle(img, (0, 0), (520, 28 + (top_n_legend * 22 if colour == "class" else 0) + 6),
                      (20, 20, 20), -1)
        cv2.putText(img, f"{img_obj.name}  kpts={int(xy.shape[0])}",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        if colour == "class" and xy.size:
            ids, cs = np.unique(tags.semantic, return_counts=True)
            order = np.argsort(-cs)[:top_n_legend]
            y0 = 50
            for rank, k in enumerate(order):
                cid = int(ids[k]); cnt = int(cs[k])
                name = id2label.get(cid, f"id_{cid}")
                bgr = palette[cid].tolist()
                cv2.circle(img, (22, y0 + rank * 22 - 5), 6, (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
                cv2.putText(img, f"{name}  {cnt}",
                            (38, y0 + rank * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (240, 240, 240), 1, cv2.LINE_AA)
        if colour == "depth":
            # Mini horizontal colour bar bottom-left so the per-frame scale is
            # interpretable at a glance.
            bar_w, bar_h = 240, 16
            x0, y0 = 12, h - bar_h - 12
            ramp = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, -1)
            bar = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
            bar = cv2.resize(bar, (bar_w, bar_h), interpolation=cv2.INTER_NEAREST)
            img[y0:y0 + bar_h, x0:x0 + bar_w] = bar
            cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
            cv2.putText(img, f"{depth_lo:.1f} m", (x0, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(img, f"{depth_hi:.1f} m", (x0 + bar_w - 60, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(img)
    writer.release()
    typer.echo(f"[perception-tag-kps] wrote {out_mp4} ({len(sorted_frames)} frames, colour={colour})")
    return out_mp4


def render_perception_overlay(
    cache_dir: Path,
    out_mp4: Path,
    fps: int = 30,
    seg_alpha: float = 0.45,
) -> Path:
    """Render a video of the perception cache: source frames + manifest banner
    + (where populated) semantic-mask overlay and depth heatmap inset.

    For an A1-only cache (no fields populated), the output is the source
    frames with the manifest banner — making explicit that the schedule is
    in place and downstream phases are not yet wired.
    """
    cache_dir = Path(cache_dir)
    manifest = CacheManifest.load(cache_dir / MANIFEST_NAME)
    source = Path(manifest.source)

    # Probe size from the source.
    first = next(iter(iter_frames(source, step=manifest.step)))
    h, w = first[1].shape[:2]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

    inset_w = w // 4
    n_written = 0

    for frame_idx, (_name, bgr) in tqdm(
        enumerate(iter_frames(source, step=manifest.step)),
        total=manifest.n_frames, desc=f"render {out_mp4.name}",
    ):
        img = bgr.copy()
        npz_path = _frame_npz_path(cache_dir, frame_idx)
        entry = {}
        if npz_path.exists():
            with np.load(npz_path) as f:
                entry = {k: f[k] for k in f.files}

        if "semantic" in entry:
            sem_bgr = _colorize_semantic(entry["semantic"])
            if sem_bgr.shape[:2] != (h, w):
                sem_bgr = cv2.resize(sem_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.addWeighted(sem_bgr, seg_alpha, img, 1.0 - seg_alpha, 0, dst=img)

        if "mono_depth" in entry:
            conf = entry.get("depth_conf")
            depth_bgr = _colorize_depth(entry["mono_depth"], conf)
            inset_h = int(depth_bgr.shape[0] * inset_w / depth_bgr.shape[1])
            inset = cv2.resize(depth_bgr, (inset_w, inset_h))
            y0 = h - inset_h - 8
            x0 = w - inset_w - 8
            img[y0:y0 + inset_h, x0:x0 + inset_w] = inset
            cv2.rectangle(img, (x0, y0), (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1)

        _draw_manifest_banner(img, manifest, frame_idx, npz_present=bool(entry))
        writer.write(img)
        n_written += 1

    writer.release()
    typer.echo(f"[perception] wrote {out_mp4} ({n_written} frames @ {fps} fps)")
    return out_mp4
