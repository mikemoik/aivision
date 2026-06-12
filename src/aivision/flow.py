"""Dense optical flow + road-plane homography POC (round-3 C4).

Goal: replace DISK's near-zero road coverage (0.55% of keypoints, per the
A4 histogram) with a pixel-dense flow field, mask it to Vistas Road +
LaneMarking, and fit a per-pair homography that will eventually feed a
metric-velocity residual into BAProblem.

This module owns:

- ``compute_flow_pair(bgr_a, bgr_b)`` — one forward pass, returns full
  (H, W, 2) flow. We always compute the full field and mask afterwards;
  there's no "road-only" inference mode in RAFT-family models and the
  non-road flow has its own uses (far-field rotation prior, dynamic-
  object outlier rejection).
- ``road_class_ids(manifest)`` — resolve the Vistas Road / Lane Marking
  class ids from the cached id2label, no model weights needed.
- ``fit_road_homography(flow, mask, ...)`` — sample a regular grid of
  pixels inside the mask, build (src, tgt) correspondences via the flow,
  RANSAC a 3×3 homography, report inliers + Sampson residual.
- ``run_flow_poc(...)`` — orchestrator: pick N consecutive pairs, run
  flow, mask, fit, write a 2×2 visualization PNG per pair and a single
  summary JSON.
- ``run_flow_video(...)`` — full-video version: every consecutive pair,
  rendered as a single mp4 with flow magnitude + road-arrow overlay.

Backends:
- ``sea_raft`` (default) — Princeton VL's SEA-RAFT, repo vendored at
  ``models/sea_raft/``, checkpoint pulled from HF ``MemorySlices/...``.
  Spring-M is the dashcam-appropriate variant (Tartan + Chairs + Things
  + TSKH + Spring finetune). Trained image_size 540×960; runs full-res
  on our 720×1280 via the InputPadder + scale=0 (override the config's
  half-res default).
- ``raft_large`` — torchvision Sintel-FT RAFT. Already in deps; useful
  as a sanity-check backend and as fallback if HF hub is unreachable.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import typer
from tqdm import tqdm

from .perception import CacheManifest, MANIFEST_NAME, id2label_from_manifest, load_frame_entry


# Default SEA-RAFT checkpoint: Tartan + Chairs + Things + TSKH (KITTI in there)
# + Spring finetune. M = medium variant — the paper's recommended general-
# purpose model. Hub repo from MemorySlices org. The corresponding config
# lives in the vendored repo at config/eval/spring-M.json.
SEA_RAFT_DEFAULT_HF = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
SEA_RAFT_DEFAULT_CONFIG = "config/eval/spring-M.json"
SEA_RAFT_REPO_DIR = Path(__file__).resolve().parents[2] / "models" / "sea_raft"


# Vistas semantic class names treated as "road plane" for the homography
# fit. Lane Marking variants are textured asphalt — exactly where dense
# flow has cleanest gradient — and Service Lane / Bike Lane are also
# locally planar with the road. Crosswalk-Plain (zebra) and Curb Cut are
# left out for v1: zebra stripes induce strong but non-planar flow
# patterns at the kerb edge.
ROAD_LABEL_NAMES: frozenset[str] = frozenset({
    "Road",
    "Lane Marking - General",
    "Lane Marking - Crosswalk",
    "Service Lane",
    "Bike Lane",
})


# ---------------------------------------------------------------------------
# Model loader


_FLOW_STATE: dict[str, object] = {}


def _get_flow_model(backend: str = "sea_raft"):
    """Lazy-load an optical-flow model, cache for the process lifetime.

    Returns ``(model, device)``. ``model`` is whatever the backend's
    inference path expects — the actual forward call lives in
    ``compute_flow_pair`` and dispatches on ``backend``.
    """
    key = f"model::{backend}"
    if key in _FLOW_STATE:
        return _FLOW_STATE[key], _FLOW_STATE["device"]

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _FLOW_STATE["device"] = device

    if backend == "raft_large":
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        weights = Raft_Large_Weights.C_T_SKHT_V2
        model = raft_large(weights=weights, progress=False).to(device).eval()
        _FLOW_STATE[key] = model
        _FLOW_STATE[f"weights::{backend}"] = weights
        return model, device

    if backend == "sea_raft":
        if not SEA_RAFT_REPO_DIR.exists():
            raise RuntimeError(
                f"SEA-RAFT repo not vendored at {SEA_RAFT_REPO_DIR}. "
                "Clone with: git clone --depth 1 "
                "https://github.com/princeton-vl/SEA-RAFT.git "
                f"{SEA_RAFT_REPO_DIR}"
            )
        # The repo's core/ uses bare imports (`from update import ...`,
        # `from utils.utils import ...`), so both the repo root *and* its
        # core/ need to be on sys.path. We prepend rather than append to
        # win over any same-named package (unlikely but cheap).
        repo_root = str(SEA_RAFT_REPO_DIR)
        core_dir = str(SEA_RAFT_REPO_DIR / "core")
        for p in (core_dir, repo_root):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Build an args object matching what the config parser would
        # produce. SimpleNamespace works because RAFT only reads
        # attributes off ``args``; no validation. ``scale=0`` overrides
        # the config's -1 (half-res) so we run at full source resolution
        # — 1280×720 isn't large enough to need the downsample.
        cfg_path = SEA_RAFT_REPO_DIR / SEA_RAFT_DEFAULT_CONFIG
        cfg = json.loads(cfg_path.read_text())
        args = SimpleNamespace(**cfg)
        args.scale = 0
        args.dropout = getattr(args, "dropout", 0)

        from raft import RAFT  # type: ignore

        model = RAFT.from_pretrained(SEA_RAFT_DEFAULT_HF, args=args)
        model = model.to(device).eval()
        _FLOW_STATE[key] = model
        _FLOW_STATE[f"args::{backend}"] = args
        return model, device

    raise NotImplementedError(f"flow backend {backend!r} not wired yet")


def _preprocess_raft_large(bgr: np.ndarray, device):
    """BGR uint8 H×W×3 → BCHW float in [-1, 1], padded to multiples of 8.

    torchvision RAFT requires divisible-by-8 spatial dims; SEA-RAFT
    handles its own padding via InputPadder.
    """
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    t = t * 2.0 - 1.0
    _, h, w = t.shape
    pad_h = (-h) % 8
    pad_w = (-w) % 8
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
    return t.unsqueeze(0), (h, w)


def _preprocess_sea_raft(bgr: np.ndarray, device):
    """BGR uint8 H×W×3 → BCHW float in [0, 255]. SEA-RAFT's forward
    does the /255 and 2x-1 normalisation internally (raft.py L84-85)."""
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float()
    return t.unsqueeze(0)


def compute_flow_pair(
    bgr_a: np.ndarray,
    bgr_b: np.ndarray,
    *,
    backend: str = "sea_raft",
    n_iters: int | None = None,
) -> np.ndarray:
    """Forward flow from frame *a* to frame *b*. Returns (H, W, 2) float32
    where channel-0 is dx (pixels) and channel-1 is dy.

    Always computes the full image — masking is the caller's job. The
    full field is also useful for future consumers (far-field rotation
    prior, dynamic-object outlier rejection); caching the raw flow
    keeps those options cheap.

    ``n_iters`` is backend-specific. For SEA-RAFT it overrides the
    config's iters (default 4 for spring-M); for raft_large it overrides
    num_flow_updates (default 12 = library default; we used 20 in the
    POC). Pass None to let the backend pick.
    """
    import torch

    if bgr_a.shape != bgr_b.shape:
        raise ValueError(f"frame shape mismatch: {bgr_a.shape} vs {bgr_b.shape}")

    model, device = _get_flow_model(backend=backend)

    if backend == "raft_large":
        a, (H, W) = _preprocess_raft_large(bgr_a, device)
        b, _ = _preprocess_raft_large(bgr_b, device)
        iters = 20 if n_iters is None else n_iters
        with torch.inference_mode():
            flow_iters = model(a, b, num_flow_updates=iters)
        flow = flow_iters[-1][0].detach().cpu().numpy()
        flow = flow[:, :H, :W].transpose(1, 2, 0).astype(np.float32)
        return flow

    if backend == "sea_raft":
        a = _preprocess_sea_raft(bgr_a, device)
        b = _preprocess_sea_raft(bgr_b, device)
        args = _FLOW_STATE[f"args::{backend}"]
        iters = args.iters if n_iters is None else n_iters
        with torch.inference_mode():
            out = model(a, b, iters=iters, test_mode=True)
        # SEA-RAFT handles its own padding/unpadding; output is already
        # at the input resolution. flow[-1] is the last iter's prediction.
        flow = out["flow"][-1][0].detach().cpu().numpy()  # (2, H, W)
        return flow.transpose(1, 2, 0).astype(np.float32)

    raise NotImplementedError(f"compute_flow_pair: unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Semantic mask


def road_class_ids(manifest: CacheManifest) -> set[int]:
    """Resolve the ``ROAD_LABEL_NAMES`` set against the cache's seg model.

    Raises if the manifest has no segmentation model recorded — the
    homography fit is meaningless without the mask.
    """
    if manifest.seg_model is None:
        raise RuntimeError(
            f"perception cache has no seg_model recorded; cannot resolve road class ids"
        )
    id2label = id2label_from_manifest(manifest)
    name_to_id = {v: int(k) for k, v in id2label.items()}
    missing = [n for n in ROAD_LABEL_NAMES if n not in name_to_id]
    if missing:
        typer.echo(f"[flow] WARNING: road label names not in id2label: {missing}")
    return {name_to_id[n] for n in ROAD_LABEL_NAMES if n in name_to_id}


def road_mask(semantic: np.ndarray, class_ids: set[int]) -> np.ndarray:
    """Boolean H×W mask of pixels whose semantic id is in *class_ids*."""
    if not class_ids:
        return np.zeros(semantic.shape, dtype=bool)
    ids = np.fromiter(class_ids, dtype=np.uint8)
    return np.isin(semantic, ids)


# ---------------------------------------------------------------------------
# Homography fit


@dataclass
class HomographyFit:
    H: np.ndarray | None        # 3×3 homography, None if RANSAC failed
    src_xy: np.ndarray          # (N, 2) sampled source pixels (all candidates)
    tgt_xy: np.ndarray          # (N, 2) src + flow
    inlier_mask: np.ndarray     # (N,) bool from RANSAC
    sampson_median: float       # over inliers
    n_road_px: int              # mask pixel count for context


def _sampson_residual(H: np.ndarray, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Symmetric transfer error in pixels: average of forward and inverse
    re-projection distances. Cheaper than the analytical Sampson form and
    has the same monotonicity for inlier counts. Returns shape (N,)."""
    src_h = np.hstack([src, np.ones((src.shape[0], 1))])
    tgt_h = np.hstack([tgt, np.ones((tgt.shape[0], 1))])
    Hinv = np.linalg.inv(H)
    f = (H @ src_h.T).T
    f = f[:, :2] / np.where(np.abs(f[:, 2:3]) < 1e-9, 1e-9, f[:, 2:3])
    b = (Hinv @ tgt_h.T).T
    b = b[:, :2] / np.where(np.abs(b[:, 2:3]) < 1e-9, 1e-9, b[:, 2:3])
    fwd = np.linalg.norm(f - tgt, axis=1)
    inv = np.linalg.norm(b - src, axis=1)
    return 0.5 * (fwd + inv)


def fit_road_homography(
    flow: np.ndarray,
    mask: np.ndarray,
    *,
    grid_stride: int = 12,
    ransac_thresh: float = 3.0,
    min_inliers: int = 50,
) -> HomographyFit:
    """Sample a regular grid inside the road mask, build (src, tgt) pairs
    via the flow, RANSAC a 3×3 homography.

    *grid_stride*=12 over a 1280×720 frame gives ~6400 candidate pixels
    if the entire image were road — typically a few hundred to a few
    thousand once the mask restricts to actual asphalt. RANSAC
    threshold is in pixels (symmetric transfer error).
    """
    H_img, W_img = mask.shape
    ys = np.arange(grid_stride // 2, H_img, grid_stride)
    xs = np.arange(grid_stride // 2, W_img, grid_stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    keep = mask[yy, xx]
    src = np.stack([xx[keep], yy[keep]], axis=-1).astype(np.float32)
    fxy = flow[yy[keep], xx[keep]]  # (N, 2)
    tgt = src + fxy

    n_road_px = int(mask.sum())
    empty = HomographyFit(
        H=None, src_xy=src, tgt_xy=tgt,
        inlier_mask=np.zeros(src.shape[0], dtype=bool),
        sampson_median=float("nan"), n_road_px=n_road_px,
    )
    if src.shape[0] < min_inliers:
        return empty

    H, inl = cv2.findHomography(src, tgt, cv2.RANSAC, ransac_thresh)
    if H is None:
        return empty
    inlier_mask = inl.ravel().astype(bool)
    if inlier_mask.sum() < min_inliers:
        return HomographyFit(
            H=H, src_xy=src, tgt_xy=tgt, inlier_mask=inlier_mask,
            sampson_median=float("nan"), n_road_px=n_road_px,
        )
    residuals = _sampson_residual(H, src[inlier_mask], tgt[inlier_mask])
    return HomographyFit(
        H=H, src_xy=src, tgt_xy=tgt, inlier_mask=inlier_mask,
        sampson_median=float(np.median(residuals)),
        n_road_px=n_road_px,
    )


# ---------------------------------------------------------------------------
# Homography → metric step length (Song-Chandraker recipe)


@dataclass
class StepFromHomography:
    """Per-pair metric step extracted from a road-plane homography.

    ``step_norm`` is the dimensionless ||t/d|| from cv2.decomposeHomographyMat.
    Multiply by camera height ``h_cam`` (m) to get a metric step length.

    ``confidence`` ∈ [0, 1] reports how plausible the disambiguated plane
    normal is — 1.0 when the chosen ``n`` is exactly (0, -1, 0) (cam-y-down,
    road directly below). Use as a per-pair down-weight when the 4-solution
    disambiguation picks a wonky normal.
    """
    step_norm: float          # ||t/d|| (dimensionless)
    t_dir: np.ndarray         # (3,) chosen translation direction (cam-a frame)
    n: np.ndarray             # (3,) chosen plane normal (cam-a frame)
    R: np.ndarray             # (3, 3) chosen relative rotation cam-a → cam-b
    confidence: float         # max(0, -n_y) — 1 = ideal road-below-camera
    n_solutions: int          # 1, 2, or 4 from cv2


def decompose_step_from_h(H_pixel: np.ndarray, K_3x3: np.ndarray) -> StepFromHomography:
    """Song-Chandraker style: decompose a pixel-space road-plane homography
    into (R, t/d, n), pick the disambiguated solution, return its scaled
    translation magnitude. Multiply by camera height to recover metric step.

    NOTE: ignores radial distortion. For a SIMPLE_RADIAL camera with non-
    trivial ``k1`` the inputs to ``cv2.findHomography`` are distorted pixels,
    so the fitted ``H_pixel`` is only an approximation of the true planar
    homography. The ``confidence`` field is the practical defence against
    this — pairs where the chosen ``n`` deviates from straight-down get
    down-weighted in the BA prior. A future v2 should undistort the road
    inlier points and re-fit a normalized homography directly.
    """
    n_sol, Rs, Ts, Ns = cv2.decomposeHomographyMat(H_pixel, K_3x3)
    # Score = how aligned n is with (0,-1,0) (cam-y-down, road below cam) +
    # a tie-breaker favouring R close to identity (consecutive frames have
    # small rotation, so identity-leaning solutions are more likely real).
    best_k = -1
    best_score = -1e18
    for k in range(n_sol):
        n_vec = Ns[k].ravel()
        R = Rs[k]
        n_score = -float(n_vec[1])              # 1 when n = (0,-1,0)
        rot_score = 0.5 * (float(np.trace(R)) - 1.0)  # 1 when R = I
        score = n_score * 10.0 + rot_score
        if score > best_score:
            best_score = score
            best_k = k
    t = Ts[best_k].ravel().astype(np.float64)
    n_v = Ns[best_k].ravel().astype(np.float64)
    R_v = Rs[best_k].astype(np.float64)
    step_norm = float(np.linalg.norm(t))
    confidence = float(max(0.0, -n_v[1]))
    return StepFromHomography(
        step_norm=step_norm,
        t_dir=(t / step_norm if step_norm > 1e-12 else t),
        n=n_v,
        R=R_v,
        confidence=confidence,
        n_solutions=int(n_sol),
    )


def load_step_priors_from_flow_json(
    json_path: Path,
    K_3x3: np.ndarray,
    h_cam: float,
    *,
    min_confidence: float = 0.3,
    min_inliers: int = 100,
) -> dict[int, dict]:
    """Read a ``flow-video`` summary JSON, decompose each per-pair homography
    into a metric step length, return ``{i_a: {step_m, confidence, ...}}``.

    Pairs with confidence < ``min_confidence`` (plane normal too far from
    straight-down) or inliers < ``min_inliers`` (homography fit thin) are
    *included* but their entry carries the low confidence so BA can
    down-weight them rather than dropping them outright.
    """
    payload = json.loads(Path(json_path).read_text())
    out: dict[int, dict] = {}
    for p in payload["pairs"]:
        H = p.get("homography")
        if H is None:
            continue
        H = np.array(H, dtype=np.float64)
        step = decompose_step_from_h(H, K_3x3)
        n_inl = int(p.get("n_inliers", 0))
        sampson = p.get("sampson_median_px")
        inlier_score = min(1.0, n_inl / 500.0) if n_inl > 0 else 0.0
        sampson_score = (
            1.0 if sampson is None or not np.isfinite(sampson)
            else float(np.clip(1.0 - (sampson - 0.3) / 1.5, 0.0, 1.0))
        )
        conf = step.confidence * inlier_score * sampson_score
        out[int(p["i_a"])] = {
            "i_a": int(p["i_a"]),
            "i_b": int(p["i_b"]),
            "step_m": h_cam * step.step_norm,
            "step_norm": step.step_norm,
            "confidence": float(conf),
            "n_chosen": step.n.tolist(),
            "t_dir": step.t_dir.tolist(),
            "n_inliers": n_inl,
            "sampson_median_px": sampson,
            "below_confidence_floor": conf < min_confidence,
            "below_inlier_floor": n_inl < min_inliers,
        }
    return out


# ---------------------------------------------------------------------------
# Visualization


def _flow_to_magnitude_bgr(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    mag = np.linalg.norm(flow, axis=-1)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 98)) or 1.0
    norm = np.clip(mag / max_mag, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _draw_flow_arrows(
    img: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    mask: np.ndarray | None = None,
    color_ok: tuple[int, int, int] = (60, 220, 60),
    color_bad: tuple[int, int, int] = (60, 60, 220),
    stride: int = 4,
    thickness: int = 1,
) -> None:
    """In-place arrow overlay. *mask* (if given) splits arrows into
    ok (green) / bad (red); without it everything is green."""
    src = src[::stride]
    tgt = tgt[::stride]
    if mask is not None:
        mask = mask[::stride]
    for i, (s, t) in enumerate(zip(src, tgt)):
        c = color_ok if (mask is None or mask[i]) else color_bad
        cv2.arrowedLine(
            img, (int(s[0]), int(s[1])), (int(t[0]), int(t[1])),
            c, thickness, cv2.LINE_AA, tipLength=0.3,
        )


def render_pair_panel(
    bgr_a: np.ndarray,
    bgr_b: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    fit: HomographyFit,
    *,
    arrow_stride: int = 4,
) -> np.ndarray:
    """2×2 PNG-ready panel: (frame_a + mask tint) | flow magnitude |
    masked flow arrows | RANSAC inliers/outliers."""
    H, W = bgr_a.shape[:2]

    # Top-left: source frame with road mask tinted blue.
    tl = bgr_a.copy()
    tint = np.zeros_like(tl)
    tint[mask] = (200, 80, 30)  # warm tint so it pops on a road
    cv2.addWeighted(tint, 0.35, tl, 0.65, 0, dst=tl)
    cv2.putText(tl, "frame_a + road mask", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Top-right: flow magnitude.
    tr = _flow_to_magnitude_bgr(flow)
    cv2.putText(tr, "|flow| (full image)", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom-left: masked flow vectors over frame_a (no RANSAC yet).
    bl = bgr_a.copy()
    _draw_flow_arrows(bl, fit.src_xy, fit.tgt_xy, mask=None,
                      stride=arrow_stride, thickness=1)
    cv2.putText(bl, f"masked flow  N={fit.src_xy.shape[0]}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom-right: inliers green / outliers red after homography RANSAC.
    br = bgr_a.copy()
    _draw_flow_arrows(br, fit.src_xy, fit.tgt_xy, mask=fit.inlier_mask,
                      stride=arrow_stride, thickness=1)
    n_inl = int(fit.inlier_mask.sum())
    sampson = f"{fit.sampson_median:.2f}" if np.isfinite(fit.sampson_median) else "n/a"
    label = f"RANSAC inliers={n_inl}/{fit.src_xy.shape[0]}  sampson_med={sampson}px"
    cv2.putText(br, label, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    top = np.hstack([tl, tr])
    bot = np.hstack([bl, br])
    return np.vstack([top, bot])


# ---------------------------------------------------------------------------
# Orchestrator


def _load_bgr(frame_path: Path) -> np.ndarray:
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"could not read frame: {frame_path}")
    return img


def run_flow_poc(
    perception_cache: Path,
    frames_dir: Path,
    out_dir: Path,
    *,
    pair_indices: list[int] | None = None,
    n_pairs: int = 3,
    grid_stride: int = 12,
    ransac_thresh: float = 3.0,
    arrow_stride: int = 4,
    backend: str = "sea_raft",
) -> dict:
    """Run flow + road-mask + homography on N pairs, write panel PNGs and
    a summary JSON. *pair_indices* is a list of source-frame indices `i`
    (each pair runs flow from frame i to i+1); when None, picks pairs
    uniformly across the available range.

    Returns the summary dict (also written to ``out_dir / "summary.json"``).
    """
    perception_cache = Path(perception_cache)
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = CacheManifest.load(perception_cache / MANIFEST_NAME)
    if not manifest.populated.get("semantic", False):
        raise RuntimeError(
            f"perception cache at {perception_cache} has no semantic mask; "
            f"rerun perception-cache with --segment"
        )
    class_ids = road_class_ids(manifest)
    if not class_ids:
        raise RuntimeError("could not resolve any Vistas road class ids in the cache")

    # The cache and the frames dir both use frame_NNNNNN indexing aligned
    # at step=manifest.step. ``i`` here is the cache index (0..n_frames-1).
    n_frames = manifest.n_frames
    if pair_indices is None:
        if n_pairs >= n_frames - 1:
            pair_indices = list(range(n_frames - 1))
        else:
            pair_indices = list(np.linspace(0, n_frames - 2, n_pairs, dtype=int))

    typer.echo(
        f"[flow-poc] backend={backend} pairs={pair_indices} "
        f"road_classes={sorted(class_ids)}"
    )

    summary = {
        "backend": backend,
        "perception_cache": str(perception_cache),
        "frames_dir": str(frames_dir),
        "grid_stride": grid_stride,
        "ransac_thresh": ransac_thresh,
        "road_class_ids": sorted(class_ids),
        "pairs": [],
    }

    for i in tqdm(pair_indices, desc="flow-poc"):
        path_a = frames_dir / f"frame_{i:06d}.jpg"
        path_b = frames_dir / f"frame_{i + 1:06d}.jpg"
        bgr_a = _load_bgr(path_a)
        bgr_b = _load_bgr(path_b)
        flow = compute_flow_pair(bgr_a, bgr_b, backend=backend)

        sem = load_frame_entry(perception_cache, i)["semantic"]
        if sem.shape != bgr_a.shape[:2]:
            sem = cv2.resize(sem, (bgr_a.shape[1], bgr_a.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        mask = road_mask(sem, class_ids)

        fit = fit_road_homography(
            flow, mask, grid_stride=grid_stride, ransac_thresh=ransac_thresh,
        )
        panel = render_pair_panel(bgr_a, bgr_b, flow, mask, fit,
                                  arrow_stride=arrow_stride)
        panel_path = out_dir / f"pair_{i:06d}_{i + 1:06d}.png"
        cv2.imwrite(str(panel_path), panel)

        summary["pairs"].append({
            "i_a": int(i),
            "i_b": int(i + 1),
            "n_road_px": int(fit.n_road_px),
            "n_candidates": int(fit.src_xy.shape[0]),
            "n_inliers": int(fit.inlier_mask.sum()),
            "sampson_median_px": (
                float(fit.sampson_median) if np.isfinite(fit.sampson_median) else None
            ),
            "homography": (fit.H.tolist() if fit.H is not None else None),
            "panel": panel_path.name,
        })

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"[flow-poc] wrote {len(pair_indices)} panels + {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# Full-video renderer


def _render_video_frame(
    bgr_a: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    fit: HomographyFit,
    *,
    frame_idx: int,
    n_frames: int,
    arrow_stride: int = 6,
    overlay_alpha: float = 0.45,
    flow_magnitude_floor: float = 0.5,
) -> np.ndarray:
    """One frame for the flow-video: source frame + translucent flow
    magnitude (turbo) + road mask outline (cyan) + RANSAC arrows
    (green/red) + HUD."""
    H, W = bgr_a.shape[:2]
    out = bgr_a.copy()

    # Turbo overlay of flow magnitude. We only tint pixels with non-
    # trivial flow so static regions keep the original look — important
    # for traffic-stop frames where the global colormap would otherwise
    # paint everything dark blue.
    mag = np.linalg.norm(flow, axis=-1)
    mag_max = float(np.percentile(mag, 99)) or 1.0
    mag_norm = np.clip(mag / mag_max, 0.0, 1.0)
    mag_u8 = (mag_norm * 255).astype(np.uint8)
    flow_bgr = cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO)
    blend_mask = (mag > flow_magnitude_floor)[..., None].astype(np.float32) * overlay_alpha
    out = (out.astype(np.float32) * (1 - blend_mask)
           + flow_bgr.astype(np.float32) * blend_mask).astype(np.uint8)

    # Road mask outline — thin contour, cyan, so the homography support
    # region is unambiguous against the turbo flow underneath.
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 60), 1, cv2.LINE_AA)

    # RANSAC arrows (subsampled) — green inlier, red outlier.
    if fit.src_xy.shape[0] > 0:
        _draw_flow_arrows(
            out, fit.src_xy, fit.tgt_xy, mask=fit.inlier_mask,
            stride=arrow_stride, thickness=1,
        )

    # HUD (top-left).
    n_inl = int(fit.inlier_mask.sum())
    n_cand = int(fit.src_xy.shape[0])
    sampson = (f"{fit.sampson_median:.2f}px"
               if np.isfinite(fit.sampson_median) else "n/a")
    lines = [
        f"frame {frame_idx:03d}/{n_frames - 1:03d}  pair {frame_idx}-{frame_idx + 1}",
        f"|flow|max={mag.max():5.1f}px  median={np.median(mag):4.1f}px",
        f"RANSAC inliers={n_inl:>4d}/{n_cand:<4d}  sampson_med={sampson}",
        f"road_px={fit.n_road_px}",
    ]
    pad, line_h = 10, 22
    box_h = pad * 2 + line_h * len(lines)
    box_w = 520
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, dst=out)
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (pad, pad + 16 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    return out


def run_flow_video(
    perception_cache: Path,
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: int = 10,
    grid_stride: int = 12,
    ransac_thresh: float = 3.0,
    arrow_stride: int = 6,
    backend: str = "sea_raft",
    stats_json: Path | None = None,
) -> Path:
    """Run flow + homography on every consecutive pair in the cache and
    render a single mp4 with the per-frame visualization. Per-pair stats
    are also written to ``stats_json`` (default: ``out_mp4`` with the
    suffix swapped to ``.json``)."""
    perception_cache = Path(perception_cache)
    frames_dir = Path(frames_dir)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if stats_json is None:
        stats_json = out_mp4.with_suffix(".json")

    manifest = CacheManifest.load(perception_cache / MANIFEST_NAME)
    if not manifest.populated.get("semantic", False):
        raise RuntimeError(
            f"perception cache at {perception_cache} has no semantic mask; "
            f"rerun perception-cache with --segment"
        )
    class_ids = road_class_ids(manifest)
    if not class_ids:
        raise RuntimeError("could not resolve any Vistas road class ids in the cache")

    n_frames = manifest.n_frames
    n_pairs = n_frames - 1

    # Probe size from the first frame.
    first = cv2.imread(str(frames_dir / f"frame_{0:06d}.jpg"))
    if first is None:
        raise FileNotFoundError(f"missing frame_000000.jpg in {frames_dir}")
    H, W = first.shape[:2]

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))

    pairs_stats: list[dict] = []
    typer.echo(
        f"[flow-video] backend={backend} n_pairs={n_pairs} "
        f"fps={fps} road_classes={sorted(class_ids)}"
    )

    bgr_a = first
    for i in tqdm(range(n_pairs), desc="flow-video"):
        bgr_b = cv2.imread(str(frames_dir / f"frame_{i + 1:06d}.jpg"))
        if bgr_b is None:
            raise FileNotFoundError(f"missing frame_{i + 1:06d}.jpg")
        flow = compute_flow_pair(bgr_a, bgr_b, backend=backend)

        sem = load_frame_entry(perception_cache, i)["semantic"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = road_mask(sem, class_ids)
        fit = fit_road_homography(
            flow, mask, grid_stride=grid_stride, ransac_thresh=ransac_thresh,
        )
        panel = _render_video_frame(
            bgr_a, flow, mask, fit,
            frame_idx=i, n_frames=n_frames, arrow_stride=arrow_stride,
        )
        writer.write(panel)
        pairs_stats.append({
            "i_a": int(i),
            "i_b": int(i + 1),
            "n_road_px": int(fit.n_road_px),
            "n_candidates": int(fit.src_xy.shape[0]),
            "n_inliers": int(fit.inlier_mask.sum()),
            "sampson_median_px": (
                float(fit.sampson_median) if np.isfinite(fit.sampson_median) else None
            ),
            "homography": (fit.H.tolist() if fit.H is not None else None),
        })
        bgr_a = bgr_b

    writer.release()
    payload = {
        "backend": backend,
        "perception_cache": str(perception_cache),
        "frames_dir": str(frames_dir),
        "grid_stride": grid_stride,
        "ransac_thresh": ransac_thresh,
        "road_class_ids": sorted(class_ids),
        "fps": fps,
        "n_pairs": n_pairs,
        "pairs": pairs_stats,
    }
    stats_json.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[flow-video] wrote {out_mp4} ({n_pairs} frames @ {fps} fps) + {stats_json}")
    return out_mp4
