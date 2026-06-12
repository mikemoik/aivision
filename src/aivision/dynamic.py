"""Geometry-based dynamic-object detection and tracking.

Detects independently-moving image regions without trusting the (flickery)
per-frame semantic label, then tracks them through the flow field so the
dynamic decision is confirmed by temporal evidence rather than smoothed.

Per consecutive pair (t, t+1) we have observed dense flow ``f_obs``, the fused
inverse-depth map ``rho_t`` (metric Z = 1/rho), and the relative camera pose
from the metric ORB-SLAM3 trajectory. Two independent residuals score "is this
pixel moving on its own?":

  - EPIPOLAR (Sampson) residual — depth-FREE. A static point obeys the
    epipolar constraint whatever its depth, so the bad-depth sky reads ~0 even
    when its fused depth is a wrong ~55 m. Blind only to motion *along* the
    epipole (a vehicle straight ahead).

  - REPROJECTION residual — depth-BASED. |f_obs - f_induced| where f_induced
    is the rigid flow predicted by reprojecting Z·K⁻¹x through the pose. Covers
    the forward-epipole degeneracy the epipolar test misses, but is
    contaminated wherever depth is wrong — so it is gated to the forward cone
    around the epipole, where (a) epipolar is blind and (b) we genuinely need
    it. Outside the cone the epipolar residual decides alone, which is what
    keeps the wrong-depth sky from false-firing.

Detection is per movable segmentation *instance* (region), not per label: the
median residual over a whole region votes static/dynamic as a unit, so a
single-frame label flip cannot strobe the mask. Regions are linked
frame-to-frame by forward-warping their masks through the flow and matching by
IoU; each track accumulates a sliding window of votes and is *confirmed*
dynamic only once enough votes agree.
"""

from __future__ import annotations

import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .flow import compute_flow_pair
from .kitti import load_sequence
from .perception import (
    CacheManifest,
    MANIFEST_NAME,
    id2label_from_manifest,
    load_frame_entry,
)
from .pocs import common


# Movable Vistas classes — the only ones that can be dynamic. Static structure
# (road, building, vegetation, poles, ...) is never tested. The flickery
# sky-as-Car blob IS a Car instance, so it gets tested and — being static —
# voted static by its epipolar residual, which is the whole point.
MOVABLE_CLASS_NAMES: frozenset[str] = frozenset({
    "Bird", "Ground Animal",
    "Person", "Bicyclist", "Motorcyclist", "Other Rider",
    "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle",
    "On Rails", "Other Vehicle", "Trailer", "Truck", "Wheeled Slow",
})


# ---------------------------------------------------------------------------
# Geometry primitives


def skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson_distance(F: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """First-order epipolar (Sampson) distance in pixels. src/dst: (N,2)."""
    s = np.hstack([src, np.ones((src.shape[0], 1))])
    d = np.hstack([dst, np.ones((dst.shape[0], 1))])
    Fs = (F @ s.T).T
    Ftd = (F.T @ d.T).T
    num = np.einsum("ij,ij->i", d, Fs) ** 2
    den = Fs[:, 0] ** 2 + Fs[:, 1] ** 2 + Ftd[:, 0] ** 2 + Ftd[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


@dataclass
class PairGeometry:
    """All per-pair residual fields plus the epipole direction for cone gating."""

    epi: np.ndarray          # H×W epipolar Sampson residual (px)
    rep: np.ndarray          # H×W reprojection residual (px)
    rep_valid: np.ndarray    # H×W bool — depth valid & in front of cam_{t+1}
    flow: np.ndarray         # H×W×2 observed flow (for tracking warp)
    epi_dir: np.ndarray      # (3,) unit viewing direction of the epipole in cam_t


def compute_pair_geometry(
    bgr_a: np.ndarray,
    bgr_b: np.ndarray,
    rho_pix: np.ndarray,
    K: np.ndarray,
    ray: np.ndarray,
    src_xy: np.ndarray,
    T_rel: np.ndarray,
    *,
    backend: str = "raft_large",
) -> PairGeometry:
    H, W = bgr_a.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]

    flow = compute_flow_pair(bgr_a, bgr_b, backend=backend)

    # epipolar (depth-free)
    E = skew(t_rel) @ R_rel
    F = Kinv.T @ E @ Kinv
    dst_xy = src_xy + flow.reshape(-1, 2)
    epi = sampson_distance(F, src_xy, dst_xy).reshape(H, W)

    # reprojection (depth-based)
    rho = (rho_pix.astype(np.float32) - 1.0) / 65534.0
    valid = rho_pix > 0
    Z = np.where(valid, 1.0 / np.maximum(rho, 1e-6), 0.0)
    P_t = ray * Z[..., None]
    P_j = P_t @ R_rel.T + t_rel
    Zj = np.maximum(P_j[..., 2], 1e-6)
    uu = src_xy[:, 0].reshape(H, W)
    vv = src_xy[:, 1].reshape(H, W)
    u_j = fx * P_j[..., 0] / Zj + cx
    v_j = fy * P_j[..., 1] / Zj + cy
    induced = np.stack([u_j - uu, v_j - vv], axis=-1)
    rep = np.linalg.norm(flow - induced, axis=-1)
    rep_valid = valid & (P_j[..., 2] > 0)

    # epipole direction (cam_{t+1} centre seen from cam_t): -R_rel^T t_rel
    epi_dir = -R_rel.T @ t_rel
    n = np.linalg.norm(epi_dir)
    epi_dir = epi_dir / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return PairGeometry(epi=epi, rep=rep, rep_valid=rep_valid, flow=flow, epi_dir=epi_dir)


# ---------------------------------------------------------------------------
# Region detection (per movable instance)


@dataclass
class Region:
    cls: int
    mask: np.ndarray         # H×W bool
    area: int
    bbox: tuple[int, int, int, int]   # x0,y0,x1,y1
    centroid: tuple[float, float]
    epi_med: float
    rep_med: float
    in_cone: bool
    is_dynamic: bool         # instantaneous decision


def detect_regions(
    sem: np.ndarray,
    inst: np.ndarray,
    geo: PairGeometry,
    movable_ids: set[int],
    ray: np.ndarray,
    *,
    min_area: int = 250,
    epi_thresh: float = 1.0,
    rep_thresh: float = 10.0,
    cone_deg: float = 22.0,
) -> list[Region]:
    """Motion-driven detection: a per-pixel "moving" mask is built from the
    gated residuals *within movable-class pixels only*, then connected
    components on that mask give the moving blobs.

    Why per-pixel + CC rather than per-instance median: the flickery
    sky-as-Car blob merges with real cars into one huge instance whose
    median residual is washed out by static sky pixels. Scoring each pixel
    and keeping only the moving ones makes the static sky drop out and the
    genuinely-moving pixels cluster into their own component, independent of
    how the (unreliable) instance map happened to group them.

    Restricting to movable-class pixels suppresses the epipolar residual's
    structured noise at static depth/occlusion edges (building corners etc.),
    which would otherwise spawn spurious blobs.
    """
    H, W = sem.shape
    movable = np.isin(sem, list(movable_ids))
    if not movable.any():
        return []

    # per-pixel forward-epipole cone mask (ray within cone_deg of epipole dir)
    rn = ray / (np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-9)
    cos_ang = rn @ geo.epi_dir
    cone = cos_ang >= np.cos(np.deg2rad(cone_deg))

    # per-pixel dynamic: epipolar everywhere; reprojection only inside the
    # cone (epipolar's blind spot) and only where depth is valid.
    dyn_px = movable & (geo.epi > epi_thresh)
    dyn_px |= movable & cone & geo.rep_valid & (geo.rep > rep_thresh)

    # de-speckle, then connected components on the moving pixels
    dyn_u8 = cv2.morphologyEx(dyn_px.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(dyn_u8, connectivity=8)

    regions: list[Region] = []
    for k in range(1, n_lbl):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # grow the moving core back onto its full movable blob so the rendered
        # mask covers the whole object — but cap the growth: if the movable
        # component is the merged sky-as-Car monster (>> the moving core),
        # keep the core (dilated) instead of flooding the sky red.
        core = lbl == k
        core_classes = sem[core]
        cls = int(np.bincount(core_classes).argmax())
        full = movable & (sem == cls)
        full_lbl = cv2.connectedComponentsWithStats(full.astype(np.uint8), connectivity=8)[1]
        touch = np.unique(full_lbl[core])
        touch = touch[touch > 0]
        grown = np.isin(full_lbl, touch) if touch.size else core
        if int(grown.sum()) <= area * 4:
            mask = grown
        else:
            mask = cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)

        ys, xs = np.where(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cxr, cyr = float(xs.mean()), float(ys.mean())
        # report residuals over the MOVING core (not the grown mask)
        epi_med = float(np.median(geo.epi[core]))
        rv = core & geo.rep_valid
        rep_med = float(np.median(geo.rep[rv])) if rv.any() else float("nan")
        in_cone = bool(cone[int(round(cyr)), int(round(cxr))])
        regions.append(Region(
            cls=cls, mask=mask, area=int(mask.sum()), bbox=(x0, y0, x1, y1),
            centroid=(cxr, cyr), epi_med=epi_med, rep_med=rep_med,
            in_cone=in_cone, is_dynamic=True,
        ))
    return regions


# ---------------------------------------------------------------------------
# Tracker


@dataclass
class Track:
    tid: int
    cls: int
    mask: np.ndarray                      # current-frame mask (propagated each step)
    votes: deque                          # recent instantaneous dynamic votes (bool)
    epi_med: float = 0.0
    rep_med: float = float("nan")
    missed: int = 0
    confirmed: bool = False

    def confirmed_dynamic(self, min_votes: int) -> bool:
        return sum(self.votes) >= min_votes


def warp_mask_forward(mask: np.ndarray, flow: np.ndarray, dilate: int = 2) -> np.ndarray:
    """Push a frame-t mask into frame t+1 by scattering each pixel along flow."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros_like(mask)
    nx = np.clip(np.round(xs + flow[ys, xs, 0]).astype(np.int64), 0, W - 1)
    ny = np.clip(np.round(ys + flow[ys, xs, 1]).astype(np.int64), 0, H - 1)
    out = np.zeros((H, W), np.uint8)
    out[ny, nx] = 1
    if dilate > 0:
        out = cv2.dilate(out, np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8))
    return out.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    if inter == 0:
        return 0.0
    return inter / float((a | b).sum())


class DynamicTracker:
    """Greedy IoU tracker with sliding-window dynamic voting."""

    def __init__(
        self,
        *,
        vote_window: int = 5,
        min_votes: int = 2,
        iou_thresh: float = 0.2,
        max_missed: int = 3,
    ) -> None:
        self.tracks: list[Track] = []
        self.vote_window = vote_window
        self.min_votes = min_votes
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self._next_id = 0

    def update(self, regions: list[Region]) -> list[Track]:
        """Match detections to tracks, update votes; return active tracks."""
        # greedy IoU matching: all (track, region) pairs above threshold, best first
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for ri, rg in enumerate(regions):
                if tr.cls != rg.cls:
                    continue
                j = _iou(tr.mask, rg.mask)
                if j >= self.iou_thresh:
                    pairs.append((j, ti, ri))
        pairs.sort(reverse=True)
        used_t, used_r = set(), set()
        for j, ti, ri in pairs:
            if ti in used_t or ri in used_r:
                continue
            used_t.add(ti); used_r.add(ri)
            tr, rg = self.tracks[ti], regions[ri]
            tr.mask = rg.mask
            tr.votes.append(rg.is_dynamic)
            tr.epi_med, tr.rep_med, tr.missed = rg.epi_med, rg.rep_med, 0

        # unmatched tracks age; unmatched detections spawn
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.missed += 1
                tr.votes.append(False)
        for ri, rg in enumerate(regions):
            if ri not in used_r:
                tr = Track(
                    tid=self._next_id, cls=rg.cls, mask=rg.mask,
                    votes=deque([rg.is_dynamic], maxlen=self.vote_window),
                    epi_med=rg.epi_med, rep_med=rg.rep_med,
                )
                self._next_id += 1
                self.tracks.append(tr)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for tr in self.tracks:
            if tr.confirmed_dynamic(self.min_votes):
                tr.confirmed = True
        return self.tracks

    def propagate(self, flow: np.ndarray) -> None:
        """Warp every track mask from frame t into frame t+1 for next matching."""
        for tr in self.tracks:
            tr.mask = warp_mask_forward(tr.mask, flow)


# ---------------------------------------------------------------------------
# Orchestrator / renderer


def _decode_rho_stream(video: Path, W: int, H: int):
    """Generator yielding successive gray16 rho frames as uint16 H×W arrays."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE,
    )
    fb = W * H * 2
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            yield np.frombuffer(raw, dtype=np.uint16).reshape(H, W).copy()
    finally:
        proc.stdout.close()
        proc.wait()


def _draw_frame(
    bgr: np.ndarray, tracks: list[Track], id2label: dict[int, str],
    *, frame_idx: int, n_frames: int, min_votes: int, vote_window: int,
) -> np.ndarray:
    out = bgr.copy().astype(np.float32)
    red = np.array([0, 0, 255], np.float32)
    amber = np.array([0, 170, 255], np.float32)
    n_conf = 0
    overlays = []  # (mask, colour) painted first, labels after
    for tr in tracks:
        if tr.missed > 0:
            continue
        confirmed = tr.confirmed
        tentative = (not confirmed) and len(tr.votes) and tr.votes[-1]
        if not (confirmed or tentative):
            continue
        col = red if confirmed else amber
        overlays.append((tr.mask, col))
        if confirmed:
            n_conf += 1
    for mask, col in overlays:
        a = (mask[..., None].astype(np.float32)) * 0.45
        out = out * (1 - a) + col * a
    out = out.astype(np.uint8)
    for tr in tracks:
        if tr.missed > 0:
            continue
        confirmed = tr.confirmed
        tentative = (not confirmed) and len(tr.votes) and tr.votes[-1]
        if not (confirmed or tentative):
            continue
        x0, y0, x1, y1 = _bbox(tr.mask)
        col = (0, 0, 255) if confirmed else (0, 170, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        name = id2label.get(tr.cls, f"id{tr.cls}")
        tag = "DYN" if confirmed else "?"
        votes = sum(tr.votes)
        cv2.putText(out, f"#{tr.tid} {name} {tag} {votes}/{len(tr.votes)} epi{tr.epi_med:.1f}",
                    (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    lines = [
        f"frame {frame_idx:03d}/{n_frames - 1}  pair {frame_idx}->{frame_idx + 1}",
        f"confirmed dynamic={n_conf}   active tracks={sum(1 for t in tracks if t.missed == 0)}",
        f"vote rule: dynamic if >= {min_votes}/{vote_window}   red=confirmed amber=tentative",
    ]
    pad, lh = 8, 20
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (560, pad * 2 + lh * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, dst=out)
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (pad, pad + 14 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def render_dynamic_tracking_video(
    seq_dir: Path,
    src_dir: Path,
    out_mp4: Path,
    *,
    rho_video: Path | None = None,
    perception: Path | None = None,
    trajectory: Path | None = None,
    backend: str = "raft_large",
    fps: int = 10,
    epi_thresh: float = 1.0,
    rep_thresh: float = 6.0,
    cone_deg: float = 22.0,
    vote_window: int = 5,
    min_votes: int = 2,
) -> Path:
    rho_video = rho_video or (src_dir / "inv_depth_fused_mono_rho_gf_16bit.mkv")
    perception = perception or (src_dir / "perception")
    trajectory = trajectory or (src_dir / "CameraTrajectory.txt")

    seq = load_sequence(seq_dir, camera="image_2")
    W, H, K = seq.width, seq.height, seq.K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    N = seq.n_frames
    poses = common.load_kitti_trajectory(trajectory)
    assert len(poses) == N, f"{len(poses)} poses vs {N} frames"

    manifest = CacheManifest.load(perception / MANIFEST_NAME)
    id2label = id2label_from_manifest(manifest)
    l2i = {v: k for k, v in id2label.items()}
    movable_ids = {l2i[n] for n in MOVABLE_CLASS_NAMES if n in l2i}

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1).astype(np.float64)
    src_xy = np.stack([uu.ravel(), vv.ravel()], axis=-1).astype(np.float64)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter.fourcc(*"mp4v"), fps, (W, H))

    tracker = DynamicTracker(vote_window=vote_window, min_votes=min_votes)
    rho_iter = _decode_rho_stream(rho_video, W, H)

    from tqdm import tqdm
    bgr_a = cv2.imread(str(seq.image_paths[0]), cv2.IMREAD_COLOR)
    rho_a = next(rho_iter)
    n_conf_total = 0
    for t in tqdm(range(N - 1), desc=f"dynamic-track {out_mp4.name}"):
        bgr_b = cv2.imread(str(seq.image_paths[t + 1]), cv2.IMREAD_COLOR)
        T_rel = np.linalg.inv(poses[t + 1]) @ poses[t]
        geo = compute_pair_geometry(bgr_a, bgr_b, rho_a, K, ray, src_xy, T_rel, backend=backend)

        entry = load_frame_entry(perception, t)
        sem, inst = entry["semantic"], entry["instance"]
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
            inst = cv2.resize(inst, (W, H), interpolation=cv2.INTER_NEAREST)
        regions = detect_regions(
            sem, inst, geo, movable_ids, ray,
            epi_thresh=epi_thresh, rep_thresh=rep_thresh, cone_deg=cone_deg,
        )
        tracks = tracker.update(regions)
        frame = _draw_frame(bgr_a, tracks, id2label, frame_idx=t, n_frames=N,
                            min_votes=min_votes, vote_window=vote_window)
        writer.write(frame)
        n_conf_total += sum(1 for tr in tracks if tr.confirmed and tr.missed == 0)

        tracker.propagate(geo.flow)
        bgr_a = bgr_b
        try:
            rho_a = next(rho_iter)
        except StopIteration:
            rho_a = rho_a

    writer.release()
    print(f"[dynamic-track] wrote {out_mp4} ({N - 1} frames @ {fps} fps); "
          f"confirmed-dynamic track-frames total={n_conf_total}")
    return out_mp4
