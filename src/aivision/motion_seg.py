"""Dense per-pixel motion segmentation by motion disparity.

Classifies EVERY pixel of a frame-pair (t, t+1) into one of three classes:

    STATIC  = 0   world-fixed (incl. sky: rho~0, zero translational parallax)
    MOVING  = 1   independently moving (non-static)
    UNKNOWN = 2   geometry genuinely can't decide -> abstain, don't guess

This is DENSE, PER-PIXEL, and CLASS-INDEPENDENT. No connected components, no
region voting, no semantic gating -- segmentation never enters the decision.
The signal is pure motion disparity: how far the observed flow departs from the
rigid (static-world) flow that ego-motion + stereo inverse-depth predict.

All quantities live in the image_0 frame, where pose, stereo rho and RAFT flow
are mutually consistent. The mono depth prior is deliberately OUT of the motion
path (it hallucinates finite sky depth); rho here is the stereo-only stream.

Decision (competence-routed; each cue used only where it is trustworthy):

  fb unreliable                       -> UNKNOWN   (occlusion / disocclusion)
  stereo invalid (no rho):
      flow_mag > FLOW_MAX             -> UNKNOWN   (near-field, no depth to trust epi)
      epi > TAU_EPI                   -> MOVING    (off-epipole motion, depth-free)
      else                            -> STATIC    (the sky case: ~zero parallax)
  stereo valid:
      flow_mag > FLOW_MAX             -> MOVING if loom>TAU_LOOM else UNKNOWN
      off-FOE (epipolar competent)    -> MOVING if epi>TAU_EPI else STATIC
      near-FOE (epipolar blind):
          loom_strong & loom>TAU_LOOM -> MOVING    (SNR-gated looming; the only cue)
          had a strong looming cue     -> STATIC
          else                        -> UNKNOWN   (no SNR -> FOE degeneracy)

The deficit cue (deviation/B) is NOT used: it fires identically on a real mover
and on a wrong-depth static structure, and was ~99% of the per-pixel FOE false
positives. Looming is robust to consistent stereo-rho error (it cancels in
drho/dt), so it is the sole in-cone moving cue.

Calibrated on KITTI seq04 in dynamic_probe.py / parallax_probe.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aivision.dynamic import skew, sampson_distance

# class codes. POT_DYN (potentially dynamic) is produced only by the object layer
# (detect.py / the renderer), never by the dense classify() -- a mover-class
# detection is never plain static; at minimum it is pot_dyn. MOVING == dynamic.
STATIC, MOVING, UNKNOWN, POT_DYN = 0, 1, 2, 3

# calibrated thresholds (dynamic_probe.py, render_dynamic_veto.py)
TAU_EPI = 2.0          # px Sampson; static bg <0.9, real lateral movers 3-6
CONE_DEG = 15.0        # epipolar-blind cone half-angle around the FOE direction
B_MIN = 2.0            # px parallax budget below which the deficit ratio abstains
B_MAX = 8.0            # px above which near+lateral depth error swamps loom/d
TAU_D = 0.5            # deficit ratio threshold
TAU_LOOM = 0.6         # looming (drho/dt) threshold; static ~0.03, leader ~1.0
RHODOT_FLOOR = 5e-4    # min static rho-rate for a meaningful looming ratio
LOOM_SNR = 3.0         # require rhodot_static > LOOM_SNR*floor before trusting loom
TAU_FB = 1.5           # px floor for the fwd/back flow inconsistency gate
FB_FRAC = 0.05         # ...plus this fraction of the flow magnitude (scale-relative)
FLOW_MAX = 80.0        # px above which near-field error amplification breaks epi
TAU_STATIC = 0.3       # max |loom_free| to rescue UNKNOWN -> STATIC (flow-free test)
MIN_MOVE_AREA = 40     # px; moving components smaller than this are FOE/flow noise
TEMP_WINDOW = 2        # +/-N pairs for temporal voting
TEMP_M_MOVE = 2        # min MOVING votes in the window to confirm moving
TEMP_M_STATIC = 2      # min STATIC votes in the window to confirm static


@dataclass
class MotionFields:
    """Per-pixel motion-disparity residuals for one frame pair (all H x W)."""

    epi: np.ndarray         # epipolar Sampson residual (px), depth-free
    d: np.ndarray           # deviation / parallax-budget ratio
    B: np.ndarray           # predicted parallax budget |f_static - f_rot| (px)
    loom: np.ndarray        # looming ratio (rhodot_static - rhodot_obs)/rhodot_static
    loom_valid: np.ndarray  # bool: looming ratio is meaningful here
    loom_strong: np.ndarray  # bool: looming ratio has adequate rhodot_static SNR
    cone: np.ndarray        # bool: inside the epipolar-blind FOE cone
    fb_ok: np.ndarray       # bool: fwd/back flow consistent (not occluded)
    valid: np.ndarray       # bool: stereo rho available
    flow: np.ndarray        # H x W x 2 observed flow
    flow_mag: np.ndarray    # |flow| (px)
    loom_free: np.ndarray   # flow-free looming residual at the rigid-predicted dest
    loom_free_valid: np.ndarray  # bool: loom_free is meaningful (testable)


def _fb_inconsistency(f_ab: np.ndarray, f_ba: np.ndarray) -> np.ndarray:
    """|f_ab(x) + f_ba(x + f_ab(x))| -- high where flow is unreliable."""
    H, W = f_ab.shape[:2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    xb = np.clip(uu + f_ab[..., 0], 0, W - 1).astype(np.float32)
    yb = np.clip(vv + f_ab[..., 1], 0, H - 1).astype(np.float32)
    bx = cv2.remap(f_ba[..., 0], xb, yb, cv2.INTER_LINEAR)
    by = cv2.remap(f_ba[..., 1], xb, yb, cv2.INTER_LINEAR)
    return np.sqrt((f_ab[..., 0] + bx) ** 2 + (f_ab[..., 1] + by) ** 2)


def compute_fields(
    f_obs: np.ndarray,
    f_ba: np.ndarray,
    rho: np.ndarray,
    rho_next: np.ndarray,
    K: np.ndarray,
    T_rel: np.ndarray,
) -> MotionFields:
    """Compute the dense motion-disparity residual fields for one frame pair.

    f_obs : forward flow t -> t+1 (H,W,2)
    f_ba  : backward flow t+1 -> t (H,W,2), for the occlusion gate
    rho, rho_next : stereo inverse depth at t and t+1 (0 = invalid)
    T_rel : relative pose inv(pose_{t+1}) @ pose_t (4x4)
    """
    H, W = rho.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    uu = uu.astype(np.float64)
    vv = vv.astype(np.float64)
    ray = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1)
    src_xy = np.stack([uu.ravel(), vv.ravel()], -1)
    grid = np.stack([uu, vv], -1)

    fb = _fb_inconsistency(f_obs, f_ba)
    flow_mag = np.linalg.norm(f_obs, axis=-1)

    # #1 epipolar Sampson (depth-free)
    F = Kinv.T @ skew(t_rel) @ R_rel @ Kinv
    epi = sampson_distance(F, src_xy, src_xy + f_obs.reshape(-1, 2)).reshape(H, W)

    # rigid field at rho and at infinity -> deviation, budget B, ratio d
    valid = rho > 0
    Rr = ray @ R_rel.T
    P = Rr + rho[..., None] * t_rel

    def proj(Pm):
        z = np.maximum(Pm[..., 2], 1e-6)
        return np.stack([fx * Pm[..., 0] / z + cx, fy * Pm[..., 1] / z + cy], -1)

    xy_static = proj(P)
    f_static = xy_static - grid
    f_rot = proj(Rr) - grid
    deviation = np.linalg.norm(f_obs - f_static, axis=-1)
    B = np.linalg.norm(f_static - f_rot, axis=-1)
    d = deviation / np.maximum(B, B_MIN)

    # #3 looming: static-predicted rho-rate vs observed (warp rho_{t+1} back)
    Pz = np.maximum(P[..., 2], 1e-6)
    rho_pred = rho / Pz                                  # rigid-predicted rho at t+1
    rhodot_static = rho_pred - rho

    # flow-FREE static test: sample stereo rho_{t+1} at the RIGID-predicted dest
    # (no observed flow), then form the SAME looming residual as #3 but warped by
    # the rigid model instead of the observed flow. Normalising by the EXPECTED
    # depth change (rhodot_static) is essential: a distance-keeping far leader has
    # a near-zero ABSOLUTE depth change yet loom_free ~ 1 (it isn't approaching as
    # a static point would), so it is NOT mistaken for static. Used to rescue
    # UNKNOWN -> STATIC where RAFT flow is too unreliable to trust.
    sx = xy_static[..., 0].astype(np.float32)
    sy = xy_static[..., 1].astype(np.float32)
    rho_rigid_next = cv2.remap(rho_next, sx, sy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    rhodot_rigid = rho_rigid_next - rho
    loom_free_valid = valid & (rho_rigid_next > 0) & (rhodot_static > RHODOT_FLOOR)
    loom_free = np.where(
        loom_free_valid,
        np.abs(rhodot_static - rhodot_rigid) / np.maximum(rhodot_static, RHODOT_FLOOR),
        np.inf,
    )
    wx = np.clip(uu + f_obs[..., 0], 0, W - 1).astype(np.float32)
    wy = np.clip(vv + f_obs[..., 1], 0, H - 1).astype(np.float32)
    rho_obs_next = cv2.remap(rho_next, wx, wy, cv2.INTER_LINEAR)
    rhodot_obs = rho_obs_next - rho
    loom_valid = valid & (rho_obs_next > 0) & (rhodot_static > RHODOT_FLOOR)
    loom = np.where(
        loom_valid,
        (rhodot_static - rhodot_obs) / np.maximum(rhodot_static, RHODOT_FLOOR),
        0.0,
    )
    # only trust the looming ratio where the EXPECTED depth change is well above
    # the noise floor. A wrong-but-consistent stereo rho gives loom~0 (it cancels),
    # so looming is robust to depth error -- unlike the deficit cue, which fires on
    # exactly such wrong-depth static structures (predicted parallax, none observed).
    loom_strong = loom_valid & (rhodot_static > LOOM_SNR * RHODOT_FLOOR)

    # epipolar-blind cone around the FOE direction
    epi_dir = -R_rel.T @ t_rel
    n = np.linalg.norm(epi_dir)
    epi_dir = epi_dir / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    rn = ray / (np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-9)
    cone = (rn @ epi_dir) >= np.cos(np.deg2rad(CONE_DEG))
    # scale-relative occlusion gate: a flat TAU_FB nukes ~30% of a fast forward
    # drive to UNKNOWN (large honest flow has >1.5px fwd/back noise without being
    # occluded). Allow ~FB_FRAC of the displacement, floored at TAU_FB so small-
    # flow regions keep the tight catch.
    fb_ok = fb < np.maximum(TAU_FB, FB_FRAC * flow_mag)

    return MotionFields(
        epi=epi, d=d, B=B, loom=loom, loom_valid=loom_valid, cone=cone,
        fb_ok=fb_ok, valid=valid, flow=f_obs, flow_mag=flow_mag,
        loom_free=loom_free, loom_free_valid=loom_free_valid,
        loom_strong=loom_strong,
    )


def classify(g: MotionFields) -> np.ndarray:
    """Per-pixel 3-class label map (uint8: STATIC/MOVING/UNKNOWN). Raw, no denoising.

    Branches are evaluated as a partition over (fb_ok, valid, flow_mag, cone),
    so the assignments below are mutually exclusive -- no priority ordering needed.
    """
    epi, B = g.epi, g.B
    loom, ls = g.loom, g.loom_strong
    valid, cone, fb_ok = g.valid, g.cone, g.fb_ok
    fast = g.flow_mag > FLOW_MAX

    label = np.full(epi.shape, UNKNOWN, dtype=np.uint8)

    # --- flow unreliable: abstain everywhere ---
    bad = ~fb_ok
    near = B < B_MAX                       # parallax budget small enough to trust loom

    # The DEFICIT cue (deviation/B) is deliberately NOT used for the moving
    # decision: it fires identically on a real mover (parallax cancelled by motion)
    # and on a wrong-depth static structure (spurious stereo rho predicts parallax
    # that was never there) -- the latter was 99% of the per-pixel false positives
    # (the "red sky" blob near the FOE). Looming is robust to that (consistent rho
    # error cancels in drho/dt), so it is the ONLY in-cone moving cue, SNR-gated.

    # --- stereo invalid (no depth): depth-free reasoning only ---
    inv = fb_ok & ~valid
    label[inv & fast] = UNKNOWN            # near-field, epi unreliable w/o depth
    label[inv & ~fast & (epi > TAU_EPI)] = MOVING
    label[inv & ~fast & (epi <= TAU_EPI)] = STATIC   # sky: ~zero parallax at infinity

    # --- stereo valid, near-field (huge flow): only strong looming survives ---
    vf = fb_ok & valid & fast
    vmov = ls & near & (loom > TAU_LOOM)
    label[vf & vmov] = MOVING
    label[vf & ~vmov] = UNKNOWN

    # --- stereo valid, off-FOE: epipolar is competent ---
    off = fb_ok & valid & ~fast & ~cone
    label[off & (epi > TAU_EPI)] = MOVING
    label[off & (epi <= TAU_EPI)] = STATIC

    # --- stereo valid, near-FOE (epipolar blind): SNR-gated looming decides ---
    foe = fb_ok & valid & ~fast & cone
    moving = ls & near & (loom > TAU_LOOM)             # looming only, strong SNR
    competent = ls & near                              # a trustworthy looming cue existed
    label[foe & moving] = MOVING
    label[foe & ~moving & competent] = STATIC
    label[foe & ~moving & ~competent] = UNKNOWN        # no SNR -> abstain (FOE degeneracy)

    label[bad] = UNKNOWN                   # flow unreliable -> base verdict UNKNOWN

    # flow-FREE static rescue: promote UNKNOWN -> STATIC where the rigid-warped
    # looming residual is near zero (the pixel approaches exactly as a static
    # point should). Pure geometry (pose + stereo rho), no observed flow -> valid
    # exactly where the flow gate abstained. Can only ADD static, never overrides
    # MOVING/STATIC, and the looming normalisation keeps a moving leader out.
    static_ok = g.loom_free_valid & (g.loom_free < TAU_STATIC)
    label[(label == UNKNOWN) & static_ok] = STATIC
    return label


def despeckle(label: np.ndarray, g: MotionFields, min_area: int = MIN_MOVE_AREA) -> np.ndarray:
    """Remove isolated MOVING speckle (the per-pixel FOE-cone noise).

    The far-leader true positive and the FOE-noise false positive both carry tiny
    absolute signals, so no per-pixel threshold separates them -- but the noise is
    incoherent (isolated < min_area blobs) while a real mover is a dense blob.
    Drop sub-min_area moving components and reclassify those pixels by the flow-
    free static test (STATIC if rigid-depth-consistent, else honest UNKNOWN). This
    is a trivial morphological cleanup, NOT semantic region voting.
    """
    moving = (label == MOVING).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
    small = np.zeros(label.shape, dtype=bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            small |= lbl == k
    out = label.copy()
    static_ok = g.loom_free_valid & (g.loom_free < TAU_STATIC)
    out[small & static_ok] = STATIC
    out[small & ~static_ok] = UNKNOWN
    return out


# ----------------------------------------------------------------------------
# Temporal +/-N persistence (bidirectional flow-warped voting)
#
# Per-pair labels live in their own frame's coordinates. To vote at frame t we
# chain the stored forward/backward flows to bring each neighbour's label into
# frame t, then take a hard per-class vote. This (a) suppresses 1-frame MOVING
# false positives (require >=m_move votes), (b) rescues UNKNOWN -> STATIC where
# neighbours agree static (the near-field road that is clean on some pairs), and
# (c) provides the aggregation substrate the noisy mono looming rescue needs.
# ----------------------------------------------------------------------------

def _advance_coord(coord: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Advance a frame-t->frame-a coordinate field by one inter-frame flow."""
    cx = coord[..., 0].astype(np.float32)
    cy = coord[..., 1].astype(np.float32)
    fx = cv2.remap(flow[..., 0], cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fy = cv2.remap(flow[..., 1], cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return coord + np.stack([fx, fy], -1)


def _coords_t_to_w(t: int, w: int, fwds: list, bwds: list, grid: np.ndarray) -> np.ndarray:
    """Sample positions in frame w for every pixel of frame t (chained flow).

    fwds[a] = flow frame a -> a+1 ; bwds[a] = flow frame a+1 -> a (pair a).
    """
    coord = grid.copy()
    if w > t:
        for a in range(t, w):
            coord = _advance_coord(coord, fwds[a])
    elif w < t:
        for a in range(t, w, -1):
            coord = _advance_coord(coord, bwds[a - 1])
    return coord


def temporal_vote(labels: list, fwds: list, bwds: list,
                  window: int = TEMP_WINDOW, m_move: int = TEMP_M_MOVE,
                  m_static: int = TEMP_M_STATIC) -> list:
    """Bidirectional flow-warped per-class voting over +/-window pairs.

    labels[t] : per-pair HxW uint8 label in frame-t coords (despeckled).
    fwds[t]   : forward flow t -> t+1.  bwds[t] : backward flow t+1 -> t.
    Returns a list of voted HxW uint8 labels, one per input pair.
    """
    n = len(labels)
    H, W = labels[0].shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    grid = np.stack([uu, vv], -1).astype(np.float32)
    out = []
    for t in range(n):
        move_votes = np.zeros((H, W), np.int16)
        static_votes = np.zeros((H, W), np.int16)
        for w in range(max(0, t - window), min(n, t + window + 1)):
            if w == t:
                warped = labels[t]
            else:
                coord = _coords_t_to_w(t, w, fwds, bwds, grid)
                warped = cv2.remap(labels[w], coord[..., 0].astype(np.float32),
                                   coord[..., 1].astype(np.float32),
                                   cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=UNKNOWN)
            move_votes += (warped == MOVING)
            static_votes += (warped == STATIC)
        voted = np.full((H, W), UNKNOWN, dtype=np.uint8)
        voted[static_votes >= m_static] = STATIC
        voted[move_votes >= m_move] = MOVING        # moving wins ties (precision of recall)
        out.append(voted)
    return out
