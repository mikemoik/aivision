"""B4 per-pair verdict overlay — diagnostic for the per-instance motion test.

Runs DISK + LightGlue + B2 sky drop + B4 motion test on a single image pair
under guide poses, then renders the matches with per-verdict colouring so a
human can eyeball whether the per-instance moving/static classification is
sensible.

This is a diagnostic — not part of the SfM pipeline. The pipeline path lives
in `triangulate._disk_lightglue_match`; this module re-implements just
enough of it to capture the per-match verdict instead of merely dropping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .triangulate import (
    _DYNAMIC_VISTAS_NAMES,
    _b4_drop_moving_instances,
    _epipolar_sampson_residuals_px,
)

# Verdict codes.
VERDICT_KEPT = 0
VERDICT_SKY = 1
VERDICT_B4_MOVING = 2


@dataclass
class PairVerdict:
    matches: np.ndarray         # (P, 2) uint32
    verdict: np.ndarray         # (P,) uint8 — VERDICT_*
    sampson_px: np.ndarray      # (P,) float64 — Sampson residual in px
    cls_a: np.ndarray           # (P,) uint8 — semantic class on side a
    cls_b: np.ndarray           # (P,) uint8
    inst_a: np.ndarray          # (P,) uint16
    inst_b: np.ndarray          # (P,) uint16
    kp_a: np.ndarray            # (Na, 2) float32 — all DISK keypoints
    kp_b: np.ndarray            # (Nb, 2)
    img_a: np.ndarray           # H×W×3 BGR
    img_b: np.ndarray
    n_b4_moving_instances: int
    n_b4_static_instances: int
    id2label: dict[int, str]
    name_a: str
    name_b: str


def _frame_name(idx: int) -> str:
    return f"frame_{idx:06d}.jpg"


def _resolve_pose(
    rec: pycolmap.Reconstruction, name: str
) -> tuple[np.ndarray, np.ndarray]:
    for img in rec.images.values():
        if img.name == name:
            pose = img.cam_from_world()
            return pose.rotation.matrix(), np.asarray(pose.translation, dtype=np.float64)
    raise KeyError(f"image {name!r} not in guide recon")


def b4_verdict_pair(
    frames_dir: Path,
    perception_cache: Path,
    guide_recon: Path,
    frame_a_idx: int,
    frame_b_idx: int,
    n_keypoints: int = 8192,
) -> PairVerdict:
    """Run the full B2+B4 path on a single frame pair and return verdicts.

    This loads DISK + LightGlue each call — fine for a one-pair diagnostic,
    too heavy for a video. If a video version is wanted, hoist the model
    setup out of the loop.
    """
    import torch
    import kornia.feature as KF
    from kornia.feature import laf_from_center_scale_ori

    from .perception import KeypointTagger, id2label_from_manifest

    frames_dir = Path(frames_dir)
    perception_cache = Path(perception_cache)
    guide_recon = Path(guide_recon)

    name_a, name_b = _frame_name(frame_a_idx), _frame_name(frame_b_idx)
    path_a, path_b = frames_dir / name_a, frames_dir / name_b
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(f"frames not found: {path_a} / {path_b}")

    img_a = cv2.imread(str(path_a))
    img_b = cv2.imread(str(path_b))
    rgb_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    rgb_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    # Camera intrinsics + guide poses come from the guide recon.
    rec = pycolmap.Reconstruction(str(guide_recon))
    cam = next(iter(rec.cameras.values()))
    focal = float(cam.params[0])
    R1, t1 = _resolve_pose(rec, name_a)
    R2, t2 = _resolve_pose(rec, name_b)

    # Perception cache wiring (B2 sky id + B4 dynamic ids).
    tagger = KeypointTagger(perception_cache)
    id2label = id2label_from_manifest(tagger.manifest)
    name_to_id = {v: int(k) for k, v in id2label.items()}
    sky_id = name_to_id.get("Sky")
    if sky_id is None:
        raise RuntimeError(
            f"'Sky' missing from id2label of {tagger.manifest.seg_model!r}"
        )
    dyn_ids = np.asarray(
        sorted(name_to_id[n] for n in _DYNAMIC_VISTAS_NAMES if n in name_to_id),
        dtype=np.uint8,
    )

    # DISK + LightGlue on this single pair.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disk = KF.DISK.from_pretrained("depth", device=device).eval()
    lg = KF.LightGlueMatcher("disk").to(device).eval()

    def _extract(rgb: np.ndarray) -> tuple[np.ndarray, "torch.Tensor", "torch.Tensor", tuple[int, int]]:
        h, w = rgb.shape[:2]
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device) / 255.0
        with torch.inference_mode():
            f = disk(t, n=n_keypoints, pad_if_not_divisible=True)[0]
        kp = f.keypoints
        desc = f.descriptors
        N = kp.shape[0]
        scale = torch.ones(1, N, 1, 1, device=device)
        ori = torch.zeros(1, N, 1, device=device)
        lafs = laf_from_center_scale_ori(kp[None], scale, ori)
        kp_np = kp.detach().cpu().numpy().astype(np.float32)
        return kp_np, desc, lafs, (h, w)

    kp_a_np, desc_a, laf_a, hw_a = _extract(rgb_a)
    kp_b_np, desc_b, laf_b, hw_b = _extract(rgb_b)

    with torch.inference_mode():
        _, matches = lg(desc_a, desc_b, laf_a, laf_b, hw1=hw_a, hw2=hw_b)
    matches_np = matches.detach().cpu().numpy().astype(np.uint32)
    if matches_np.shape[0] == 0:
        raise RuntimeError(f"LightGlue returned no matches for {name_a} ↔ {name_b}")

    # Tag every match endpoint.
    tags_a = tagger.tag(frame_a_idx, kp_a_np)
    tags_b = tagger.tag(frame_b_idx, kp_b_np)
    cls_a = tags_a.semantic[matches_np[:, 0]]
    cls_b = tags_b.semantic[matches_np[:, 1]]
    inst_a = tags_a.instance[matches_np[:, 0]]
    inst_b = tags_b.instance[matches_np[:, 1]]

    # B2 sky verdict.
    sky_mask = (cls_a == sky_id) | (cls_b == sky_id)

    # Sampson residuals (under guide poses), then B4 on the non-sky subset —
    # matches pipeline ordering.
    kp_a_norm = cam.cam_from_img(kp_a_np.astype(np.float64))
    kp_b_norm = cam.cam_from_img(kp_b_np.astype(np.float64))
    sampson_px = _epipolar_sampson_residuals_px(
        kp_a_norm, kp_b_norm, matches_np, R1, t1, R2, t2, focal,
    )

    keep_for_b4 = ~sky_mask
    if keep_for_b4.any():
        b4_keep, n_mov, n_stat, _ = _b4_drop_moving_instances(
            cls_a[keep_for_b4], inst_a[keep_for_b4],
            cls_b[keep_for_b4], inst_b[keep_for_b4],
            sampson_px[keep_for_b4],
            dyn_ids,
        )
    else:
        b4_keep = np.ones(0, dtype=bool)
        n_mov = n_stat = 0

    verdict = np.full(len(matches_np), VERDICT_KEPT, dtype=np.uint8)
    verdict[sky_mask] = VERDICT_SKY
    non_sky_idx = np.where(keep_for_b4)[0]
    if non_sky_idx.size:
        b4_dropped_local = non_sky_idx[~b4_keep]
        verdict[b4_dropped_local] = VERDICT_B4_MOVING

    return PairVerdict(
        matches=matches_np,
        verdict=verdict,
        sampson_px=sampson_px,
        cls_a=cls_a, cls_b=cls_b,
        inst_a=inst_a, inst_b=inst_b,
        kp_a=kp_a_np, kp_b=kp_b_np,
        img_a=img_a, img_b=img_b,
        n_b4_moving_instances=n_mov,
        n_b4_static_instances=n_stat,
        id2label=id2label,
        name_a=name_a, name_b=name_b,
    )


# BGR colour conventions for the renderer.
_COLOUR_STATIC_KEPT = (160, 160, 160)        # grey — context only
_COLOUR_DYNAMIC_KEPT = (0, 220, 0)           # bright green — passed B4
_COLOUR_DYNAMIC_DROPPED = (40, 40, 230)      # bright red — B4 says moving
_COLOUR_SKY = (220, 220, 40)                 # cyan — sky-class drop
_PAD = 8


def _draw_side_panel(canvas: np.ndarray, pv: PairVerdict) -> None:
    """Draw the legend + summary stats at the top of the canvas."""
    h, w_total = canvas.shape[:2]

    n_total = len(pv.matches)
    n_sky = int((pv.verdict == VERDICT_SKY).sum())
    n_b4 = int((pv.verdict == VERDICT_B4_MOVING).sum())
    n_kept = int((pv.verdict == VERDICT_KEPT).sum())

    # Candidate-dynamic accounting on the non-sky subset.
    dyn_endpoint = (
        np.isin(pv.cls_a, list(_dynamic_ids_from_label_map(pv.id2label))) |
        np.isin(pv.cls_b, list(_dynamic_ids_from_label_map(pv.id2label)))
    )
    n_dyn_total = int((dyn_endpoint & (pv.verdict != VERDICT_SKY)).sum())

    lines = [
        f"{pv.name_a} -> {pv.name_b}",
        f"matches: {n_total}  kept: {n_kept}  sky: {n_sky}  B4: {n_b4}",
        f"candidate-dynamic: {n_dyn_total}  "
        f"B4 instance-keys: moving={pv.n_b4_moving_instances} static={pv.n_b4_static_instances}",
    ]

    y = 18
    for line in lines:
        cv2.putText(
            canvas, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA,
        )
        y += 18

    # Legend swatches in the top-right.
    legend = [
        ("static kept", _COLOUR_STATIC_KEPT),
        ("dynamic kept", _COLOUR_DYNAMIC_KEPT),
        ("dynamic dropped (B4)", _COLOUR_DYNAMIC_DROPPED),
        ("sky dropped (B2)", _COLOUR_SKY),
    ]
    x0 = w_total - 260
    y = 18
    for label, col in legend:
        cv2.rectangle(canvas, (x0, y - 10), (x0 + 14, y + 2), col, -1)
        cv2.putText(
            canvas, label, (x0 + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA,
        )
        y += 16


def _dynamic_ids_from_label_map(id2label: dict[int, str]) -> set[int]:
    inv = {v: int(k) for k, v in id2label.items()}
    return {inv[n] for n in _DYNAMIC_VISTAS_NAMES if n in inv}


def render_b4_verdict(pv: PairVerdict, out_png: Path) -> Path:
    """Side-by-side render of the two frames with per-match verdicts.

    Static kept matches are drawn as small grey dots only (no line) — they
    are the majority and would otherwise drown the dynamic signal in
    spaghetti. Candidate-dynamic and sky matches get full coloured lines.
    """
    img_a = pv.img_a.copy()
    img_b = pv.img_b.copy()
    h, w = img_a.shape[:2]
    canvas = np.zeros((h, 2 * w + _PAD, 3), dtype=np.uint8)
    canvas[:, :w] = img_a
    canvas[:, w + _PAD:] = img_b

    dyn_ids = _dynamic_ids_from_label_map(pv.id2label)
    dyn_endpoint = np.isin(pv.cls_a, list(dyn_ids)) | np.isin(pv.cls_b, list(dyn_ids))

    # Order: static kept first (background), then dynamic kept, then dynamic
    # dropped (red on top), then sky.
    order = [
        # mask, colour, draw_line, dot_radius
        ((pv.verdict == VERDICT_KEPT) & ~dyn_endpoint, _COLOUR_STATIC_KEPT, False, 1),
        ((pv.verdict == VERDICT_KEPT) & dyn_endpoint,  _COLOUR_DYNAMIC_KEPT, True, 2),
        ((pv.verdict == VERDICT_B4_MOVING),            _COLOUR_DYNAMIC_DROPPED, True, 2),
        ((pv.verdict == VERDICT_SKY),                  _COLOUR_SKY, False, 2),
    ]

    for mask, colour, draw_line, r in order:
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        m = pv.matches[idx]
        xy_a = pv.kp_a[m[:, 0]].astype(int)
        xy_b = pv.kp_b[m[:, 1]].astype(int)
        for (xa, ya), (xb, yb) in zip(xy_a, xy_b):
            cv2.circle(canvas, (xa, ya), r, colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, (xb + w + _PAD, yb), r, colour, -1, cv2.LINE_AA)
            if draw_line:
                cv2.line(canvas, (xa, ya), (xb + w + _PAD, yb), colour, 1, cv2.LINE_AA)

    _draw_side_panel(canvas, pv)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    return out_png
