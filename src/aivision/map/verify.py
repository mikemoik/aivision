"""Map-diff verifier: compare two ADAS maps in a common ENU frame.

The map-generation method is non-deterministic (ORB-SLAM3, depth fusion,
Mask2Former), so a regenerated map is never bit-identical to the saved
ground truth. The acceptance criterion is therefore *stability*, not
reproducibility: does the regenerated map describe the same world, within
tolerance?

The two maps live in their own gauge-free map frames ``M`` (each run's ORB
world is arbitrary up to a rigid+scale transform), so they cannot be compared
in ``M`` directly. Instead each map is lifted into the shared Earth frame
(local ENU) via its *own* stored georeference, and the comparison happens
there. This is exactly the "traceability over reproducibility" thesis: the
method may wander, but the georeferenced map must not.

What is compared:
  * per-class nearest-neighbour object matching within a tolerance radius →
    matched / missing (GT only) / spurious (new only) + position RMSE;
  * the georef transform delta (rotation angle, translation, scale, offset);
  * lane polyline lateral deviation (point-to-polyline, both directions);
  * road-arrow centroid + heading (yaw in the ENU ground plane) delta.

A loose, configurable :class:`VerifyTolerances` drives the PASS/FAIL verdict;
visual inspection of the rendered overlays remains the real acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .frame import GeoReference
from .geometry import GaussianPoint, GaussianPose, LineSegment, OrientedBox, Polyline
from .map_adas import MapAdas
from .objects import MapClass, MapObject

# Classes compared by simple centre nearest-neighbour matching. Lanes and
# arrows get richer, dedicated comparisons below.
_POINT_CLASSES = (
    MapClass.POLE,
    MapClass.SIGN,
    MapClass.TRAFFIC_LIGHT,
    MapClass.GUIDE_POST,
)


@dataclass
class VerifyTolerances:
    """Loose, stability-oriented thresholds for the PASS/FAIL verdict.

    The method (ORB-SLAM3 + stereo/mono depth fusion + Mask2Former) is
    non-deterministic, so these are deliberately loose: the gate asks whether
    the *map* is stable, not whether the *method* is reproducible. The design
    radius is 1-2 m (res_m + ATE 0.19 m). Off-road point objects triangulated
    from fused depth (poles/lights/guide posts) carry ~1.5 m run-to-run scatter
    — the depth-triangulation noise floor — while road-surface elements
    (lanes, arrows) reproduce to ~0.3-0.4 m.

    The georef *rotation* delta is intentionally NOT a gate: it is the ORB
    gauge freedom (each run's world frame is arbitrary up to a rigid rotation),
    fully absorbed by comparing in ENU. It is reported for visibility only.
    Georef *scale* and *translation* ARE gated — both runs anchor to the same
    OXTS datum, so those must stay stable.
    """

    match_r_m: float = 2.0          # max centre distance to call two objects the same
    pos_rms_m: float = 1.60         # max matched-object position RMSE (depth-triangulation floor)
    min_match_frac: float = 0.80    # min fraction of GT objects matched (per total)
    max_spurious_frac: float = 0.25  # max spurious-as-fraction-of-GT
    georef_scale_pct: float = 1.0   # max georef scale delta (%)
    georef_t_m: float = 1.0         # max georef translation delta
    lane_lateral_m: float = 1.00    # max lane polyline lateral deviation
    arrow_yaw_deg: float = 10.0     # max arrow heading delta


# -- representative geometry → a single point in frame M ----------------
def representative_point(obj: MapObject) -> np.ndarray | None:
    """A single point standing in for an object's location, in frame ``M``."""
    slot = next(iter(obj.geometry))
    g = obj.geometry[slot]
    if isinstance(g, GaussianPoint):
        return g.mean
    if isinstance(g, GaussianPose):
        return g.translation
    if isinstance(g, OrientedBox):
        return g.pose.translation
    if isinstance(g, LineSegment):
        return 0.5 * (g.a.mean + g.b.mean)
    if isinstance(g, Polyline) and g.verts:
        return np.mean([v.mean for v in g.verts], axis=0)
    return None


def _linear(gr: GeoReference) -> np.ndarray:
    return np.asarray(gr.T_M_to_ENU, dtype=float)[:3, :3]


def _scale(gr: GeoReference) -> float:
    """Recover the similarity scale from the georef linear block."""
    return float(np.cbrt(np.linalg.det(_linear(gr))))


def _direction_to_enu(gr: GeoReference, v_M: np.ndarray) -> np.ndarray:
    """Rotate a direction (no translation/offset) from M into ENU, unit length."""
    d = _linear(gr) @ np.asarray(v_M, dtype=float)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else d


def _enu_points(m: MapAdas, objs: list[MapObject]) -> np.ndarray:
    """Representative points of ``objs`` lifted into ENU via ``m``'s georef."""
    pts = [representative_point(o) for o in objs]
    pts = [p for p in pts if p is not None]
    if not pts:
        return np.empty((0, 3))
    return m.to_enu(np.asarray(pts, dtype=float))


# -- per-class point matching -------------------------------------------
def _match(gt_enu: np.ndarray, new_enu: np.ndarray, r: float) -> dict:
    """Greedy nearest-neighbour match GT→new within radius ``r`` (m)."""
    used: set[int] = set()
    devs: list[float] = []
    matched_pairs: list[tuple[int, int]] = []
    for gi, g in enumerate(gt_enu):
        if not len(new_enu):
            break
        d = np.linalg.norm(new_enu - g, axis=1)
        for j in np.argsort(d):
            if int(j) in used:
                continue
            if d[j] <= r:
                used.add(int(j))
                devs.append(float(d[j]))
                matched_pairs.append((gi, int(j)))
            break
    devs_a = np.asarray(devs)
    return {
        "n_gt": int(len(gt_enu)),
        "n_new": int(len(new_enu)),
        "matched": int(len(devs)),
        "missing": int(len(gt_enu) - len(devs)),
        "spurious": int(len(new_enu) - len(used)),
        "dev_max": float(devs_a.max()) if len(devs_a) else 0.0,
        "dev_rms": float(np.sqrt((devs_a**2).mean())) if len(devs_a) else 0.0,
        "pairs": matched_pairs,
    }


# -- georef delta --------------------------------------------------------
def _georef_delta(gt: GeoReference, new: GeoReference) -> dict:
    s_gt, s_new = _scale(gt), _scale(new)
    R_gt = _linear(gt) / s_gt
    R_new = _linear(new) / s_new
    angle = float(np.degrees(
        Rotation.from_matrix(R_gt.T @ R_new).magnitude()))
    t_gt = np.asarray(gt.T_M_to_ENU)[:3, 3]
    t_new = np.asarray(new.T_M_to_ENU)[:3, 3]
    return {
        "angle_deg": angle,
        "scale_gt": s_gt,
        "scale_new": s_new,
        "scale_pct": float(abs(s_new - s_gt) / s_gt * 100.0),
        "t_delta_m": float(np.linalg.norm(t_gt - t_new)),
        "offset_delta_m": float(np.linalg.norm(
            np.asarray(gt.enu_offset) - np.asarray(new.enu_offset))),
    }


# -- lanes ---------------------------------------------------------------
def _polyline_enu(m: MapAdas, obj: MapObject) -> np.ndarray:
    line = obj.geometry["line"]
    return m.to_enu(np.asarray([v.mean for v in line.verts], dtype=float))


def _point_to_polyline(p: np.ndarray, poly: np.ndarray) -> float:
    """Min distance from point ``p`` to a polyline (set of segments)."""
    best = np.inf
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _lane_deviation(gt_m: MapAdas, new_m: MapAdas) -> list[dict]:
    """Symmetric point-to-polyline lateral deviation per matched lane side."""
    out: list[dict] = []
    gt_lanes = {o.attributes.get("side"): o for o in gt_m.of_class(MapClass.LANE_MARKING)}
    new_lanes = {o.attributes.get("side"): o for o in new_m.of_class(MapClass.LANE_MARKING)}
    for side in sorted(set(gt_lanes) | set(new_lanes)):
        if side not in gt_lanes or side not in new_lanes:
            out.append({"side": side, "present": False})
            continue
        g = _polyline_enu(gt_m, gt_lanes[side])
        n = _polyline_enu(new_m, new_lanes[side])
        d_gn = [_point_to_polyline(p, n) for p in g] if len(n) > 1 else []
        d_ng = [_point_to_polyline(p, g) for p in n] if len(g) > 1 else []
        devs = np.asarray(d_gn + d_ng)
        out.append({
            "side": side, "present": True,
            "n_gt": int(len(g)), "n_new": int(len(n)),
            "dev_max": float(devs.max()) if len(devs) else 0.0,
            "dev_rms": float(np.sqrt((devs**2).mean())) if len(devs) else 0.0,
        })
    return out


# -- arrows --------------------------------------------------------------
def _arrow_yaw_enu(m: MapAdas, obj: MapObject) -> float:
    """Heading of the arrow's forward axis in the ENU ground plane (deg)."""
    pose: GaussianPose = obj.geometry["pose"]
    fwd_M = pose.T[:3, 2]  # columns are [right, down, forward]
    d = _direction_to_enu(m.georef, fwd_M)
    return float(np.degrees(np.arctan2(d[0], d[1])))  # atan2(E, N) from north


def _arrow_deltas(gt_m: MapAdas, new_m: MapAdas, r: float) -> dict:
    gt = gt_m.of_class(MapClass.ROAD_ARROW)
    new = new_m.of_class(MapClass.ROAD_ARROW)
    gt_enu = _enu_points(gt_m, gt)
    new_enu = _enu_points(new_m, new)
    mm = _match(gt_enu, new_enu, r)
    yaw_devs = []
    for gi, nj in mm["pairs"]:
        dy = abs(_arrow_yaw_enu(gt_m, gt[gi]) - _arrow_yaw_enu(new_m, new[nj]))
        dy = min(dy, 360.0 - dy)
        yaw_devs.append(dy)
    yaw_a = np.asarray(yaw_devs)
    return {
        **{k: mm[k] for k in mm if k != "pairs"},
        "yaw_max_deg": float(yaw_a.max()) if len(yaw_a) else 0.0,
        "yaw_rms_deg": float(np.sqrt((yaw_a**2).mean())) if len(yaw_a) else 0.0,
    }


# -- top-level comparison ------------------------------------------------
def compare_maps(gt: MapAdas, new: MapAdas,
                 tol: VerifyTolerances | None = None) -> dict:
    """Compare two georeferenced maps in ENU; return a structured report."""
    tol = tol or VerifyTolerances()
    if gt.georef is None or new.georef is None:
        raise RuntimeError("both maps must be georeferenced to compare in ENU")

    classes: dict[str, dict] = {}
    for cls in _POINT_CLASSES:
        g = gt.of_class(cls)
        n = new.of_class(cls)
        classes[cls.value] = _match(
            _enu_points(gt, g), _enu_points(new, n), tol.match_r_m)
        classes[cls.value].pop("pairs", None)

    georef = _georef_delta(gt.georef, new.georef)
    lanes = _lane_deviation(gt, new)
    arrows = _arrow_deltas(gt, new, tol.match_r_m)

    n_gt = sum(c["n_gt"] for c in classes.values()) + arrows["n_gt"]
    n_matched = sum(c["matched"] for c in classes.values()) + arrows["matched"]
    n_spurious = sum(c["spurious"] for c in classes.values()) + arrows["spurious"]
    pos_devs = [c["dev_rms"] for c in classes.values() if c["matched"]]
    pos_devs += [arrows["dev_rms"]] if arrows["matched"] else []
    worst_pos = max([c["dev_max"] for c in classes.values()]
                    + [arrows["dev_max"]], default=0.0)
    lane_worst = max((ln["dev_max"] for ln in lanes if ln.get("present")), default=0.0)

    match_frac = n_matched / max(n_gt, 1)
    spurious_frac = n_spurious / max(n_gt, 1)
    checks = {
        "match_frac": (match_frac, match_frac >= tol.min_match_frac),
        "spurious_frac": (spurious_frac, spurious_frac <= tol.max_spurious_frac),
        "worst_pos_m": (worst_pos, worst_pos <= tol.match_r_m),
        "pos_rms_m": (max(pos_devs, default=0.0),
                      max(pos_devs, default=0.0) <= tol.pos_rms_m),
        "georef_scale_pct": (georef["scale_pct"], georef["scale_pct"] <= tol.georef_scale_pct),
        "georef_t_m": (georef["t_delta_m"], georef["t_delta_m"] <= tol.georef_t_m),
        "lane_lateral_m": (lane_worst, lane_worst <= tol.lane_lateral_m),
        "arrow_yaw_deg": (arrows["yaw_max_deg"], arrows["yaw_max_deg"] <= tol.arrow_yaw_deg),
    }
    verdict = all(ok for _, ok in checks.values())
    return {
        "verdict": "PASS" if verdict else "FAIL",
        "summary": {
            "n_gt": n_gt, "n_matched": n_matched, "n_spurious": n_spurious,
            "match_frac": match_frac, "spurious_frac": spurious_frac,
            "worst_pos_m": worst_pos,
        },
        "classes": classes,
        "georef": georef,
        "lanes": lanes,
        "arrows": arrows,
        "checks": {k: {"value": v, "ok": ok} for k, (v, ok) in checks.items()},
    }


def format_report(rep: dict, tol: VerifyTolerances | None = None) -> str:
    """Human-readable rendering of a :func:`compare_maps` report."""
    tol = tol or VerifyTolerances()
    L = []
    s = rep["summary"]
    L.append(f"ADAS map-diff (GT vs new, in ENU)  →  {rep['verdict']}")
    L.append(f"  objects: {s['n_matched']}/{s['n_gt']} matched "
             f"({s['match_frac']:.0%}), {s['n_spurious']} spurious "
             f"({s['spurious_frac']:.0%}), worst centre dev {s['worst_pos_m']:.3f} m")
    L.append("  per class:")
    for name, c in rep["classes"].items():
        L.append(f"    {name:15s} GT {c['n_gt']:3d} | new {c['n_new']:3d} | "
                 f"matched {c['matched']:3d} miss {c['missing']:2d} spur {c['spurious']:2d}  "
                 f"dev max={c['dev_max']:.3f} rms={c['dev_rms']:.3f} m")
    a = rep["arrows"]
    L.append(f"    {'road_arrow':15s} GT {a['n_gt']:3d} | new {a['n_new']:3d} | "
             f"matched {a['matched']:3d} miss {a['missing']:2d} spur {a['spurious']:2d}  "
             f"dev max={a['dev_max']:.3f} m | yaw max={a['yaw_max_deg']:.2f}°")
    L.append("  lanes:")
    for lane in rep["lanes"]:
        if lane.get("present"):
            L.append(f"    {lane['side']:6s} verts {lane['n_gt']}/{lane['n_new']}  "
                     f"lateral dev max={lane['dev_max']:.3f} rms={lane['dev_rms']:.3f} m")
        else:
            L.append(f"    {lane['side']:6s} MISSING in one map")
    g = rep["georef"]
    L.append(f"  georef: Δangle {g['angle_deg']:.3f}°  Δscale {g['scale_pct']:.3f}%  "
             f"Δt {g['t_delta_m']:.3f} m  Δoffset {g['offset_delta_m']:.3f} m")
    L.append("  checks:")
    for name, c in rep["checks"].items():
        mark = "✓" if c["ok"] else "✗"
        L.append(f"    {mark} {name:18s} = {c['value']:.4f}")
    L.append(f"\n  VERDICT: {rep['verdict']}")
    return "\n".join(L)
