"""Fit a coupled lane model to the ego-lane point cloud (constant-ish width).

The previous polylines fit the left and right borders INDEPENDENTLY, so they can
drift apart / together in ways no real lane does. This re-fits them under a lane
model: a shared centerline c(z) plus a half-width hw(z), with the borders
reconstructed as  left = c - hw,  right = c + hw  (lateral offset along the road).

Pipeline (all parametrised by the forward axis Z, the drive being ~straight):
  1. Split the lane-map cloud by colour (ego-left blue / ego-right pink).
  2. Per Z-bin robust lateral X and height Y for each border (median + MAD trim).
  3. RANSAC a low-degree polynomial to each border's X(z) and drop bins whose
     robust estimate disagrees with it -> kills the pose-drift spikes.
  4. centerline xc=(xl+xr)/2, half-width hw=(xr-xl)/2.
  5. Whittaker smoothing (penalised 2nd differences) of xc, hw, yl, yr. The
     penalty weight lambda is the RIGIDITY knob: big lambda on the WIDTH makes it
     change only slowly (start rigid); the centre is allowed to bend more freely.
  6. Reconstruct left = xc - hw, right = xc + hw and write new polylines JSON.

Frame: first-camera / world frame, metric metres (same as lane_polylines.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def robust_bin(z, x, y, edges, minpts, mad_k=3.0):
    """Per-Z-bin robust (median, MAD-trimmed) lateral X and height Y."""
    zc, xb, yb = [], [], []
    for i in range(len(edges) - 1):
        sel = (z >= edges[i]) & (z < edges[i + 1])
        if sel.sum() < minpts:
            continue
        xs, ys = x[sel], y[sel]
        mx = np.median(xs)
        mad = np.median(np.abs(xs - mx)) + 1e-6
        keep = np.abs(xs - mx) < mad_k * 1.4826 * mad
        if keep.sum() < minpts:
            keep = np.ones_like(xs, bool)
        zc.append(0.5 * (edges[i] + edges[i + 1]))
        xb.append(np.median(xs[keep]))
        yb.append(np.median(ys[keep]))
    return np.array(zc), np.array(xb), np.array(yb)


def ransac_poly(z, x, deg, thresh, iters=300, seed=0):
    """RANSAC polynomial fit of x(z); returns inlier mask over the input bins."""
    n = len(z)
    if n <= deg + 1:
        return np.ones(n, bool)
    rng = np.random.default_rng(seed)
    zn = (z - z.mean()) / (z.std() + 1e-9)          # condition the Vandermonde
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, deg + 1, replace=False)
        try:
            c = np.polyfit(zn[idx], x[idx], deg)
        except Exception:
            continue
        inl = np.abs(np.polyval(c, zn) - x) < thresh
        if inl.sum() > best.sum():
            best = inl
    if best.sum() > deg + 1:                         # refit on all inliers, re-threshold
        c = np.polyfit(zn[best], x[best], deg)
        best = np.abs(np.polyval(c, zn) - x) < thresh
    return best


def whittaker(y, w, lam, dz):
    """Penalised-2nd-difference smoother. lam = rigidity (large -> near a straight
    low-curvature line). w=0 marks missing samples (interpolated by the penalty)."""
    n = len(y)
    D = np.diff(np.eye(n), 2, axis=0) / (dz ** 2)    # 2nd-derivative operator (1/m^2)
    W = np.diag(w)
    A = W + lam * (D.T @ D)
    return np.linalg.solve(A, W @ np.nan_to_num(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=Path, default=Path("outputs/ego_lanes/lane_map/lane_map.ply"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_polylines_model.json"))
    ap.add_argument("--dz", type=float, default=2.0, help="Z-bin size (m)")
    ap.add_argument("--minpts", type=int, default=12, help="min points per bin")
    ap.add_argument("--ransac-deg", type=int, default=3)
    ap.add_argument("--ransac-thresh", type=float, default=0.35,
                    help="lateral inlier band for RANSAC border fit (m)")
    ap.add_argument("--lam-center", type=float, default=20.0,
                    help="centerline rigidity (lower = follows road bends)")
    ap.add_argument("--lam-width", type=float, default=400.0,
                    help="WIDTH rigidity: big = near-constant lane width (start rigid)")
    ap.add_argument("--lam-height", type=float, default=80.0)
    args = ap.parse_args()

    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(args.ply))
    P = np.asarray(pc.points)
    C = np.asarray(pc.colors)
    left_m = C[:, 2] > C[:, 0]          # blue = ego-left
    right_m = C[:, 0] > C[:, 2]         # pink = ego-right

    z_all = P[:, 2]
    zmin = max(z_all.min(), P[left_m, 2].min(), P[right_m, 2].min())
    zmax = min(z_all.max(), P[left_m, 2].max(), P[right_m, 2].max())
    edges = np.arange(zmin, zmax + args.dz, args.dz)

    raw = {}
    for name, m in (("left", left_m), ("right", right_m)):
        zc, xb, yb = robust_bin(P[m, 2], P[m, 0], P[m, 1], edges, args.minpts)
        inl = ransac_poly(zc, xb, args.ransac_deg, args.ransac_thresh)
        print(f"[lanemodel] {name}: {len(zc)} bins, {int(inl.sum())} RANSAC inliers")
        raw[name] = dict(z=zc, x=xb, y=yb, inl=inl)

    # common Z grid; weight a bin only where BOTH borders have a RANSAC-inlier sample
    grid = 0.5 * (edges[:-1] + edges[1:])
    def to_grid(d):
        x = np.full(len(grid), np.nan); y = np.full(len(grid), np.nan); w = np.zeros(len(grid))
        gi = np.round((d["z"] - grid[0]) / args.dz).astype(int)
        ok = (gi >= 0) & (gi < len(grid)) & d["inl"]
        x[gi[ok]] = d["x"][ok]; y[gi[ok]] = d["y"][ok]; w[gi[ok]] = 1.0
        return x, y, w
    xl, yl, wl = to_grid(raw["left"]); xr, yr, wr = to_grid(raw["right"])
    wboth = wl * wr

    xc = 0.5 * (np.nan_to_num(xl) + np.nan_to_num(xr))
    hw = 0.5 * (np.nan_to_num(xr) - np.nan_to_num(xl))

    xc_s = whittaker(xc, wboth, args.lam_center, args.dz)
    hw_s = whittaker(hw, wboth, args.lam_width, args.dz)
    yl_s = whittaker(yl, wl, args.lam_height, args.dz)
    yr_s = whittaker(yr, wr, args.lam_height, args.dz)

    hw_s = np.clip(hw_s, 0.3, None)                 # never cross / collapse
    xl_s = xc_s - hw_s
    xr_s = xc_s + hw_s
    full_w = 2.0 * hw_s
    print(f"[lanemodel] lane width: mean {full_w.mean():.2f} m  "
          f"range {full_w.min():.2f}-{full_w.max():.2f}  "
          f"max |dW/dz| {np.abs(np.diff(full_w) / args.dz).max():.4f} m/m")

    keep = wboth > 0                                # only emit the supported span
    z = grid[keep]
    out = {
        "frame": "cam0 (X-right, Y-down, Z-forward), metric meters",
        "method": (f"coupled lane model: RANSAC(deg{args.ransac_deg}) border reject + "
                   f"Whittaker smooth (lam_center={args.lam_center}, lam_width={args.lam_width}); "
                   f"left=center-halfwidth, right=center+halfwidth"),
        "lane_width_m": {"mean": round(float(full_w[keep].mean()), 3),
                         "min": round(float(full_w[keep].min()), 3),
                         "max": round(float(full_w[keep].max()), 3)},
        "left": [[round(float(xl_s[i]), 4), round(float(yl_s[i]), 4), round(float(z[j]), 4)]
                 for j, i in enumerate(np.where(keep)[0])],
        "right": [[round(float(xr_s[i]), 4), round(float(yr_s[i]), 4), round(float(z[j]), 4)]
                  for j, i in enumerate(np.where(keep)[0])],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"[lanemodel] wrote {args.out}  ({len(out['left'])} verts/side)")


if __name__ == "__main__":
    main()
