"""Step 1b of the 3D static-environment thread: lift straight-arrow markings onto the
ground plane GIVEN BY THE LANE MODEL (not per-frame stereo), then estimate one 3D
pose + shape per physical arrow and render it.

Why the lane model and not stereo: the stereo-SGBM road plane (lift_arrows.py) is fit
per frame from low-texture asphalt -> noisy and recomputed every frame. The fitted ego-lane
model (lane_polylines_model.json: left/right borders as world-frame [X,Y,Z], time-integrated,
RANSAC-rejected, Whittaker-smoothed) already IS the road surface: the borders sit on it and
carry its height + tilt + grade. So we take the ground plane from the lane model and drop
stereo from this script entirely.

The detection / lift / ego-lane gate / cluster / pose-fit CORE now lives in
``aivision.map.steps_markings.lift_arrows`` (the map-gen T6 ``markings`` step calls
the same fn) -- this script is the thin CLI that drives it and adds the BEV + RGB
renders. Frame conventions match render_lane_reprojection.py / build_lane_map.py
(cam0->world poses, rectified P2 K, render on colour-left image_2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from aivision.kitti import load_sequence
from aivision.map.steps_markings import arrows_to_json, lift_arrows

YELLOW = (0, 255, 255)        # BGR -- the arrows
CORRIDOR = (255, 150, 0)      # BGR -- the ego-lane blue corridor fill
BORDER = {"left": (255, 130, 30), "right": (190, 60, 255)}
ZNEAR = 5.0                   # m, near-plane clip for projection


def read_calib(p: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in p.read_text().strip().splitlines():
        k, *vals = line.replace(":", "").split()
        out[k] = np.array(vals, np.float64).reshape(3, 4)
    return out


def read_poses(p: Path) -> np.ndarray:
    raw = np.loadtxt(p, dtype=np.float64)
    T = np.tile(np.eye(4), (raw.shape[0], 1, 1))
    T[:, :3, :4] = raw.reshape(-1, 3, 4)
    return T


def project_poly(Pw: np.ndarray, Rinv: np.ndarray, C: np.ndarray, fx, fy, cx, cy):
    """World polygon -> int pixel polygon, or None if any vertex is behind the near plane."""
    Pc = (Pw - C) @ Rinv.T
    if np.any(Pc[:, 2] <= ZNEAR):
        return None
    u = fx * Pc[:, 0] / Pc[:, 2] + cx
    v = fy * Pc[:, 1] / Pc[:, 2] + cy
    return np.round(np.stack([u, v], 1)).astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", type=Path, default=Path("data/kitti/04"))
    ap.add_argument("--poses", type=Path,
                    default=Path("outputs/pocs/orb_slam3/CameraTrajectory.txt"))
    ap.add_argument("--lane-model", type=Path,
                    default=Path("outputs/ego_lanes/lane_map/lane_polylines_model.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/static_env/arrows_lanemodel"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--z-max", type=float, default=45.0, help="lift / trust plane to here (m)")
    ap.add_argument("--los", type=float, default=50.0, help="ego-lane corridor line of sight (m)")
    ap.add_argument("--subsample", type=int, default=2, help="keep every Nth arrow-mask pixel")
    ap.add_argument("--lane-margin", type=float, default=0.4, help="ego-lane gate slack (m)")
    ap.add_argument("--cluster-eps", type=float, default=2.0, help="footprint clustering radius (m)")
    ap.add_argument("--min-obs", type=int, default=2, help="min frames per accepted arrow")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    color = load_sequence(args.seq_dir, camera="image_2")
    W, H, N = color.width, color.height, color.n_frames
    end = N if args.end < 0 else min(args.end, N)

    calib = read_calib(args.seq_dir / "calib.txt")
    K = calib["P2"][:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    poses = read_poses(args.poses)
    model = json.loads(args.lane_model.read_text())
    Lb = np.array(model["left"], np.float64)
    Rb = np.array(model["right"], np.float64)

    # ---- core: detect -> lane-model-plane lift -> ego-lane gate -> cluster -> fit
    res = lift_arrows(args.seq_dir, poses, model, conf=args.conf, z_max=args.z_max,
                      subsample=args.subsample, lane_margin=args.lane_margin,
                      cluster_eps=args.cluster_eps, min_obs=args.min_obs,
                      start=args.start, end=end)
    arrows, footprints = res["arrows"], res["footprints"]
    contours_by_frame = res["contours_by_frame"]
    if not arrows:
        print("[arrows] no arrows estimated -- aborting render")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- BEV world scatter + glyphs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 14))
    track = poses[:end, :3, 3]
    ax.plot(track[:, 0], track[:, 2], "0.6", lw=1.0, label="camera track")
    ax.plot(Lb[:, 0], Lb[:, 2], color=np.array(BORDER["left"][::-1]) / 255, lw=1.2)
    ax.plot(Rb[:, 0], Rb[:, 2], color=np.array(BORDER["right"][::-1]) / 255, lw=1.2)
    allf = np.vstack([f["pts"] for f in footprints])
    ax.scatter(allf[:, 0], allf[:, 2], s=1, c="0.75", label="lifted footprints")
    for a in arrows:
        g = a["glyph_world"]
        poly = plt.Polygon(g[:, [0, 2]], closed=True, facecolor="gold",
                           edgecolor="darkgoldenrod", lw=1.3, alpha=0.85)
        ax.add_patch(poly)
    ax.scatter([], [], c="gold", marker="^", label="estimated arrows")
    ax.set_aspect("equal"); ax.set_xlabel("X world (m)"); ax.set_ylabel("Z world (m)")
    ax.set_title(f"Ego-lane straight arrows ({len(arrows)}) lifted via lane-model plane")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.savefig(args.out_dir / "arrows_bev.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- RGB video: blue ego-lane corridor + reprojected yellow arrow glyphs
    writer = cv2.VideoWriter(str(args.out_dir / "arrows_lift.mp4"),
                             cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (W, H))
    for t in tqdm(range(args.start, end), desc="render"):
        im = cv2.imread(str(color.image_paths[t]), cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        R, C = poses[t][:3, :3], poses[t][:3, 3]
        Rinv = R.T
        overlay = im.copy()

        # blue ego-lane corridor out to the line-of-sight horizon
        Pcl = (Lb - C) @ Rinv.T
        win = (Pcl[:, 2] > ZNEAR) & (Pcl[:, 2] < args.los)
        if win.sum() >= 2:
            order = np.argsort(Pcl[win, 2])
            li = np.where(win)[0][order]
            corridor = np.vstack([Lb[li], Rb[li][::-1]])
            poly = project_poly(corridor, Rinv, C, fx, fy, cx, cy)
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(overlay, [poly], CORRIDOR)
        im = cv2.addWeighted(overlay, 0.28, im, 0.72, 0)
        # border polylines (in-window portion) for a crisp corridor edge
        for name, B in (("left", Lb), ("right", Rb)):
            Pc = (B - C) @ Rinv.T
            wb = (Pc[:, 2] > ZNEAR) & (Pc[:, 2] < args.los)
            if wb.sum() >= 2:
                seg = B[np.where(wb)[0][np.argsort(Pc[wb, 2])]]
                poly = project_poly(seg, Rinv, C, fx, fy, cx, cy)
                if poly is not None and len(poly) >= 2:
                    cv2.polylines(im, [poly], False, BORDER[name], 2, cv2.LINE_AA)

        # estimated arrow glyphs (yellow), reprojected -> they stick to the road across frames
        ov2 = im.copy()
        drew = False
        for a in arrows:
            poly = project_poly(a["glyph_world"], Rinv, C, fx, fy, cx, cy)
            if poly is None or len(poly) < 3:
                continue
            cv2.fillPoly(ov2, [poly], YELLOW)
            drew = True
        if drew:
            im = cv2.addWeighted(ov2, 0.45, im, 0.55, 0)
        for a in arrows:
            poly = project_poly(a["glyph_world"], Rinv, C, fx, fy, cx, cy)
            if poly is not None and len(poly) >= 3:
                cv2.polylines(im, [poly], True, YELLOW, 2, cv2.LINE_AA)
        # faint raw detection outline (estimate-vs-detection check)
        for c in contours_by_frame.get(t, []):
            cv2.drawContours(im, [c], -1, (120, 120, 120), 1, cv2.LINE_AA)
        writer.write(im)
    writer.release()

    # ---- persist the estimated static-arrow map
    np.save(args.out_dir / "footprints_world.npy",
            np.array([{"frame": f["frame"], "cls": f["cls"], "pts": f["pts"]}
                      for f in footprints], dtype=object), allow_pickle=True)
    json.dump(arrows_to_json(arrows), open(args.out_dir / "arrows_world.json", "w"), indent=1)
    print(f"[arrows] wrote {args.out_dir}/ : arrows_bev.png, arrows_lift.mp4, "
          f"arrows_world.json, footprints_world.npy")


if __name__ == "__main__":
    main()
