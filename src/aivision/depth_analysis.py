"""Per-track depth statistics for a COLMAP reconstruction.

For each 3D point, computes the depth (camera +Z) in every observing camera,
then summarises the track by its median depth and median inverse depth.
The median-inv-depth distribution is what drives Phase 2 BA weighting:
"far" tracks (inv_d → 0) carry almost-pure rotation evidence and should not
be allowed to drag translation residuals during forward-motion BA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pycolmap


def analyze_track_depths(rec: pycolmap.Reconstruction) -> dict:
    """Compute per-track depth stats. Returns a dict with arrays + summary."""
    pose_by_img: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for img_id, img in rec.images.items():
        pose = img.cam_from_world()
        pose_by_img[img_id] = (pose.rotation.matrix(), np.asarray(pose.translation, dtype=np.float64))

    point_ids: list[int] = []
    median_depth: list[float] = []
    median_inv_d: list[float] = []
    track_len: list[int] = []
    min_depth: list[float] = []
    max_depth: list[float] = []

    for p3d_id, p in rec.points3D.items():
        xyz = np.asarray(p.xyz, dtype=np.float64)
        depths: list[float] = []
        for elem in p.track.elements:
            R, t = pose_by_img[elem.image_id]
            z = float((R @ xyz + t)[2])
            if z > 0.0:
                depths.append(z)
        if not depths:
            continue
        d = np.asarray(depths, dtype=np.float64)
        point_ids.append(int(p3d_id))
        median_depth.append(float(np.median(d)))
        median_inv_d.append(float(np.median(1.0 / d)))
        track_len.append(int(len(d)))
        min_depth.append(float(d.min()))
        max_depth.append(float(d.max()))

    med_d = np.asarray(median_depth)
    med_invd = np.asarray(median_inv_d)
    tlen = np.asarray(track_len)

    depth_percentiles = {
        f"p{int(q)}": float(np.percentile(med_d, q))
        for q in (1, 5, 25, 50, 75, 95, 99)
    }

    # Far-track fractions at depth thresholds (metres). Anything beyond ~50 m
    # contributes negligible parallax for typical dashcam baselines (~10 m of
    # forward motion over the clip), so they are essentially rotation-only.
    far_thresholds = [25.0, 50.0, 100.0, 200.0]
    far_fraction = {
        f"depth>={t:.0f}m": float((med_d >= t).mean()) for t in far_thresholds
    }

    return {
        "n_tracks": int(len(med_d)),
        "n_tracks_total": int(rec.num_points3D()),
        "track_length": {
            "mean": float(tlen.mean()),
            "median": float(np.median(tlen)),
            "max": int(tlen.max()),
        },
        "median_depth_percentiles_m": depth_percentiles,
        "median_inv_depth_percentiles": {
            f"p{int(q)}": float(np.percentile(med_invd, q))
            for q in (1, 5, 25, 50, 75, 95, 99)
        },
        "far_track_fraction": far_fraction,
        "arrays": {
            "point_ids": np.asarray(point_ids, dtype=np.int64),
            "median_depth": med_d,
            "median_inv_depth": med_invd,
            "track_length": tlen,
            "min_depth": np.asarray(min_depth),
            "max_depth": np.asarray(max_depth),
        },
    }


def write_depth_report(reconstruction_dir: Path, out_dir: Path) -> dict:
    """Run the analysis and write summary JSON + histogram PNG."""
    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    result = analyze_track_depths(rec)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in result.items() if k != "arrays"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    arrays = result["arrays"]
    np.savez(
        out_dir / "track_depths.npz",
        point_ids=arrays["point_ids"],
        median_depth=arrays["median_depth"],
        median_inv_depth=arrays["median_inv_depth"],
        track_length=arrays["track_length"],
        min_depth=arrays["min_depth"],
        max_depth=arrays["max_depth"],
    )

    # Histogram. Lazy-import matplotlib so headless envs only pay for it here.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    med_d = arrays["median_depth"]
    med_invd = arrays["median_inv_depth"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.clip(med_d, 0, 300), bins=60, color="#3060a0")
    axes[0].set_xlabel("median depth across observing cameras (m, clipped 300)")
    axes[0].set_ylabel("# tracks")
    axes[0].set_title(f"depth distribution (n={len(med_d)} tracks)")
    axes[0].axvline(50, color="red", linestyle="--", linewidth=1, label="50 m (far cutoff)")
    axes[0].legend()
    axes[1].hist(np.clip(med_invd, 0, 0.2), bins=60, color="#a06030")
    axes[1].set_xlabel("median inverse depth (1/m, clipped 0.2 = 5 m near-clip)")
    axes[1].set_ylabel("# tracks")
    axes[1].set_title("inverse-depth distribution")
    axes[1].axvline(1 / 50.0, color="red", linestyle="--", linewidth=1, label="1/50 m")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "depth_histogram.png", dpi=120)
    plt.close(fig)

    print(f"[depth] {summary['n_tracks']} tracks, median depth = "
          f"{summary['median_depth_percentiles_m']['p50']:.1f} m, "
          f"p95 = {summary['median_depth_percentiles_m']['p95']:.1f} m")
    for k, v in summary["far_track_fraction"].items():
        print(f"[depth]   {k}: {v * 100:.1f}%")
    print(f"[depth] wrote {out_dir}/summary.json + depth_histogram.png + track_depths.npz")
    return summary
