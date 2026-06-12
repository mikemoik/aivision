"""Render GPS and reconstructed-pose tracks on top of an OpenStreetMap tile.

Self-contained: fetches OSM tiles (slippy-map XYZ) directly and stitches
them into a single basemap image, then overlays points with matplotlib. No
extra deps beyond what's already in pyproject.
"""

from __future__ import annotations

import math
import time
from io import BytesIO
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import requests
from PIL import Image

from .gps import EARTH_R, load_track


_TILE_SIZE = 256
_USER_AGENT = "aivision/0.1 (https://github.com/mikemoik/aivision)"


def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    """Lat/lon → fractional tile XY (slippy-map convention)."""
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _num2deg(x_tile: float, y_tile: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    lon = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / n)))
    return math.degrees(lat_rad), lon


def _pick_zoom(
    lat_span_deg: float, lon_span_deg: float, target_px: int = 1024
) -> int:
    """Pick the largest zoom such that the bbox fits in ~target_px pixels."""
    for z in range(19, 0, -1):
        n = 2 ** z
        # Approximate tile-pixel span of the bbox
        px_per_lon_deg = n * _TILE_SIZE / 360.0
        lon_px = lon_span_deg * px_per_lon_deg
        # latitude tile size depends on lat; use the center for the estimate
        if lon_px <= target_px and (lat_span_deg * px_per_lon_deg) <= target_px:
            return z
    return 1


def _fetch_tile(z: int, x: int, y: int) -> Image.Image:
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGBA")


def _build_basemap(
    bbox_lat: tuple[float, float], bbox_lon: tuple[float, float], zoom: int,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Stitch OSM tiles covering the bbox into one PIL image.

    Returns (image, extent_in_lon_lat) — extent is (lon_min, lon_max,
    lat_min, lat_max) matching the actual covered area (snaps out to whole
    tiles)."""
    x_min_f, y_max_f = _deg2num(bbox_lat[0], bbox_lon[0], zoom)  # south-west
    x_max_f, y_min_f = _deg2num(bbox_lat[1], bbox_lon[1], zoom)  # north-east
    x0, y0 = int(math.floor(x_min_f)), int(math.floor(y_min_f))
    x1, y1 = int(math.floor(x_max_f)), int(math.floor(y_max_f))
    n = 2 ** zoom
    x0, x1 = max(0, x0), min(n - 1, x1)
    y0, y1 = max(0, y0), min(n - 1, y1)

    width = (x1 - x0 + 1) * _TILE_SIZE
    height = (y1 - y0 + 1) * _TILE_SIZE
    out = Image.new("RGBA", (width, height))
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            tile = _fetch_tile(zoom, tx, ty)
            out.paste(tile, ((tx - x0) * _TILE_SIZE, (ty - y0) * _TILE_SIZE))
            time.sleep(0.05)  # be polite to the tile server

    lat_top, lon_left = _num2deg(x0, y0, zoom)
    lat_bot, lon_right = _num2deg(x1 + 1, y1 + 1, zoom)
    extent = (lon_left, lon_right, lat_bot, lat_top)
    return out, extent


def _enu_to_latlon(
    enu: np.ndarray, origin_lat: float, origin_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of aivision.gps._project_enu."""
    east, north = enu[:, 0], enu[:, 1]
    lat = origin_lat + np.degrees(north / EARTH_R)
    lon = origin_lon + np.degrees(east / (EARTH_R * math.cos(math.radians(origin_lat))))
    return lat, lon


def _figure_with_basemap(
    bbox_lat: tuple[float, float], bbox_lon: tuple[float, float],
    title: str, target_px: int = 1024,
):
    lat_pad = 0.05 * (bbox_lat[1] - bbox_lat[0])
    lon_pad = 0.05 * (bbox_lon[1] - bbox_lon[0])
    bbox_lat_p = (bbox_lat[0] - lat_pad, bbox_lat[1] + lat_pad)
    bbox_lon_p = (bbox_lon[0] - lon_pad, bbox_lon[1] + lon_pad)
    zoom = _pick_zoom(bbox_lat_p[1] - bbox_lat_p[0],
                      bbox_lon_p[1] - bbox_lon_p[0], target_px=target_px)
    basemap, extent = _build_basemap(bbox_lat_p, bbox_lon_p, zoom)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    ax.imshow(np.asarray(basemap), extent=extent, origin="upper", aspect="auto")
    ax.set_xlim(bbox_lon_p)
    ax.set_ylim(bbox_lat_p)
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.set_title(title)
    return fig, ax, zoom


def render_gps_map(gps_json: Path, out_png: Path) -> Path:
    """OSM basemap + GPS sample points coloured by time + accuracy circles."""
    track = load_track(gps_json)
    lats = track.origin_lat + np.degrees(track.positions_enu[:, 1] / EARTH_R)
    lons = (
        track.origin_lon
        + np.degrees(track.positions_enu[:, 0] / (EARTH_R * math.cos(math.radians(track.origin_lat))))
    )

    bbox_lat = (float(lats.min()), float(lats.max()))
    bbox_lon = (float(lons.min()), float(lons.max()))
    fig, ax, zoom = _figure_with_basemap(
        bbox_lat, bbox_lon,
        title=f"GPS track — {len(lats)} samples, accuracy ≈ {track.accuracies_m.mean():.1f} m",
    )

    # Accuracy circles (radius in metres → degrees of latitude/longitude).
    lat_center = float(np.mean(bbox_lat))
    deg_per_m_lat = 1.0 / (EARTH_R * math.pi / 180.0)
    deg_per_m_lon = deg_per_m_lat / math.cos(math.radians(lat_center))
    for lat, lon, acc in zip(lats, lons, track.accuracies_m):
        circle = mpatches.Ellipse(
            (lon, lat),
            width=2 * acc * deg_per_m_lon,
            height=2 * acc * deg_per_m_lat,
            facecolor=(0.2, 0.5, 0.9, 0.10),
            edgecolor=(0.2, 0.5, 0.9, 0.35),
            linewidth=0.6,
        )
        ax.add_patch(circle)

    sc = ax.scatter(
        lons, lats, c=np.arange(len(lats)), cmap="viridis",
        s=18, edgecolor="white", linewidth=0.6, zorder=4,
    )
    ax.plot(lons, lats, color="navy", linewidth=0.8, alpha=0.6, zorder=3)
    plt.colorbar(sc, ax=ax, label="GPS sample index", shrink=0.7)

    ax.text(0.01, 0.99, f"OSM zoom {zoom}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[map] wrote {out_png}")
    return out_png


def render_pose_map(
    reconstruction_dir: Path, gps_json: Path, out_png: Path,
) -> Path:
    """OSM basemap + reconstructed camera trajectory (with GPS track overlaid
    faintly for reference)."""
    track = load_track(gps_json)
    rec = pycolmap.Reconstruction(str(reconstruction_dir))

    img_pairs = sorted(rec.images.items(), key=lambda kv: kv[1].name)
    centers_enu = np.stack([
        (-img.cam_from_world().rotation.matrix().T @ img.cam_from_world().translation)
        for _, img in img_pairs
    ], axis=0)
    cam_lat, cam_lon = _enu_to_latlon(centers_enu, track.origin_lat, track.origin_lon)

    # GPS for the reference overlay
    gps_lat = track.origin_lat + np.degrees(track.positions_enu[:, 1] / EARTH_R)
    gps_lon = (
        track.origin_lon
        + np.degrees(track.positions_enu[:, 0] / (EARTH_R * math.cos(math.radians(track.origin_lat))))
    )

    bbox_lat = (
        float(min(cam_lat.min(), gps_lat.min())),
        float(max(cam_lat.max(), gps_lat.max())),
    )
    bbox_lon = (
        float(min(cam_lon.min(), gps_lon.min())),
        float(max(cam_lon.max(), gps_lon.max())),
    )
    fig, ax, zoom = _figure_with_basemap(
        bbox_lat, bbox_lon,
        title=f"Reconstructed camera trajectory — {len(cam_lat)} poses",
    )

    ax.plot(gps_lon, gps_lat, color="gray", linewidth=1.0, alpha=0.7,
            label="GPS samples", zorder=2)
    ax.scatter(gps_lon, gps_lat, c="gray", s=8, alpha=0.5, zorder=3)

    sc = ax.scatter(
        cam_lon, cam_lat, c=np.arange(len(cam_lat)), cmap="plasma",
        s=20, edgecolor="white", linewidth=0.6, zorder=5, label="cameras",
    )
    ax.plot(cam_lon, cam_lat, color="darkred", linewidth=1.0, alpha=0.7, zorder=4)
    plt.colorbar(sc, ax=ax, label="frame index", shrink=0.7)

    # Per-frame GPS↔camera residual stats
    gps_interp_enu, _ = track.interp_at(
        np.linspace(track.timestamps_ms[0], track.timestamps_ms[-1], len(cam_lat)).astype(np.int64)
    )
    err = np.linalg.norm(centers_enu - gps_interp_enu, axis=1)
    summary = (
        f"|cam − GPS|  mean = {err.mean():.2f} m   median = {np.median(err):.2f} m   "
        f"max = {err.max():.2f} m"
    )
    ax.text(0.01, 0.99, f"OSM zoom {zoom}\n{summary}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="black",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    ax.legend(loc="lower right", framealpha=0.9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[map] wrote {out_png}")
    return out_png
