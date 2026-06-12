"""Load per-video GPS sidecars and interpolate positions to frame timestamps.

Sidecar schema (one JSON file per video):
    {
      "locations": [
        {"timestamp": <unix_ms>, "latitude": <deg>, "longitude": <deg>,
         "course": <deg>, "speed": <m/s>, "accuracy": <m>}, ...
      ],
      "startTime": <unix_ms>,
      "endTime":   <unix_ms>,
      "fileName":  "<video>.mp4"
    }

Coordinates are projected to local ENU metres with the first GPS sample as
origin (equirectangular — accurate to <1 m over ~100 m of driving).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EARTH_R = 6378137.0  # WGS84 equatorial radius, metres


@dataclass(frozen=True)
class GpsTrack:
    """GPS samples + frame-time mapping for one video."""

    start_time_ms: int
    end_time_ms: int
    timestamps_ms: np.ndarray   # (N,) int64
    positions_enu: np.ndarray   # (N, 3) float64, metres (E, N, U), Up=0
    accuracies_m: np.ndarray    # (N,) float64
    origin_lat: float
    origin_lon: float

    def interp_at(self, t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate positions to arbitrary timestamps.

        Returns (positions [M, 3], accuracies [M]). Times outside the GPS
        coverage are clamped to the nearest sample.
        """
        t = np.asarray(t_ms, dtype=np.float64)
        t_clip = np.clip(t, self.timestamps_ms[0], self.timestamps_ms[-1])
        out = np.empty((t.shape[0], 3), dtype=np.float64)
        for axis in range(3):
            out[:, axis] = np.interp(
                t_clip, self.timestamps_ms, self.positions_enu[:, axis]
            )
        acc = np.interp(t_clip, self.timestamps_ms, self.accuracies_m)
        return out, acc


def _project_enu(
    lats: np.ndarray, lons: np.ndarray, origin_lat: float, origin_lon: float
) -> np.ndarray:
    """Equirectangular projection — good to <1 m over ~100 m driving distance."""
    lat_rad = np.radians(lats)
    origin_rad = np.radians(origin_lat)
    east = np.radians(lons - origin_lon) * EARTH_R * np.cos(origin_rad)
    north = np.radians(lats - origin_lat) * EARTH_R
    up = np.zeros_like(east)  # sidecar has no altitude
    return np.stack([east, north, up], axis=1)


def load_track(json_path: Path) -> GpsTrack:
    data = json.loads(Path(json_path).read_text())
    locs = data["locations"]
    if not locs:
        raise ValueError(f"{json_path}: empty 'locations' array")
    locs.sort(key=lambda x: x["timestamp"])

    lats = np.array([l["latitude"] for l in locs], dtype=np.float64)
    lons = np.array([l["longitude"] for l in locs], dtype=np.float64)
    times = np.array([l["timestamp"] for l in locs], dtype=np.int64)
    accs = np.array([l.get("accuracy", 10.0) for l in locs], dtype=np.float64)

    origin_lat, origin_lon = float(lats[0]), float(lons[0])
    positions_enu = _project_enu(lats, lons, origin_lat, origin_lon)

    return GpsTrack(
        start_time_ms=int(data["startTime"]),
        end_time_ms=int(data["endTime"]),
        timestamps_ms=times,
        positions_enu=positions_enu,
        accuracies_m=accs,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def frame_timestamps_ms(
    n_frames: int, fps: float, start_time_ms: int, step: int = 1
) -> np.ndarray:
    """Timestamps (ms) for the frames kept by `iter_video(..., step=step)`."""
    frame_indices = np.arange(0, n_frames, step, dtype=np.int64)
    return start_time_ms + np.round(frame_indices * 1000.0 / fps).astype(np.int64)
