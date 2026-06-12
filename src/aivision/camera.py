"""Camera intrinsics. For real automotive use, calibrate per-camera; this is a stand-in."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics. fx, fy in pixels; (cx, cy) principal point in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def guess_from_hfov(cls, width: int, height: int, hfov_deg: float = 60.0) -> "Intrinsics":
        """Rough intrinsics from horizontal FOV. Use only as a placeholder."""
        fx = 0.5 * width / np.tan(np.radians(hfov_deg) * 0.5)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)
