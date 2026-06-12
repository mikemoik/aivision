"""Feature detection and matching. ORB by default — fast and license-friendly."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameFeatures:
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray  # uint8 for ORB

    @property
    def xy(self) -> np.ndarray:
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float64)


def detect_orb(image_bgr: np.ndarray, n_features: int = 4000) -> FrameFeatures:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=7)
    kps, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        desc = np.empty((0, 32), dtype=np.uint8)
    return FrameFeatures(keypoints=tuple(kps), descriptors=desc)


def match_mutual(a: FrameFeatures, b: FrameFeatures, ratio: float = 0.75) -> np.ndarray:
    """Mutual-nearest-neighbour matching with Lowe ratio test.

    Returns an (N, 2) int array of (idx_in_a, idx_in_b).
    """
    if len(a.descriptors) == 0 or len(b.descriptors) == 0:
        return np.empty((0, 2), dtype=np.int64)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(a.descriptors, b.descriptors, k=2)

    pairs: list[tuple[int, int]] = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            pairs.append((m.queryIdx, m.trainIdx))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)
