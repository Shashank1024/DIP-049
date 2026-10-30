"""Way 1: the simple classical baseline (gray -> blur -> Canny -> ROI -> Hough)."""

from __future__ import annotations

import cv2
import numpy as np

from lanekit.improved import TOP_FRACTION, draw_lanes, roi_mask
from lanekit.lanes import Lane


def detect(bgr: np.ndarray, canny_low: int = 50, canny_high: int = 150
           ) -> tuple[list[Lane], dict[str, np.ndarray]]:
    """Returns at most one left and one right lane, as straight lines, plus stage images."""
    h, w = bgr.shape[:2]
    stages = {"0_input": bgr}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    stages["1_blur"] = blur
    edges = cv2.Canny(blur, canny_low, canny_high)
    stages["2_canny"] = edges
    edges = roi_mask(edges)
    stages["3_roi"] = edges
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                               minLineLength=int(0.03 * w), maxLineGap=int(0.1 * w))

    hough = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    sides: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}
    for x1, y1, x2, y2 in (segments.reshape(-1, 4) if segments is not None else []):
        cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:  # nearly horizontal: not a lane
            continue
        side = "left" if slope < 0 else "right"
        length = np.hypot(x2 - x1, y2 - y1)
        sides[side].append((slope, y1 - slope * x1, length))
    stages["4_hough"] = hough

    lanes = []
    y_bottom, y_top = h - 1, TOP_FRACTION * h
    for side in ("left", "right"):
        if not sides[side]:
            continue
        slopes, intercepts, lengths = map(np.array, zip(*sides[side]))
        m, b = np.average(slopes, weights=lengths), np.average(intercepts, weights=lengths)
        y = np.linspace(y_bottom, y_top, 30)
        lanes.append(np.stack([(y - b) / m, y], axis=1))
    stages["5_overlay"] = draw_lanes(bgr, lanes, color=(255, 0, 0))
    return lanes, stages
