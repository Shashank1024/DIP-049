"""Way 2: the improved classical lane pipeline (Person A).

Stages: CLAHE/gamma enhancement -> white+yellow HLS masks -> morphological
closing -> road-area mask -> bird's-eye warp -> sliding-window search ->
2nd-degree polynomial fit -> warp back. Every stage can be saved for the report,
and the main ones can be switched off for the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lanekit.lanes import Lane

# Camera geometry as fractions of (width, height), measured from CULane's own lane labels
# (scripts/calibrate_geometry.py): lanes meet at about (0.45, 0.51); the two lanes beside
# the car reach the bottom edge near x = 0.23 and 0.74 (the quad below uses 0.20 / 0.80 so
# both lanes sit inside it); the car bonnet covers y > 0.90.
ROI_FRACTIONS = [(0.0, 0.90), (0.40, 0.55), (0.51, 0.55), (1.0, 0.90)]
WARP_SRC_FRACTIONS = [(0.20, 1.0), (0.397, 0.62), (0.53, 0.62), (0.80, 1.0)]
WARP_DST_FRACTIONS = [(0.30, 1.0), (0.30, 0.0), (0.70, 0.0), (0.70, 1.0)]
TOP_FRACTION = 0.56  # lanes are reported from the bottom of the image up to here


@dataclass
class Options:
    """Switches for the ablation: turning one off removes that stage."""
    enhance: bool = True
    color: bool = True    # off -> Sobel-x gradient threshold instead of colour masks
    close: bool = False   # tuned off: on CULane it slightly lowered F1 (kept for the ablation)
    warp: bool = True     # off -> fit lanes in the normal camera view
    curves: bool = False  # tuned off: CULane lanes are mostly straight; curves added error


def enhance(bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8,
            dark_threshold: float = 80.0, gamma: float = 0.6) -> np.ndarray:
    """CLAHE on the LAB L channel; gamma-brighten frames whose mean L is below dark_threshold."""
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    if l.mean() < dark_threshold:
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        l = cv2.LUT(l, lut)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def white_mask(bgr: np.ndarray, contrast: int = 70, max_saturation: int = 120) -> np.ndarray:
    """White paint: pixels clearly lighter than the road right beside them.

    A morphological top-hat (image minus its opening with a wide flat kernel)
    keeps thin bright stripes and removes slow changes in brightness, so faded
    paint, shadows and night frames are handled by one threshold.
    """
    hls = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HLS)
    lightness, saturation = hls[:, :, 1], hls[:, :, 2]
    width = max(15, bgr.shape[1] // 25) | 1
    tophat = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1)))
    return np.where((tophat > contrast) & (saturation < max_saturation), 255, 0).astype(np.uint8)


def yellow_mask(bgr: np.ndarray, hue_range: tuple[int, int] = (15, 35),
                min_lightness: int = 80, min_saturation: int = 80) -> np.ndarray:
    """Yellow paint: OpenCV hue 15-35 (of 0-180), reasonably saturated (HLS)."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    return cv2.inRange(hls, (hue_range[0], min_lightness, min_saturation), (hue_range[1], 255, 255))


def color_mask(bgr: np.ndarray) -> np.ndarray:
    """Union of the white and yellow lane-paint masks (0 or 255)."""
    return cv2.bitwise_or(white_mask(bgr), yellow_mask(bgr))


def gradient_mask(bgr: np.ndarray, threshold: int = 40) -> np.ndarray:
    """Ablation stand-in for the colour masks: strong horizontal gradients (Sobel x)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobel = np.uint8(255 * sobel / max(sobel.max(), 1e-6))
    return np.where(sobel > threshold, 255, 0).astype(np.uint8)


def close_gaps(mask: np.ndarray, size: tuple[int, int] = (5, 15)) -> np.ndarray:
    """Morphological closing with a tall kernel joins the dashes of a dashed lane."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _points(fractions, w: int, h: int) -> np.ndarray:
    return np.float32([(fx * w, fy * h) for fx, fy in fractions])


def roi_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the trapezoid of road in front of the car."""
    h, w = mask.shape[:2]
    region = np.zeros_like(mask)
    cv2.fillPoly(region, [_points(ROI_FRACTIONS, w, h).astype(np.int32)], 255)
    return cv2.bitwise_and(mask, region)


def warp_matrices(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    src, dst = _points(WARP_SRC_FRACTIONS, w, h), _points(WARP_DST_FRACTIONS, w, h)
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def sliding_window(binary: np.ndarray, n_windows: int = 9, margin_frac: float = 0.06,
                   min_pixels: int = 30, min_windows: int = 3, y_top_frac: float = 0.0,
                   search=((0.15, 0.45), (0.55, 0.85))
                   ) -> tuple[list[np.ndarray], np.ndarray]:
    """Find the left and right lane pixels by stepping windows up from histogram peaks.

    Each lane's starting peak is looked for only inside its `search` band (fractions of
    the width), where the lanes beside the car are expected; this keeps guardrails and
    kerbs at the image edges from being taken as lanes.
    Returns one (x, y) pixel array per lane found, and a picture of the search.
    """
    h, w = binary.shape
    y_top = int(y_top_frac * h)
    histogram = binary[(h + y_top) // 2:, :].sum(axis=0).astype(float)
    margin = int(margin_frac * w)
    ys, xs = binary.nonzero()
    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    lanes = []
    for lo, hi in search:
        offset = int(lo * w)
        half = histogram[offset:int(hi * w)]
        if half.max() <= 0:
            continue
        x_current = int(np.argmax(half)) + offset
        window_h = (h - y_top) // n_windows
        chosen, hits = [], 0
        for i in range(n_windows):
            y_low, y_high = h - (i + 1) * window_h, h - i * window_h
            x_low, x_high = x_current - margin, x_current + margin
            cv2.rectangle(debug, (x_low, y_low), (x_high, y_high), (0, 255, 0), 2)
            inside = ((ys >= y_low) & (ys < y_high) & (xs >= x_low) & (xs < x_high)).nonzero()[0]
            chosen.append(inside)
            if len(inside) > min_pixels:
                x_current = int(xs[inside].mean())
                hits += 1
        chosen = np.concatenate(chosen)
        if hits >= min_windows:
            lanes.append(np.stack([xs[chosen], ys[chosen]], axis=1))
    return lanes, debug


def fit_lane(pixels: np.ndarray, y_from: float, y_to: float, n: int = 30,
             min_curve_span: float = 0.85, curves: bool = True) -> tuple[np.ndarray, Lane]:
    """Fit x = a*y^2 + b*y + c and sample it from y_from (bottom) up to y_to.

    A curve needs paint over a long stretch to be trusted: when the pixels span less than
    `min_curve_span` of the range, a straight line is fitted instead (a = 0), so the lane
    can be extended safely beyond the paint that was found.
    """
    ys = pixels[:, 1]
    span = (ys.max() - ys.min()) / max(abs(y_from - y_to), 1)
    degree = 2 if curves and span >= min_curve_span else 1
    coeffs = np.polyfit(ys, pixels[:, 0], degree)
    coeffs = np.concatenate([np.zeros(3 - len(coeffs)), coeffs])
    y = np.linspace(y_from, y_to, n)
    return coeffs, np.stack([np.polyval(coeffs, y), y], axis=1)


class LaneTracker:
    """Temporal smoothing for video: exponential moving average of each lane's curve."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.coeffs: dict[int, np.ndarray] = {}

    def update(self, side: int, coeffs: np.ndarray) -> np.ndarray:
        prev = self.coeffs.get(side)
        self.coeffs[side] = coeffs if prev is None else self.alpha * coeffs + (1 - self.alpha) * prev
        return self.coeffs[side]


def detect(bgr: np.ndarray, opts: Options | None = None, tracker: LaneTracker | None = None
           ) -> tuple[list[Lane], dict[str, np.ndarray]]:
    """Run Way 2 on one frame. Returns lanes (image coordinates) and every stage image."""
    opts = opts or Options()
    h, w = bgr.shape[:2]
    stages = {"0_input": bgr}

    img = enhance(bgr) if opts.enhance else bgr
    stages["1_enhanced"] = img

    if opts.color:
        stages["2_white"] = white_mask(img)
        stages["3_yellow"] = yellow_mask(img)
        mask = cv2.bitwise_or(stages["2_white"], stages["3_yellow"])
    else:
        mask = gradient_mask(img)
    stages["4_mask"] = mask

    mask = roi_mask(mask)
    stages["5_roi"] = mask

    y_bottom, y_top = h - 1, TOP_FRACTION * h
    lanes: list[Lane] = []
    if opts.warp:
        to_bird, to_cam = warp_matrices(w, h)
        bird = cv2.warpPerspective(mask, to_bird, (w, h), flags=cv2.INTER_NEAREST)
        stages["6_birdseye"] = bird
        if opts.close:  # lanes are vertical here, so a tall kernel joins dashes without smearing
            bird = close_gaps(bird)
        stages["7_closed"] = bird
        found, stages["8_windows"] = sliding_window(bird)
        for side, pixels in enumerate(found):
            coeffs, sampled = fit_lane(pixels, h - 1, 0, curves=opts.curves)
            if tracker:
                coeffs = tracker.update(side, coeffs)
            y = np.linspace(h - 1, sampled[-1, 1], 40)
            bird_pts = np.stack([np.polyval(coeffs, y), y], axis=1).astype(np.float32)
            cam = cv2.perspectiveTransform(bird_pts[None], to_cam)[0]
            cam = cam[(cam[:, 1] <= y_bottom) & (cam[:, 1] >= y_top)].astype(float)
            if len(cam) >= 2:
                lanes.append(extend_to(cam, y_top))
    else:
        if opts.close:
            mask = close_gaps(mask, (5, 5))
        stages["7_closed"] = mask
        found, stages["8_windows"] = sliding_window(mask, y_top_frac=TOP_FRACTION,
                                                    search=((0.05, 0.45), (0.55, 0.95)))
        for side, pixels in enumerate(found):
            coeffs, lane = fit_lane(pixels, y_bottom, y_top, curves=opts.curves)
            if tracker:
                coeffs = tracker.update(side, coeffs)
                y = lane[:, 1]
                lane = np.stack([np.polyval(coeffs, y), y], axis=1)
            lanes.append(lane)

    lanes = [lane for lane in lanes if np.all(np.isfinite(lane))]
    stages["9_overlay"] = draw_lanes(bgr, lanes)
    return lanes, stages


def extend_to(lane: Lane, y_top: float, n: int = 8) -> Lane:
    """Continue a lane in a straight line from its top end up to y_top.

    The bird's-eye view stops below the horizon band, so the top of each lane is
    extended along the direction of its upper part.
    """
    top = lane[np.argsort(lane[:, 1])[:max(3, len(lane) // 4)]]
    if lane[:, 1].min() <= y_top + 1 or np.ptp(top[:, 1]) < 1:
        return lane
    slope, icpt = np.polyfit(top[:, 1], top[:, 0], 1)
    y = np.linspace(lane[:, 1].min() - 1, y_top, n)
    return np.vstack([lane, np.stack([slope * y + icpt, y], axis=1)])


def draw_lanes(bgr: np.ndarray, lanes: list[Lane], color=(0, 0, 255), thickness: int = 6) -> np.ndarray:
    out = bgr.copy()
    for lane in lanes:
        cv2.polylines(out, [np.round(lane).astype(np.int32)], False, color, thickness)
    return out


def run_stages(bgr: np.ndarray, opts: Options | None = None) -> dict[str, np.ndarray]:
    """All stage images for one frame (for report figures)."""
    return detect(bgr, opts)[1]
