"""Measure CULane's camera geometry from its own lane labels.

Prints the median vanishing point (where lanes meet) and where the two lanes beside
the car reach the bottom edge; lanekit/improved.py's ROI and bird's-eye quad are set
from these numbers. Uses `normal` frames sampled with a seed different from the test set.

    uv run python scripts/calibrate_geometry.py
"""

import numpy as np

from lanekit.culane import frames, ground_truth

W, H = 1640, 590
vps, bottoms = [], []
for path in frames("data/culane", "normal", 300, seed=3):
    lanes = [lane for lane in ground_truth(path) if len(lane) > 5]
    fits = [np.polyfit(lane[:, 1], lane[:, 0], 1) for lane in lanes]
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            a, b = fits[i], fits[j]
            if abs(a[0] - b[0]) > 0.3:  # skip near-parallel pairs
                y = (b[1] - a[1]) / (a[0] - b[0])
                vps.append((np.polyval(a, y), y))
    nearest = sorted(fits, key=lambda f: abs(np.polyval(f, H) - W / 2))[:2]
    bottoms += [np.polyval(f, H) for f in nearest]

vp = np.median(vps, axis=0)
print(f"vanishing point: x={vp[0]:.0f} y={vp[1]:.0f}  (fractions {vp[0] / W:.3f}, {vp[1] / H:.3f})")
left = [x for x in bottoms if x < W / 2]
right = [x for x in bottoms if x >= W / 2]
print(f"lanes beside the car at the bottom edge: left x={np.median(left) / W:.2f}, right x={np.median(right) / W:.2f} (fractions)")
