"""The lane format shared by every method and the evaluator.

A lane is an (N, 2) float array of (x, y) pixel points ordered from the bottom
of the image upwards. A frame's prediction is a list of lanes. CULane's
`.lines.txt` files store one lane per line as "x1 y1 x2 y2 ...", so the same
format reads and writes those files directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Lane = np.ndarray  # shape (N, 2), columns x, y


def read_lines_txt(path: str | Path) -> list[Lane]:
    lanes = []
    for line in Path(path).read_text().splitlines():
        values = [float(v) for v in line.split()]
        if len(values) >= 4:
            lanes.append(np.array(values, dtype=float).reshape(-1, 2))
    return lanes


def write_lines_txt(path: str | Path, lanes: list[Lane]) -> None:
    rows = [" ".join(f"{x:.2f} {y:.2f}" for x, y in lane) for lane in lanes if len(lane) >= 2]
    Path(path).write_text("\n".join(rows) + ("\n" if rows else ""))
