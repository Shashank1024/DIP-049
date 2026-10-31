"""Interactive ground-truth labelling for the 4 Indian road photos.

Click points along each true lane, bottom to top. Produces the same .lines.txt
format CULane uses, so the existing evaluator (lanekit/evaluate.py) can score
these photos for real instead of only showing qualitative comparisons.

Controls (one window per photo):
    Left click   add a point to the current lane
    n            finish this lane, start a new one
    u            undo the last point
    s            save this photo's lanes and move to the next photo
    q            quit without saving this photo

Run locally (needs a display — this cannot run over SSH/headless):
    uv run python scripts/label_india_ground_truth.py

Writes data/india_samples/frame_<name>.lines.txt next to each photo, in the
same (x y x y ...) format lanekit/lanes.py reads and writes.
"""

from pathlib import Path

import cv2
import numpy as np

from lanekit.lanes import write_lines_txt

SAMPLES = Path("data/india_samples")
PHOTOS = [
    "frame_expressway.jpg",
    "frame_bangalore_traffic.jpg",
    "frame_bangalore_urban.jpg",
    "frame_bangalore_suburban.jpg",
]


def label_one(path: Path) -> list[np.ndarray]:
    img = cv2.imread(str(path))
    lanes: list[list[tuple[int, int]]] = [[]]
    window = f"Label: {path.name}  (n=new lane, u=undo, s=save+next, q=skip)"

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            lanes[-1].append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    while True:
        vis = img.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for i, lane in enumerate(lanes):
            color = colors[i % len(colors)]
            for pt in lane:
                cv2.circle(vis, pt, 5, color, -1)
            if len(lane) >= 2:
                cv2.polylines(vis, [np.array(lane)], False, color, 2)
        cv2.putText(vis, f"lane {len(lanes)}, {len(lanes[-1])} points", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n"):
            if lanes[-1]:
                lanes.append([])
        elif key == ord("u") and lanes[-1]:
            lanes[-1].pop()
        elif key == ord("s"):
            break
        elif key == ord("q"):
            cv2.destroyWindow(window)
            return []

    cv2.destroyWindow(window)
    return [np.array(lane, dtype=float) for lane in lanes if len(lane) >= 2]


def main():
    for name in PHOTOS:
        path = SAMPLES / name
        if not path.exists():
            print(f"skip (not found): {path}")
            continue
        lanes = label_one(path)
        if not lanes:
            print(f"skipped {name}")
            continue
        out = path.with_suffix(".lines.txt")
        write_lines_txt(out, lanes)
        print(f"saved {len(lanes)} lanes -> {out}")


if __name__ == "__main__":
    main()
