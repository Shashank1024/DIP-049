"""Score the 3 methods on the 4 Indian photos, IF ground truth has been labelled.

Needs data/india_samples/frame_*.lines.txt from scripts/label_india_ground_truth.py.
With n=4 photos this is not a benchmark, but it's a real, non-fabricated number
instead of none at all. Writes outputs/india_results.csv and prints the table.

    uv run python scripts/score_india_roads.py
"""

import csv
from pathlib import Path

import cv2

from lanekit import baseline, improved
from lanekit.evaluate import Counts, score_frame
from lanekit.lanes import read_lines_txt

SAMPLES = Path("data/india_samples")
PHOTOS = [
    ("expressway", "frame_expressway.jpg"),
    ("traffic", "frame_bangalore_traffic.jpg"),
    ("urban", "frame_bangalore_urban.jpg"),
    ("suburban", "frame_bangalore_suburban.jpg"),
]


def methods():
    found = {
        "baseline": lambda img: baseline.detect(img)[0],
        "improved": lambda img: improved.detect(img)[0],
    }
    try:
        from lanekit import learned
        if learned.available():
            found["learned_ufld"] = learned.detect
    except ImportError:
        pass
    return found


def main():
    rows = []
    missing = []
    for scene, filename in PHOTOS:
        img_path = SAMPLES / filename
        gt_path = img_path.with_suffix(".lines.txt")
        if not img_path.exists():
            continue
        if not gt_path.exists():
            missing.append(filename)
            continue
        img = cv2.imread(str(img_path))
        gt = read_lines_txt(gt_path)
        for name, run in methods().items():
            pred = run(img)
            c = score_frame(pred, gt, img.shape[:2])
            rows.append({"scene": scene, "method": name, "gt_lanes": len(gt),
                         "pred_lanes": len(pred), "tp": c.tp, "fp": c.fp, "fn": c.fn,
                         "precision": round(c.precision, 3), "recall": round(c.recall, 3),
                         "f1": round(c.f1, 3)})

    if missing:
        print("No ground truth yet for:", ", ".join(missing))
        print("Run scripts/label_india_ground_truth.py first.")
    if not rows:
        print("Nothing scored — no labelled photos found.")
        return

    out = Path("outputs/india_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(r)
    print(f"\nwrote {out}  (n={len(set(r['scene'] for r in rows))} photos — "
          f"too few for a benchmark claim, but a real, non-fabricated score)")


if __name__ == "__main__":
    main()
