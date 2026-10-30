"""Score every method on every CULane condition and write outputs/results.csv.

    uv run python scripts/run_eval.py --culane data/culane --per-condition 250
    uv run python scripts/run_eval.py --ablation        # improved method, one stage off at a time
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

from lanekit import baseline, improved
from lanekit.culane import CONDITIONS, frames, ground_truth
from lanekit.evaluate import Counts, ego_lanes, score_frame


def methods(ablation: bool):
    if ablation:
        variants = {
            "improved_full": improved.Options(),
            "no_enhance": improved.Options(enhance=False),
            "no_color": improved.Options(color=False),
            "no_warp": improved.Options(warp=False),
            "plus_closing": improved.Options(close=True),
            "plus_curves": improved.Options(curves=True),
        }
        return {name: (lambda img, o=o: improved.detect(img, o)[0]) for name, o in variants.items()}
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--culane", default="data/culane")
    ap.add_argument("--per-condition", type=int, default=250)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ("outputs/ablation.csv" if args.ablation else "outputs/results.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition in CONDITIONS:
        paths = frames(args.culane, condition, args.per_condition)
        if not paths:
            print(f"skip {condition}: no frames found")
            continue
        for name, run in methods(args.ablation).items():
            all_lanes, ego, seconds = Counts(), Counts(), 0.0
            for path in paths:
                img = cv2.imread(str(path))
                gt = ground_truth(path)
                start = time.perf_counter()
                pred = run(img)
                seconds += time.perf_counter() - start
                shape = img.shape[:2]
                all_lanes.add(score_frame(pred, gt, shape))
                ego.add(score_frame(ego_lanes(pred, shape[1]), ego_lanes(gt, shape[1]), shape))
            rows.append({
                "method": name, "condition": condition, "frames": len(paths),
                "f1": round(all_lanes.f1, 4), "precision": round(all_lanes.precision, 4),
                "recall": round(all_lanes.recall, 4), "ego_f1": round(ego.f1, 4),
                "fps": round(len(paths) / seconds, 1) if seconds else 0.0,
            })
            print(rows[-1])

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
