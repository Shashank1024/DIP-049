"""Report figures from the scored results and CULane frames.

    uv run python scripts/make_figures.py --culane data/culane

Writes to outputs/figures/:
  results_table.md      F1 / precision / recall / ego F1 / FPS per method and condition
  f1_by_condition.png   grouped bars: F1 per condition for each method
  ablation_table.md     improved-method ablation (if outputs/ablation.csv exists)
  compare_<condition>_<n>.jpg  ground truth + every method side by side, for frames
                        where the methods disagree most (failure-case candidates)
  stages_<condition>.jpg  every stage of the improved method on one frame
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lanekit import baseline, improved  # noqa: E402
from lanekit.culane import CONDITIONS, frames, ground_truth  # noqa: E402
from lanekit.evaluate import score_frame  # noqa: E402

OUT = Path("outputs/figures")


def tables():
    res = Path("outputs/results.csv")
    if res.exists():
        df = pd.read_csv(res)
        (OUT / "results_table.md").write_text(df.to_markdown(index=False))
        pivot = df.pivot(index="condition", columns="method", values="f1").reindex(list(CONDITIONS))
        ax = pivot.plot.bar(figsize=(8, 4), rot=0)
        ax.set_ylabel("F1 (CULane, IoU > 0.5)")
        ax.set_ylim(0, 1)
        ax.set_title("Lane F1 by condition")
        plt.tight_layout()
        plt.savefig(OUT / "f1_by_condition.png", dpi=150)
        plt.close()
    abl = Path("outputs/ablation.csv")
    if abl.exists():
        df = pd.read_csv(abl)
        (OUT / "ablation_table.md").write_text(
            df.pivot(index="method", columns="condition", values="f1").to_markdown())


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return out


def comparisons(culane: str, per_condition: int, sample: int):
    methods = {"baseline": lambda i: baseline.detect(i)[0], "improved": lambda i: improved.detect(i)[0]}
    try:
        from lanekit import learned
        if learned.available():
            methods["learned_ufld"] = learned.detect
    except ImportError:
        pass
    for condition in CONDITIONS:
        paths = frames(culane, condition, sample)
        if not paths:
            continue
        scored = []
        for path in paths:
            img = cv2.imread(str(path))
            gt = ground_truth(path)
            preds = {n: m(img) for n, m in methods.items()}
            f1s = [score_frame(p, gt, img.shape[:2]).f1 for p in preds.values()]
            scored.append((max(f1s) - min(f1s), path, img, gt, preds, f1s))
        scored.sort(key=lambda t: -t[0])
        for n, (_, path, img, gt, preds, f1s) in enumerate(scored[:per_condition]):
            panels = [label(improved.draw_lanes(img, gt, (0, 255, 0)), "ground truth")]
            for (name, pred), f1 in zip(preds.items(), f1s):
                panels.append(label(improved.draw_lanes(img, pred), f"{name}  F1={f1:.2f}"))
            grid = np.vstack(panels)
            cv2.imwrite(str(OUT / f"compare_{condition}_{n}.jpg"), cv2.resize(grid, None, fx=0.5, fy=0.5))
        _, stages = improved.detect(scored[0][2])
        tiles = []
        for name, im in stages.items():
            im = im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            tiles.append(label(cv2.resize(im, (820, 295)), name))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        cv2.imwrite(str(OUT / f"stages_{condition}.jpg"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--culane", default="data/culane")
    ap.add_argument("--per-condition", type=int, default=4, help="comparison figures per condition")
    ap.add_argument("--sample", type=int, default=60, help="frames searched per condition")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tables()
    comparisons(args.culane, args.per_condition, args.sample)
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
