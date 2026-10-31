"""Qualitative evaluation and comparison on Indian Road scenes.

Executes baseline, improved, and learned UFLD models on Indian driving captures
(expressways, urban city traffic, suburban arterials) without claiming benchmark scores,
producing side-by-side qualitative figures and stage breakdowns.

Usage:
    uv run python scripts/evaluate_indian_roads.py
"""

from pathlib import Path
import cv2
import numpy as np

from lanekit import baseline, improved, learned

SAMPLES_DIR = Path("data/india_samples")
OUT_FIGURES = Path("outputs/figures")
OUT_DOCS = Path("docs/assets")


def label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return out


def process_scene(name: str, img_path: Path, title: str):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipping {img_path}, unable to load.")
        return

    # Resize to 1280x720 standard dashcam if needed
    h, w = img.shape[:2]
    if (w, h) != (1280, 720):
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

    # 1. Baseline
    b_lanes, b_stages = baseline.detect(img)
    b_vis = label_panel(improved.draw_lanes(img, b_lanes), f"Classical Baseline ({len(b_lanes)} lanes)")

    # 2. Improved Classical
    i_lanes, i_stages = improved.detect(img)
    i_vis = label_panel(improved.draw_lanes(img, i_lanes), f"Classical Improved ({len(i_lanes)} lanes)")

    # 3. Learned UFLD
    l_lanes = learned.detect(img) if learned.available() else []
    l_vis = label_panel(improved.draw_lanes(img, l_lanes), f"Learned UFLD CNN ({len(l_lanes)} lanes)")

    # Original
    orig_vis = label_panel(img, f"Input: {title}")

    # Stack 2x2 grid
    top_row = np.hstack([orig_vis, b_vis])
    bot_row = np.hstack([i_vis, l_vis])
    grid = np.vstack([top_row, bot_row])

    # Resize for report
    grid_out = cv2.resize(grid, (1280, 720), interpolation=cv2.INTER_AREA)

    out_file = OUT_FIGURES / f"compare_india_{name}.jpg"
    cv2.imwrite(str(out_file), grid_out)
    cv2.imwrite(str(OUT_DOCS / f"compare_india_{name}.jpg"), grid_out)
    print(f"Generated comparison: {out_file}")

    # Save stage breakdown for improved pipeline on the first scene
    if name == "expressway":
        tiles = []
        for stage_name, stage_img in i_stages.items():
            im = stage_img if stage_img.ndim == 3 else cv2.cvtColor(stage_img, cv2.COLOR_GRAY2BGR)
            tiles.append(label_panel(cv2.resize(im, (640, 360)), stage_name))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        stages_grid = np.vstack(rows)
        cv2.imwrite(str(OUT_FIGURES / "stages_india_expressway.jpg"), stages_grid)
        cv2.imwrite(str(OUT_DOCS / "stages_india_expressway.jpg"), stages_grid)
        print("Generated stages breakdown: stages_india_expressway.jpg")


def main():
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)

    test_cases = [
        ("expressway", SAMPLES_DIR / "frame_expressway.jpg", "Delhi-Meerut Expressway (Multi-lane)"),
        ("urban", SAMPLES_DIR / "frame_bangalore_urban.jpg", "Bangalore Urban Arterial (Mixed Traffic)"),
        ("traffic", SAMPLES_DIR / "frame_bangalore_traffic.jpg", "Bangalore Dense Traffic / Auto-Rickshaw Occlusion"),
        ("suburban", SAMPLES_DIR / "frame_bangalore_suburban.jpg", "Suburban / Partial Lane Markings"),
    ]

    for name, path, title in test_cases:
        if path.exists():
            process_scene(name, path, title)
        else:
            print(f"File not found: {path}")


if __name__ == "__main__":
    main()

