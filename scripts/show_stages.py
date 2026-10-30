"""Save every stage of Way 2 for each image in a folder.

    uv run python scripts/show_stages.py data/samples outputs/improved
"""

import sys
from pathlib import Path

import cv2

from lanekit.improved import run_stages

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
images = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
for path in images:
    bgr = cv2.imread(str(path))
    for name, img in run_stages(bgr).items():
        out = dst / name
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / path.name), img)
print(f"Saved stages for {len(images)} images to {dst}")
