"""Loading the CULane test split by condition.

Expected layout (see data/README.md):
    data/culane/driver_*/<clip>/<frame>.jpg  + matching <frame>.lines.txt
    data/culane/list/test_split/test0_normal.txt ... test8_night.txt
"""

from __future__ import annotations

import random
from pathlib import Path

from lanekit.lanes import Lane, read_lines_txt

# Condition -> keyword in the split list's file name (test0_normal.txt, test1_crowd.txt, ...).
CONDITIONS = {
    "normal": "normal",
    "crowded": "crowd",
    "shadow": "shadow",
    "no_line": "noline",
    "night": "night",
}


def frames(root: str | Path, condition: str, limit: int | None = None, seed: int = 0) -> list[Path]:
    """Image paths for one condition; a fixed random sample of `limit` frames if given."""
    root = Path(root)
    matches = sorted((root / "list" / "test_split").glob(f"test*_{CONDITIONS[condition]}.txt"))
    if not matches:
        return []
    listing = matches[0]
    paths = [root / line.strip().lstrip("/") for line in listing.read_text().splitlines() if line.strip()]
    paths = [p for p in paths if p.exists()]
    if limit and len(paths) > limit:
        paths = random.Random(seed).sample(paths, limit)
    return sorted(paths)


def ground_truth(image_path: Path) -> list[Lane]:
    label = image_path.with_suffix(".lines.txt")
    return read_lines_txt(label) if label.exists() else []
