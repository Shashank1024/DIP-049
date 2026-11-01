# Data

Nothing in this folder is committed except this file.

## CULane (quantitative evaluation)

Public lane dataset with a test set split by condition (normal, crowded, night, no-line, shadow,
arrow, dazzle light, curve, crossroad). We use only a subset of the test split. Expected layout:

```
data/culane/
  driver_*/...            test images (.jpg) with matching .lines.txt annotations
  list/test_split/        per-condition file lists (test0_normal.txt, test3_night.txt, ...)
```

Note: the crossroad split has no lanes, so F1 is undefined there; it reports false positives only.

## India samples (qualitative study)

Used by `scripts/evaluate_indian_roads.py` to produce the "Qualitative study on Indian roads"
section of the main README. Four single frames extracted (via `ffmpeg`) from publicly available
Indian driving footage. Expected layout:

```
data/india_samples/
  frame_expressway.jpg           Delhi–Meerut Expressway
  frame_bangalore_urban.jpg      Bengaluru urban arterial, under a flyover
  frame_bangalore_traffic.jpg    Bengaluru dense mixed traffic
  frame_bangalore_suburban.jpg   Bengaluru suburban underpass, no visible lane paint
```

Exact source footage is not tracked in this repo. To recreate this folder, extract one
representative frame per scene from any dashcam-style Indian driving clip with
`ffmpeg -i <clip> -vf "select=eq(n\,<frame_number)" -frames:v 1 <output>.jpg` and resize to
1280×720 (the script resizes automatically if the input isn't already that size).

A matching `<name>.lines.txt` beside each photo is its ground truth, in CULane's point format
(`scripts/label_india_ground_truth.py` to create your own, `scripts/score_india_roads.py` to
score against existing ones). The committed set: dashed-line coordinates for
`frame_expressway.jpg`; empty ground-truth files for the three Bengaluru photos, which contain no
visible lane paint.
