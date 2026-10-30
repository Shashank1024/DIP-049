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

## Own road clips (qualitative only)

Put phone clips in `data/own_clips/`. No labels and camera geometry varies, so these are used for
failure-case analysis only, never for metrics.
