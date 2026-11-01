# DIP-049 — Shashank Patel's technical contribution

## Scope

This document covers the classical baseline, the shared lane evaluator, the learned UFLD
inference integration and ONNX export workflow, the qualitative Indian-roads study, result
reporting, and reproducibility support. The improved classical detector is Aditya Pandey's
implementation in `lanekit/improved.py` and is treated here as a compared system; it was not
modified for this contribution.

The pipeline overview is in [lane_detection_architecture.md](lane_detection_architecture.md).
The copied report artifacts are in [`assets/`](assets/): [F1 chart](assets/f1_by_condition.png),
[full CSV-derived results table](assets/results_table.md), five representative qualitative
comparisons, and stage visualizations. These are copies of generated outputs; the originals remain
under `outputs/figures/`.

![F1 by condition for all three methods](assets/f1_by_condition.png)

Representative comparisons (each panel includes ground truth and the available detector outputs):

| Condition | Comparison | Stage visualization |
|---|---|---|
| Normal | [normal](assets/compare_normal.jpg) | [stages](assets/stages_normal.jpg) |
| Crowded | [crowded](assets/compare_crowded.jpg) | [stages](assets/stages_crowded.jpg) |
| Shadow | [shadow](assets/compare_shadow.jpg) | [stages](assets/stages_shadow.jpg) |
| No-line | [no-line](assets/compare_no_line.jpg) | [stages](assets/stages_no_line.jpg) |
| Night | [night](assets/compare_night.jpg) | [stages](assets/stages_night.jpg) |

The complete per-condition numeric table is available as [results_table.md](assets/results_table.md).

## Evaluation implementation (`lanekit/evaluate.py`)

Each lane is rasterized onto a blank mask matching the frame dimensions as a 30-pixel-wide
polyline. `iou_matrix` computes intersection over union (IoU) for every predicted/ground-truth
lane pair. `score_frame` uses SciPy's Hungarian assignment on the negative IoU matrix to find a
one-to-one assignment maximizing total IoU. An assigned pair counts as a true positive only when
its IoU is strictly greater than 0.5. Unmatched predictions are false positives; unmatched ground
truth lanes are false negatives. Empty predictions or ground truth are handled directly.

`Counts` accumulates TP, FP and FN across frames. Precision, recall and F1 are then calculated
from the accumulated counts, rather than averaging per-frame F1. `ego_lanes` chooses the lane
closest to the horizontal image center on each side, based on each lane's lowest image point;
the evaluation script separately scores those selected ego lanes. This is a project-side
CULane-style implementation. The repository does not invoke the official CULane evaluator, so
the values below should be described as results from this implementation, not as certified
official-evaluator scores.

## Classical baseline (`lanekit/baseline.py`)

For each BGR input, the detector converts to grayscale, applies a 5×5 Gaussian blur, and runs
Canny with thresholds 50 and 150 by default. It restricts edges with the shared road-region mask,
then calls probabilistic Hough with 1-pixel / 1-degree resolution, threshold 30, minimum segment
length 3% of image width, and maximum gap 10% of image width. Vertical segments and near-horizontal
segments (absolute slope below 0.3) are discarded. Remaining segments are grouped by slope sign
and each side is combined with a segment-length-weighted slope and intercept. The two fitted
straight lines are sampled at 30 vertical positions from the bottom of the frame to the shared
top fraction. It returns at most one lane on each side, plus intermediate stage images for
inspection.

## UFLD inference and ONNX conversion

`scripts/export_ufld_onnx.py` reconstructs the upstream Ultra-Fast Lane Detection parsing network
with ResNet-18 and CULane dimensions `(201, 18, 4)`, loads the CULane checkpoint from
`models/culane_18.pth`, removes a leading `module.` from state-dictionary keys, and exports a
fixed input tensor of shape `1×3×288×800` to `models/ufld_culane_18.onnx` (opset 11). The export
script uses PyTorch and torchvision; the inference path does not require PyTorch.

`lanekit/learned.py` includes UFLD automatically when the ONNX file exists. It resizes each image
to 800×288, converts BGR to RGB, scales to [0, 1], applies ImageNet mean/std normalization, and
uses OpenCV's `cv2.dnn` ONNX reader and forward pass. The model output is interpreted as 201
classes (200 horizontal grid positions plus a no-lane class), 18 row anchors and four lane slots.
Softmax-weighted grid positions are decoded, rows whose winning class is the no-lane class are
removed, coordinates are mapped back to source-image dimensions, and slots with at most two
valid rows are omitted. This is pretrained-model inference, not model training by this project.

The ONNX conversion and three-method evaluation were independently reproduced by re-running the
full evaluation from the exported model on a separate machine; all F1, precision, recall and
ego-F1 values matched exactly. The local model binary and upstream UFLD checkout remain local
assets (gitignored) and are not part of the committed repository files, consistent with CULane's
own data not being redistributed.

## Recorded CULane comparison

Each method-condition row has 250 frames. F1, precision, recall and ego-lane F1 are the
`outputs/results.csv` values. FPS is the recorded inference speed on the evaluation machine and
depends on hardware, so it varies between independent reproductions even when the accuracy
metrics match exactly.

| Condition | Method | F1 | Precision | Recall | Ego F1 | FPS |
|---|---|---:|---:|---:|---:|---:|
| Normal | Baseline | 0.2494 | 0.3699 | 0.1881 | 0.2993 | 145.8 |
| Normal | Improved | 0.2613 | 0.4790 | 0.1796 | 0.3560 | 50.3 |
| Normal | UFLD | 0.8809 | 0.8820 | 0.8799 | 0.8546 | 12.5 |
| Crowded | Baseline | 0.0795 | 0.1102 | 0.0621 | 0.0882 | 138.5 |
| Crowded | Improved | 0.1356 | 0.2023 | 0.1020 | 0.1791 | 50.1 |
| Crowded | UFLD | 0.7254 | 0.7324 | 0.7186 | 0.8141 | 11.1 |
| Shadow | Baseline | 0.1257 | 0.1678 | 0.1005 | 0.1333 | 118.8 |
| Shadow | Improved | 0.0546 | 0.0730 | 0.0437 | 0.0708 | 47.1 |
| Shadow | UFLD | 0.6772 | 0.6745 | 0.6799 | 0.6674 | 11.1 |
| No-line | Baseline | 0.0759 | 0.1208 | 0.0554 | 0.0556 | 161.0 |
| No-line | Improved | 0.0193 | 0.0378 | 0.0130 | 0.0152 | 52.4 |
| No-line | UFLD | 0.3618 | 0.3817 | 0.3439 | 0.2839 | 11.0 |
| Night | Baseline | 0.1627 | 0.2434 | 0.1222 | 0.1957 | 255.8 |
| Night | Improved | 0.0617 | 0.0830 | 0.0491 | 0.0756 | 34.0 |
| Night | UFLD | 0.6319 | 0.6543 | 0.6109 | 0.6644 | 11.0 |

Across the five conditions, the unweighted arithmetic means of the reported condition metrics are:

| Method | Mean F1 | Mean precision | Mean recall | Mean ego F1 | Mean FPS |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.1386 | 0.2024 | 0.1057 | 0.1544 | 163.98 |
| Improved | 0.1065 | 0.1750 | 0.0775 | 0.1393 | 46.78 |
| UFLD | 0.6554 | 0.6650 | 0.6466 | 0.6569 | 11.34 |

These means are descriptive averages over five equally sized condition rows, not a new pooled
frame-level calculation. In this recorded experiment, UFLD has higher F1 in all five conditions;
the classical baseline has the highest recorded FPS in all five. The figures illustrate this
tradeoff and the per-condition variation.

## Qualitative study on Indian roads (`scripts/evaluate_indian_roads.py`)

All three detectors are also run on four Indian road photographs: an expressway scene and three
Bengaluru street scenes, each a single frame extracted via `ffmpeg` from publicly available
dashcam-style footage (see [data/README.md](../data/README.md)). `evaluate_indian_roads.py`
resizes each photo to 1280×720, runs all three detectors, and writes a 2×2 comparison grid
(ground truth / baseline / improved / UFLD) plus a stage breakdown for the expressway scene to
`docs/assets/compare_india_*.jpg` and `stages_india_expressway.jpg`.

`scripts/label_india_ground_truth.py` and `scripts/score_india_roads.py` extend this to scoring.
The expressway photo's two dashed lane lines have measured ground-truth coordinates
(`data/india_samples/frame_expressway.lines.txt`); the three Bengaluru photos contain no visible
lane paint and are labelled with zero true lanes. Scoring these with the same evaluator and the
same IoU > 0.5 rule as the CULane results gives:

| Scene | Baseline F1 | Improved F1 | UFLD F1 | Notes |
|---|---:|---:|---:|---|
| Expressway | 0.00 | 0.00 | 0.00 | UFLD's best-matching lane reaches IoU 0.39, short of 0.5 |
| Dense traffic | — | — | — | No visible lane paint; true lane count is 0 |
| Urban/flyover | — | — | — | No visible lane paint; true lane count is 0 |
| Suburban/no paint | — | — | — | No visible lane paint; true lane count is 0 |

Full values: [docs/assets/india_results.csv](assets/india_results.csv). With a single scored
scene and three zero-lane scenes, this is not a benchmark; it is a direct, measured check of
how the three methods behave outside the CULane distribution. The README's Indian-roads section
has the per-scene qualitative breakdown and discussion.

## Reproducibility

### Environment and assets

The README specifies Python 3.12 and `uv`; dependencies are declared in `pyproject.toml` and
`uv.lock`. CULane data and model assets are local and ignored by Git. For the recorded five-way
evaluation, the code expects `data/culane/list/test_split/` and the referenced test images and
matching `.lines.txt` annotations. The loader selects `normal`, `crowd`, `shadow`, `noline`, and
`night`, takes up to 250 frames per condition with fixed sampling seed 0, and filters missing
image paths. Meaningful metric calculation needs the annotation files. UFLD additionally needs
`models/ufld_culane_18.onnx`; without it, the learned detector is omitted from the methods list.

### Setup and commands

```bash
uv sync

# One-time export, after placing the upstream UFLD checkout and CULane checkpoint at the paths
# documented in scripts/export_ufld_onnx.py:
uv run --no-project --with torch --with torchvision --with onnx --with numpy \
  --index https://download.pytorch.org/whl/cpu python scripts/export_ufld_onnx.py

# Three-method evaluation; this is intentionally not run by figure generation:
uv run python scripts/run_eval.py --culane data/culane --per-condition 250

# Reports from the saved results and qualitative samples/stage visualizations:
uv run python scripts/make_figures.py --culane data/culane
```

The evaluation writes `outputs/results.csv`. `make_figures.py` reads that file to create
`outputs/figures/results_table.md` and `f1_by_condition.png`. It also samples up to 60 existing
frames per condition for detector comparison and improved-pipeline stage images. These qualitative
images require the dataset; UFLD panels appear only when the ONNX file is available. The full set
of generated outputs is ignored by the repository's `.gitignore`; a representative copy of each
(tables, the F1 chart, one comparison and one stage image per condition) is committed under
`docs/assets/` for the report and this document.
