# Classical vs Learned Lane Detection for Indian and Adverse Driving Conditions

DIP course project (HSR-AIML-049), two-person team.

## What it does

Three lane-detection methods run on the same frames and are scored the same way:

| Method | Pipeline |
|---|---|
| Baseline (classical) | Grayscale + Gaussian blur → Canny → ROI mask → probabilistic Hough |
| Improved (classical) | CLAHE (LAB L-channel) → HLS white/yellow masks → morphological closing → bird's-eye warp → sliding window + polyfit → temporal smoothing |
| Learned | Ultra-Fast Lane Detection (pretrained on CULane, inference only) via ONNX + `cv2.dnn` |

Evaluation: CULane lane F1 / precision / recall per condition (normal, shadow, night, no-line,
crowded), all-lanes and ego-lane-only scores, FPS, and a stage-by-stage ablation.

## Layout

```
lanekit/    core pipelines, evaluation, telemetry (the course project)
scripts/    command-line helpers (e.g. show_stages.py)
tests/      pytest
data/       datasets (gitignored; see data/README.md)
outputs/    stage images, overlays, result tables (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

## Running

```bash
# Data: CULane list + one test archive (driver_100_30frame covers all 5 conditions)
#   extract into data/culane/ (see data/README.md)

uv run python scripts/show_stages.py data/samples outputs/improved   # stage images for any folder of photos
uv run python scripts/run_eval.py --per-condition 250                # scores -> outputs/results.csv
uv run python scripts/run_eval.py --per-condition 250 --ablation     # -> outputs/ablation.csv
uv run python scripts/make_figures.py                                # tables, charts, comparisons -> outputs/figures/
```

The learned method (UFLD) needs a one-time export to ONNX: download `culane_18.pth` from the
[UFLD repo](https://github.com/cfzd/Ultra-Fast-Lane-Detection) into `models/`, clone the repo to
`models/ufld_repo`, then run `scripts/export_ufld_onnx.py` (instructions inside). Once
`models/ufld_culane_18.onnx` exists, the scripts include it automatically.

## License

MIT, see [LICENSE](LICENSE).
