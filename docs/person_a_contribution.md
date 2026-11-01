# DIP-049 — Aditya Pandey's technical contribution

## Scope

This document covers the improved classical detector, `lanekit/improved.py`, and the analysis
built on it: the per-stage ablation and the failure-case comparisons against the baseline and the
learned model. The baseline, the CULane evaluator, and the UFLD integration are Shashank Patel's work
and are treated here as compared systems.

The pipeline overview is in [lane_detection_architecture.md](lane_detection_architecture.md).
This document covers what the "Improved" box in that diagram actually does, what the ablation
shows about each of its stages, and where it fails against real footage.

## How the improved pipeline works (`lanekit/improved.py`)

Each BGR frame goes through nine stages, every one of which is saved for inspection
(`scripts/show_stages.py`, and [stages_normal.jpg](assets/stages_normal.jpg) /
[stages_night.jpg](assets/stages_night.jpg) / [stages_shadow.jpg](assets/stages_shadow.jpg) /
[stages_no_line.jpg](assets/stages_no_line.jpg) / [stages_crowded.jpg](assets/stages_crowded.jpg)):

1. **Enhancement.** CLAHE on the LAB L channel (clip limit 2.0, 8×8 tiles) evens out local
   contrast on every frame. Frames whose mean lightness falls below a threshold additionally get
   a gamma lookup table (γ = 0.6) applied first, to pull detail out of underexposed night shots
   before CLAHE runs.
2. **Colour masks.** A morphological top-hat (the image minus its opening with a wide flat
   kernel) on the HLS lightness channel picks out any stripe that is clearly brighter than the
   road surface immediately beside it — this is what finds white paint, including faded paint,
   without a fixed brightness threshold that would fail differently in sun and shade. A separate
   HLS hue/saturation range (hue 15–35 of 180, moderately saturated) finds yellow paint. The two
   masks are combined with OR.
3. **Road-area mask.** The combined mask is cut to a trapezoid in front of the car, so guardrails,
   kerbs and signage near the image edges cannot be picked up by the window search later.
4. **Bird's-eye warp.** A perspective transform maps the road onto a top-down view. The source
   quad was not guessed — it was measured from CULane's own lane labels
   (`scripts/calibrate_geometry.py`): the vanishing point sits at 45% across and 51% down the
   frame, and the two lanes beside the car reach the bottom edge at about 23% and 74% across.
5. **Sliding window.** Starting from the two strongest columns of paint in two fixed left/right
   search bands, nine stacked windows step upward, re-centring on the mean of whatever paint
   falls inside them. A side is only accepted if paint was found in at least 3 of the 9 windows,
   which keeps a single stray highlight from being read as a lane.
6. **Fit and extend.** A straight line is fit through each side's pixels by default; a 2nd-degree
   curve is only used when the paint found spans at least 85% of the search height (CULane's
   lanes are mostly straight over the scored range, so this avoids fitting curvature to noise).
   The fitted line is extended in its own direction up to the reporting horizon, since the
   bird's-eye search can lose the lane before reaching it.
7. **Warp back and draw.** The curve is mapped back into the camera view and drawn on the frame.

## What each stage is worth (ablation)

Each stage was switched off (or, for two stages that are off by default, switched on) one at a
time and re-scored on the same 250 frames per condition used for the main results. All numbers
are F1 (CULane protocol).

| Version | Normal | Crowded | Shadow | No line | Night |
| --- | --- | --- | --- | --- | --- |
| **Full pipeline** | 0.261 | 0.136 | 0.055 | 0.019 | 0.062 |
| Without enhancement | 0.179 | 0.059 | 0.020 | 0.006 | 0.056 |
| Without colour masks (gradient instead) | 0.224 | 0.114 | 0.054 | 0.018 | 0.097 |
| Without bird's-eye warp | 0.054 | 0.023 | 0.009 | 0.010 | 0.005 |
| With closing added | 0.241 | 0.126 | 0.060 | 0.026 | 0.057 |
| With curve fitting added | 0.231 | 0.122 | 0.043 | 0.021 | 0.053 |

(Full table, all metrics, is [ablation_table.md](assets/ablation_table.md).)

**The bird's-eye warp matters most by a wide margin.** Removing it drops normal-road F1 from
0.261 to 0.054 — an 80% fall. Searching for lanes directly in the camera view means the two lanes
converge toward the vanishing point and overlap in the same columns near the top of the frame, so
the fixed left/right search bands and the straight-line fit both become unreliable. The top-down
view is what makes a simple column-wise search work at all.

**Enhancement is the second most important stage**, worth about a third of normal-road F1 (0.261
→ 0.179 without it) and more than half of crowded-road F1 (0.136 → 0.059). CLAHE recovers contrast
in the darker lower half of frames where the nearest, most reliable paint sits.

**Colour masks help on clean roads but hurt at night.** Removing them (falling back to a Sobel
gradient mask) costs some F1 on normal/crowded roads (0.261→0.224, 0.136→0.114) but *improves*
night F1 (0.062→0.097). At night, headlight glare and wet-road reflections are brighter than the
actual lane paint, so the top-hat finds the glare instead; a plain edge mask is less fooled by
smooth bright blobs, since glare has fewer sharp edges than painted lines do.

**Closing and curve fitting were tested and left off by default.** Both slightly reduce F1 on
every condition except no-line, where closing helps marginally (0.019→0.026) by bridging the rare
fragment of paint that does exist. CULane's scored lane segments are short and close to straight,
so the gap-filling and curvature these stages add mostly introduces drift rather than fixing
anything. They are kept in the code (`Options(close=True)` / `Options(curves=True)`) as tested,
available alternatives, not removed.

## Failure analysis

Representative comparisons (ground truth, baseline, improved, UFLD side by side) for all five
conditions are in [`assets/compare_*.jpg`](assets/). Three specific failures of the improved
method, visible in the committed images:

**Night — [compare_night.jpg](assets/compare_night.jpg) / [stages_night.jpg](assets/stages_night.jpg).**
Both classical methods score F1 = 0.00 on this frame; UFLD scores 1.00. The stage images show why:
the white-paint mask (`2_white`) lights up a large diffuse blob in the upper-right quadrant that
is not paint at all — it is headlight glare and a wet-road reflection from the oncoming and
parked cars. That blob survives the road-area mask and the bird's-eye warp, and the right-side
sliding window locks onto it instead of the real lane, producing a line that runs through empty
road at a steep, wrong angle. The left window, working on much fainter real paint, stays closer
to correct but still misses the IoU-0.5 bar. This is the clearest demonstration in the whole
project of why a fixed brightness-based paint mask is the wrong tool after dark: it cannot tell
"painted" from "merely bright."

**Shadow — [compare_shadow.jpg](assets/compare_shadow.jpg).** At a tunnel exit with a hard
light/dark boundary and the lanes fanning apart, the improved method draws only one line, on the
left, and misses the fork entirely (F1 = 0.00); the baseline's straight Hough lines happen to
catch one of the true lanes well enough to score partial credit (F1 = 0.40) on this frame. The
improved method's single fixed bird's-eye quad assumes one lane per side holds roughly straight
through the warp region; at a genuine fork, the road splits inside that region and the sliding
window follows only whichever branch its starting histogram peak happens to land on.

**No markings — [compare_no_line.jpg](assets/compare_no_line.jpg).** With effectively no painted
lane visible between the parked cars on both sides, both classical methods score F1 = 0.00 — there
is no paint for the colour masks to find, so the windows either find nothing or lock onto car
edges and curb lines instead. UFLD scores 1.00 on the same frame. This is the clearest limitation
of a paint-only method: it has no notion of a lane boundary other than paint, while the learned
model was trained on cases where the road edge, parked cars and lane position jointly imply where
the lane is, even without a visible line.

**Pattern across all three:** the improved method's failures are not random — each one traces
directly to the top-hat/colour-mask assumption that lane paint is simply "the brightest thin
stripe near the road." That assumption holds on clean, well-lit, unobstructed roads (where the
method matches or beats the baseline) and breaks in exactly the ways documented above whenever
something else in the scene is brighter than the paint, the geometry stops being a single
straight stripe per side, or there is no paint to find at all.

## Reproducibility

The improved detector and its stage images are generated with:

```bash
uv run python scripts/show_stages.py <folder-of-images> outputs/improved
```

The ablation is produced with:

```bash
uv run python scripts/run_eval.py --culane data/culane --per-condition 250 --ablation
```

which writes `outputs/ablation.csv`; `scripts/make_figures.py` turns it into the markdown table
above. `ROI_FRACTIONS`, `WARP_SRC_FRACTIONS` and `TOP_FRACTION` in `lanekit/improved.py` are the
measured camera geometry described in step 4; `scripts/calibrate_geometry.py` reproduces that
measurement from CULane's labels directly.
