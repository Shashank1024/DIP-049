# Contributions

The course requires each team member to state their individual contribution and be able to
explain the full pipeline. Add a line whenever you finish a piece of work or review the other's.

| Date | Member | Work | Reviewed by |
|---|---|---|---|
| 2026-10-30/31 | Aditya Pandey | Improved classical detector (`lanekit/improved.py`): CLAHE/gamma enhancement, top-hat white + HLS yellow colour masks, road-area mask, measured bird's-eye warp, sliding-window search, straight/curve fitting with extension; ablation of each stage; failure-case analysis — see [docs/person_a_contribution.md](docs/person_a_contribution.md) | |
| 2026-10-30/31 | Shashank Patel | CULane loader (`lanekit/culane.py`), baseline detector (`lanekit/baseline.py`), CULane-protocol evaluator (`lanekit/evaluate.py`), UFLD ONNX export and `cv2.dnn` inference (`scripts/export_ufld_onnx.py`, `lanekit/learned.py`), evaluation/figure scripts, pipeline architecture diagram and results documentation — see [docs/person_b_contribution.md](docs/person_b_contribution.md) | |
| 2026-11-01 | — | Full three-method evaluation (baseline, improved, UFLD) run end to end and figures regenerated with all three methods | |
