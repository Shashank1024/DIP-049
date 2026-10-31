# Lane detection pipeline architecture

```mermaid
flowchart TD
    A[Input BGR frame] --> B{Detector}

    B -->|Baseline| C[Grayscale]
    C --> C1[Gaussian blur 5 x 5]
    C1 --> C2[Canny edges 50 / 150]
    C2 --> C3[Road ROI mask]
    C3 --> C4[Probabilistic Hough segments]
    C4 --> C5[Reject horizontal segments; group by slope]
    C5 --> C6[Length-weighted line fit, left and right]
    C6 --> O[Lane point arrays]

    B -->|Improved, Person A| D[Enhancement, color masks, perspective warp and lane fitting]
    D --> O

    B -->|Learned UFLD| E[Resize 800 x 288; BGR to RGB; normalize]
    E --> E1[ONNX model via OpenCV DNN]
    E1 --> E2[Decode 201 x 18 x 4 row-anchor output]
    E2 --> O

    O --> F[CULane .lines.txt ground truth]
    F --> G[Draw each lane as a 30 px mask]
    O --> G
    G --> H[Pair predicted and true lanes by maximum-IoU Hungarian assignment]
    H --> I[Count TP / FP / FN; IoU threshold > 0.5]
    I --> J[Precision / recall / F1; ego-lane F1; FPS]
    J --> K[outputs/results.csv]
    K --> L[outputs/figures: table and per-condition F1 chart]
    A --> M[Sampled frames for qualitative comparisons and improved stages]
    O --> M
```

The diagram summarizes the paths implemented in `lanekit/baseline.py`, `lanekit/improved.py`,
`lanekit/learned.py`, `lanekit/evaluate.py`, and the evaluation/figure scripts. The report scores
are read from `outputs/results.csv`; generating qualitative comparisons runs the detectors again
on a smaller sample and does not rewrite that CSV.
