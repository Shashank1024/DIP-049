"""CULane-style lane scoring.

Each lane is drawn as a 30-pixel-wide line on a blank image of the frame's size.
A predicted lane matches a true lane when the IoU of their drawn lines is above
0.5; lanes are paired one-to-one with the Hungarian algorithm, as in the official
CULane evaluator. Precision, recall and F1 are computed from the summed counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from lanekit.lanes import Lane

LANE_WIDTH = 30
IOU_THRESHOLD = 0.5


def lane_mask(lane: Lane, shape: tuple[int, int], width: int = LANE_WIDTH) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.polylines(mask, [np.round(lane).astype(np.int32)], False, 1, width)
    return mask


def iou_matrix(pred: list[Lane], gt: list[Lane], shape: tuple[int, int]) -> np.ndarray:
    pred_masks = [lane_mask(p, shape) for p in pred]
    gt_masks = [lane_mask(g, shape) for g in gt]
    ious = np.zeros((len(pred), len(gt)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            union = np.count_nonzero(pm | gm)
            ious[i, j] = np.count_nonzero(pm & gm) / union if union else 0.0
    return ious


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_frame(pred: list[Lane], gt: list[Lane], shape: tuple[int, int]) -> Counts:
    if not pred or not gt:
        return Counts(tp=0, fp=len(pred), fn=len(gt))
    ious = iou_matrix(pred, gt, shape)
    rows, cols = linear_sum_assignment(-ious)
    tp = int(np.sum(ious[rows, cols] > IOU_THRESHOLD))
    return Counts(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp)


def ego_lanes(lanes: list[Lane], width: int) -> list[Lane]:
    """The lane nearest the image centre on each side, judged at the lowest point of each lane."""
    centre = width / 2
    left, right = [], []
    for lane in lanes:
        x_bottom = lane[np.argmax(lane[:, 1]), 0]
        (left if x_bottom < centre else right).append((abs(x_bottom - centre), lane))
    return [min(side, key=lambda t: t[0])[1] for side in (left, right) if side]
