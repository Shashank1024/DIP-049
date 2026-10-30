"""The scorer and both classical methods on known inputs."""

import cv2
import numpy as np

from lanekit import baseline, improved
from lanekit.evaluate import Counts, ego_lanes, score_frame

SHAPE = (590, 1640)  # CULane frame size (h, w)


def lane(x_bottom, x_top):
    y = np.linspace(589, 350, 20)
    return np.stack([np.linspace(x_bottom, x_top, 20), y], axis=1)


def test_perfect_prediction_scores_one():
    gt = [lane(300, 700), lane(1300, 900)]
    c = score_frame(gt, gt, SHAPE)
    assert (c.tp, c.fp, c.fn) == (2, 0, 0) and c.f1 == 1.0


def test_shifted_lane_is_a_miss_and_extra_lane_is_false_positive():
    gt = [lane(300, 700)]
    pred = [lane(400, 800), lane(1300, 900)]  # 100 px off: IoU well under 0.5
    c = score_frame(pred, gt, SHAPE)
    assert (c.tp, c.fp, c.fn) == (0, 2, 1)


def test_small_offset_still_matches():
    c = score_frame([lane(305, 705)], [lane(300, 700)], SHAPE)
    assert c.tp == 1


def test_counts_sum_and_f1():
    total = Counts()
    total.add(Counts(tp=3, fp=1, fn=1))
    total.add(Counts(tp=1, fp=1, fn=1))
    assert total.precision == 4 / 6 and total.recall == 4 / 6


def test_ego_lanes_picks_nearest_each_side():
    lanes = [lane(100, 600), lane(500, 750), lane(1100, 880), lane(1500, 1000)]
    ego = ego_lanes(lanes, 1640)
    assert [l[0, 0] for l in ego] == [500, 1100]


# Lanes aimed at CULane's measured vanishing point (743, 302), as a real road would be.
LEFT, RIGHT = (330, 589), (1310, 589)


def toward_vp(bottom, y=360):
    t = (bottom[1] - y) / (bottom[1] - 302)
    return int(bottom[0] + (743 - bottom[0]) * t), y


def synthetic_road():
    img = np.full((590, 1640, 3), 60, np.uint8)
    cv2.line(img, LEFT, toward_vp(LEFT), (235, 235, 235), 14)
    cv2.line(img, RIGHT, toward_vp(RIGHT), (0, 200, 230), 14)
    return img


def test_both_methods_find_the_two_lanes():
    img = synthetic_road()
    gt = [lane(LEFT[0], toward_vp(LEFT, 350)[0]), lane(RIGHT[0], toward_vp(RIGHT, 350)[0])]
    for method in (baseline.detect, improved.detect):
        pred, _ = method(img)
        assert score_frame(pred, gt, SHAPE).tp >= 1, method.__module__
