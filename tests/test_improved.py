"""Checks on a synthetic road: grey asphalt with one white and one yellow lane."""

import cv2
import numpy as np

from lanekit.improved import color_mask, enhance, white_mask, yellow_mask
from lanekit.lanes import read_lines_txt, write_lines_txt


def road(brightness: float = 1.0) -> np.ndarray:
    img = np.full((240, 320, 3), 70, np.uint8)  # dark grey asphalt
    cv2.line(img, (80, 239), (140, 100), (240, 240, 240), 6)   # white lane (BGR)
    cv2.line(img, (240, 239), (180, 100), (0, 200, 230), 6)    # yellow lane (BGR)
    return (img * brightness).astype(np.uint8)


def test_white_and_yellow_masks_separate_the_two_lanes():
    img = road()
    white, yellow = white_mask(img), yellow_mask(img)
    assert white[200, 100] == 255 and white[200, 220] == 0
    assert yellow[200, 220] == 255 and yellow[200, 100] == 0
    assert color_mask(img)[50, 160] == 0  # asphalt above the lanes is not paint


def test_enhance_brightens_a_night_frame():
    night = road(brightness=0.3)
    assert enhance(night).mean() > night.mean()


def test_lines_txt_round_trip(tmp_path):
    lanes = [np.array([[10.0, 590.0], [20.0, 580.0]]), np.array([[300.0, 590.0], [290.0, 580.0]])]
    path = tmp_path / "frame.lines.txt"
    write_lines_txt(path, lanes)
    back = read_lines_txt(path)
    assert len(back) == 2 and np.allclose(back[0], lanes[0])
