"""Way 3: the pretrained Ultra-Fast Lane Detection model (ResNet-18, CULane), run with cv2.dnn.

UFLD treats lane finding as row-wise classification: for each of 18 fixed image
rows ("row anchors") and each of 4 lane slots, it picks one of 200 horizontal
grid cells, or "no lane". This module only runs inference; the ONNX file comes
from scripts/export_ufld_onnx.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lanekit.lanes import Lane

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ufld_culane_18.onnx"
INPUT_W, INPUT_H = 800, 288
GRIDING_NUM = 200
ROW_ANCHORS = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209,
                        219, 228, 238, 248, 258, 267, 277, 287])  # rows in the 288-high input
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

_net = None


def available() -> bool:
    return MODEL_PATH.exists()


def _load():
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    return _net


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def detect(bgr: np.ndarray) -> list[Lane]:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(bgr, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB) / 255.0
    blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    net = _load()
    net.setInput(blob)
    out = net.forward()[0]                    # (201, 18, 4): grid cell (+ "none") x row x lane

    prob = _softmax(out[:-1], axis=0)         # expected grid position, as in the UFLD demo
    loc = (prob * np.arange(1, GRIDING_NUM + 1)[:, None, None]).sum(axis=0)
    loc[out.argmax(axis=0) == GRIDING_NUM] = 0  # rows where "no lane" wins

    cell_w = (INPUT_W - 1) / (GRIDING_NUM - 1)
    lanes = []
    for k in range(loc.shape[1]):
        rows = np.nonzero(loc[:, k] > 0)[0]
        if len(rows) > 2:
            x = loc[rows, k] * cell_w * w / INPUT_W - 1
            y = ROW_ANCHORS[rows] * h / INPUT_H - 1
            lane = np.stack([x, y], axis=1)
            lanes.append(lane[np.argsort(-lane[:, 1])])  # bottom first
    return lanes
