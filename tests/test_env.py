"""Sanity checks that the environment matches what the project requires."""

import cv2


def test_opencv_5():
    # The OpenCV AI Competition requires OpenCV 5.
    assert cv2.__version__.startswith("5."), cv2.__version__


def test_opencv_dnn_available():
    # The learned model (UFLD, ONNX) runs through cv2.dnn.
    assert hasattr(cv2, "dnn") and hasattr(cv2.dnn, "readNetFromONNX")
