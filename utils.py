import cv2
import numpy as np


def preprocess(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (640, 480))