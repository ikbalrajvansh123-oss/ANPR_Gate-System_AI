import cv2

def preprocess(frame):
    return cv2.resize(frame, (640, 480))