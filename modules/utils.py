import cv2
import os
import uuid

def preprocess(frame):
    return cv2.resize(frame, (640, 480))

def save_plate(img, plate):
    os.makedirs("plates", exist_ok=True)
    filename = f"{plate}_{uuid.uuid4().hex[:6]}.jpg"
    cv2.imwrite(os.path.join("plates", filename), img)