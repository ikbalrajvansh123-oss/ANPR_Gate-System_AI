from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, conf=0.5, imgsz=640, verbose=False)

        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append((x1, y1, x2, y2))

        return boxes