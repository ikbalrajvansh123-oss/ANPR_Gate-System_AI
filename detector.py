from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        """
        Detect license plates in frame
        conf=0.25 is lowered to catch more plates
        """
        try:
            results = self.model(frame, conf=0.25, imgsz=640, verbose=False)
            boxes = []
            
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
            
            return boxes
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []