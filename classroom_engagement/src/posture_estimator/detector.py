from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, model_name="yolov8n.pt", conf_thres=0.4):
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres

    def detect_persons(self, frame):
        results = self.model.predict(source=frame, classes=[0], conf=self.conf_thres, verbose=False)
        bboxes = []

        for result in results:
            for box in result.boxes:
                if box.conf >= self.conf_thres:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    bboxes.append((x1, y1, x2 - x1, y2 - y1))  # (x, y, w, h)

        return bboxes
