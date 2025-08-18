import cv2
from collections import defaultdict, deque
import time
from .model import EmotionModel
from .config import EmotionDetectorConfig

class EmotionAnalyzer:
    def __init__(self, config: EmotionDetectorConfig):
        self.session_data = {
            "start_time": time.time(),
            "total_detections": 0,
            "focus_stats": defaultdict(int),
        }
        self.model = EmotionModel(config.model_name)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + config.haarcascade_path)

        self.focus_mapping = {
            'HAPPY': 'engaged', 'NEUTRAL': 'focused', 'SURPRISED': 'engaged',
            'CONFUSED': 'struggling', 'SAD': 'disengaged', 'ANGRY': 'frustrated',
            'FEARFUL': 'anxious', 'DISGUSTED': 'disengaged'
        }

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    def map_emotion(self, label: str, confidence: float) -> str:
        focus = self.focus_mapping.get(label.upper(), 'unknown')
        return 'uncertain' if confidence < 0.6 else focus

    def analyze(self, frame):
        results = []
        for i, (x, y, w, h) in enumerate(self.detect_faces(frame)):
            face_img = frame[y:y+h, x:x+w]
            preds = self.model.predict(face_img)
            if preds:
                top = preds[0]
                focus = self.map_emotion(top['label'], top['score'])
                results.append({
                    "id": f"student_{i+1}",
                    "bbox": (x, y, w, h),
                    "emotion": top['label'],
                    "confidence": top['score'],
                    "focus_state": focus,
                    "timestamp": time.time()
                })
                self.session_data["total_detections"] += 1
                self.session_data["focus_stats"][focus] += 1
                
        return results
    
    def get_class_summary(self) -> dict:
        total = self.session_data["total_detections"]
        if total == 0:
            return {"overall": "no_data", "engagement_percentage": 0.0, "distribution": {}}

        distribution = {}
        for state, count in self.session_data["focus_stats"].items():
            distribution[state] = {
                "count": count,
                "percentage": (count / total) * 100,
            }

        engaged = distribution.get("engaged", {}).get("count", 0)
        focused = distribution.get("focused", {}).get("count", 0)
        engagement_pct = ((engaged + focused) / total) * 100

        if engagement_pct > 70:
            overall = "highly_engaged"
        elif engagement_pct > 50:
            overall = "moderately_engaged"
        elif engagement_pct > 30:
            overall = "low_engagement"
        else:
            overall = "very_low_engagement"

        return {
            "overall": overall,
            "engagement_percentage": engagement_pct,
            "distribution": distribution,
            "session_duration": time.time() - self.session_data["start_time"],
        }

