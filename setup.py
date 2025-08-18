import os

BASE_DIR = "classroom_engagement"
SRC_DIR = os.path.join(BASE_DIR, "src", "emotion_detector")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
TESTS_DIR = os.path.join(BASE_DIR, "tests")

FILES = {
    os.path.join(BASE_DIR, "pyproject.toml"): """\
[project]
name = "classroom-engagement"
version = "0.1.0"
description = "Real-time classroom engagement detection system"
authors = [{ name = "Your Name" }]
requires-python = ">=3.9"

dependencies = [
    "torch>=2.1",
    "torchvision",
    "transformers",
    "opencv-python",
    "numpy",
    "pillow",
    "pydantic",
]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
target-version = "py39"
""",

    os.path.join(BASE_DIR, "README.md"): "# Classroom Engagement Detection\n\nModular real-time system for emotion, posture, gaze, and attention inference in classrooms.",

    os.path.join(SRC_DIR, "__init__.py"): "",

    os.path.join(SRC_DIR, "config.py"): """\
from pydantic import BaseModel

class EmotionDetectorConfig(BaseModel):
    haarcascade_path: str = "haarcascade_frontalface_default.xml"
    model_name: str = "trpakov/vit-face-expression"
""",

    os.path.join(SRC_DIR, "model.py"): """\
from transformers import pipeline
from PIL import Image
import cv2

class EmotionModel:
    def __init__(self, model_name: str):
        self.pipe = pipeline("image-classification", model=model_name, device=-1, top_k=None)

    def predict(self, face_img) -> list:
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((224, 224))
        return sorted(self.pipe(pil_img), key=lambda x: x["score"], reverse=True)
""",

    os.path.join(SRC_DIR, "analyzer.py"): """\
import cv2
from collections import defaultdict, deque
import time
from .model import EmotionModel
from .config import EmotionDetectorConfig

class EmotionAnalyzer:
    def __init__(self, config: EmotionDetectorConfig):
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
        return results
""",

    os.path.join(SRC_DIR, "visualizer.py"): """\
import cv2

FOCUS_COLORS = {
    'engaged': (0, 255, 0),
    'focused': (0, 255, 255),
    'struggling': (0, 165, 255),
    'disengaged': (0, 0, 255),
    'frustrated': (0, 0, 139),
    'anxious': (255, 0, 255),
    'uncertain': (128, 128, 128)
}

def draw_detections(frame, results):
    for res in results:
        x, y, w, h = res['bbox']
        color = FOCUS_COLORS.get(res['focus_state'], (255, 255, 255))
        label = f"{res['id']}: {res['emotion']} ({res['confidence']:.2f})"
        focus = f"Focus: {res['focus_state']}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, focus, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame
""",

    os.path.join(SCRIPTS_DIR, "run_emotion_demo.py"): """\
import cv2
from emotion_detector.config import EmotionDetectorConfig
from emotion_detector.analyzer import EmotionAnalyzer
from emotion_detector.visualizer import draw_detections

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    config = EmotionDetectorConfig()
    analyzer = EmotionAnalyzer(config)

    print("✅ Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = analyzer.analyze(frame)
        annotated = draw_detections(frame, results)

        cv2.imshow("Emotion Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""
}

def ensure_dirs():
    for d in [SRC_DIR, SCRIPTS_DIR, TESTS_DIR]:
        os.makedirs(d, exist_ok=True)

def write_files():
    for path, content in FILES.items():
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)
                print(f"✅ Created: {path}")
        else:
            print(f"⚠️ Skipped (already exists): {path}")

if __name__ == "__main__":
    print("🔧 Setting up classroom_engagement project...")
    ensure_dirs()
    write_files()
    print("\n🚀 Done! Next:\n")
    print("1. (Optional) Create venv & install deps:")
    print("   python3 -m venv .venv && source .venv/bin/activate")
    print("   pip install -e classroom_engagement")
    print("2. Run the demo:")
    print("   python classroom_engagement/scripts/run_emotion_demo.py")
