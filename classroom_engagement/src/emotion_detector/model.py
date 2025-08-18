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
