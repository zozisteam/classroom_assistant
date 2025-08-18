from pydantic import BaseModel

class EmotionDetectorConfig(BaseModel):
    haarcascade_path: str = "haarcascade_frontalface_default.xml"
    model_name: str = "trpakov/vit-face-expression"
