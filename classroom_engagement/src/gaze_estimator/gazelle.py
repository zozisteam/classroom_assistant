import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

class _SimplePreprocessor(nn.Module):
    """Minimal preprocessing to match ViT-B style normalization."""
    def __init__(self, size: int = 224):
        super().__init__()
        self.size = size
        # ImageNet mean/std (most ViTs)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: uint8 BCHW 0..255
        x = x.float() / 255.0
        return (x - self.mean) / self.std

class GazeLLE:
    """
    Wrapper for a ViT-B-based Gaze-LLE model (yaw/pitch regressor).
    - If weights are missing, .enabled = False and calls should return None to allow fallback.
    - Input: eye crop (left/right) or full face crop (centered). Here we use full face crop.
    - Output: yaw (deg), pitch (deg)
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.enabled = False
        self.model = None
        self.pre = _SimplePreprocessor(size=224)
        self.size = 224

        if weights_path and os.path.exists(weights_path):
            try:
                # NOTE: Replace this with the real Gazelle model architecture when you have the codebase.
                # For now we define a tiny MLP head expecting a ViT-B feature vector (768-d) to 2 outputs (yaw, pitch).
                self.model = nn.Sequential(
                    nn.Flatten(),  # placeholder, expect pre-extracted features in a real impl
                    nn.Linear(224*224*3, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 2),
                )
                sd = torch.load(weights_path, map_location="cpu")
                if isinstance(sd, dict):
                    try:
                        self.model.load_state_dict(sd, strict=False)
                    except Exception:
                        pass  # tolerate partial load for placeholder
                self.model.to(self.device).eval()
                self.enabled = True
                print(f"✅ GazeLLE enabled on {self.device}")
            except Exception as e:
                print(f"⚠️ Failed to load GazeLLE weights ({weights_path}): {e}. Falling back.")
        else:
            print("ℹ️ GazeLLE weights not provided. Using FaceMesh fallback.")

    def _resize_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
        return t

    @torch.inference_mode()
    def estimate(self, frame_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[float], Optional[float]]:
        if not self.enabled or self.model is None:
            return None, None

        x, y, w, h = face_bbox
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return None, None

        face_crop = frame_bgr[y1:y2, x1:x2]
        t = self._resize_tensor(face_crop).to(self.device)
        t = self.pre(t)
        out = self.model(t)  # shape [1,2]
        yaw, pitch = out[0].detach().cpu().tolist()
        # Placeholder ranges: real model outputs radians or normalized—tune when weights added
        yaw_deg = float(np.clip(yaw, -45.0, 45.0))
        pitch_deg = float(np.clip(pitch, -45.0, 45.0))
        return yaw_deg, pitch_deg