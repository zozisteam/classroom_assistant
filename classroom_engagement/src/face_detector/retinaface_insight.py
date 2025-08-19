# src/face_detector/retinaface_insight.py
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    # InsightFace (RetinaFace-based) high-level API
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:
    FaceAnalysis = None  # gracefully handle missing package

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


class RetinaFaceDetector:
    """
    RetinaFace detector via InsightFace's FaceAnalysis.

    - Fail-soft: if InsightFace/ORT is unavailable or init fails, `enabled=False`
      and `detect(...)` simply returns [].
    - Only loads the 'detection' module to avoid extra ONNX models and reduce crashes.
    - On Apple Silicon, prefers CoreML EP then CPU; elsewhere uses CPU by default.

    Usage:
        fd = RetinaFaceDetector(det_size=640)
        boxes = fd.detect(frame_bgr)                      # full-frame
        boxes = fd.detect(frame_bgr, roi=(x, y, w, h))    # restrict to ROI (track bbox)
    """

    def __init__(
        self,
        det_size: int = 640,
        providers: Optional[list] = None,
        prefer_coreml: bool = True,
    ) -> None:
        self.enabled: bool = False
        self.app = None

        if FaceAnalysis is None:
            print("ℹ️ RetinaFaceDetector disabled: 'insightface' not installed.")
            return

        # Choose ONNX Runtime providers
        if providers is None:
            # Prefer CoreML on Apple Silicon (if available) then CPU; otherwise CPU
            providers = ["CPUExecutionProvider"]
            if prefer_coreml:
                # CoreML EP name as seen in ORT: 'CoreMLExecutionProvider'
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=["detection"],  # only load the detector
            )
            # Prepare detector
            self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
            self.enabled = True
            print(f"✅ RetinaFaceDetector enabled (providers={providers}, det_size={det_size})")
        except Exception as e:
            # Do not crash the app; just disable and print reason
            print(f"⚠️ RetinaFaceDetector init failed: {e}")
            self.app = None
            self.enabled = False

    def _clip_box(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """Clip x1,y1,x2,y2 to image size and convert to (x,y,w,h)."""
        x1c = max(0, min(w, x1))
        y1c = max(0, min(h, y1))
        x2c = max(0, min(w, x2))
        y2c = max(0, min(h, y2))
        ww = max(0, x2c - x1c)
        hh = max(0, y2c - y1c)
        return x1c, y1c, ww, hh

    def detect(self, frame_bgr: np.ndarray, roi: Optional[BBox] = None) -> List[BBox]:
        """
        Detect faces and return a list of (x, y, w, h) in IMAGE coordinates.
        If `roi` is provided, detection is performed on that crop and mapped back.
        """
        if not self.enabled or self.app is None:
            return []

        H, W = frame_bgr.shape[:2]

        # ROI path
        if roi is not None:
            x, y, bw, bh = roi
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + bw), min(H, y + bh)
            if x2 <= x1 or y2 <= y1:
                return []

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return []

            try:
                faces = self.app.get(crop)  # list of objects with .bbox = [x1,y1,x2,y2] in crop coords
            except Exception:
                return []

            out: List[BBox] = []
            for f in faces:
                fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
                # Map back to image coords
                ix1, iy1 = x1 + fx1, y1 + fy1
                ix2, iy2 = x1 + fx2, y1 + fy2
                bx, by, bw2, bh2 = self._clip_box(ix1, iy1, ix2, iy2, W, H)
                if bw2 > 0 and bh2 > 0:
                    out.append((bx, by, bw2, bh2))
            return out

        # Full-frame path
        try:
            faces = self.app.get(frame_bgr)
        except Exception:
            return []

        out: List[BBox] = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            bx, by, bw2, bh2 = self._clip_box(x1, y1, x2, y2, W, H)
            if bw2 > 0 and bh2 > 0:
                out.append((bx, by, bw2, bh2))
        return out