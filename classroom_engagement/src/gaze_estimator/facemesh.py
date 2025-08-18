import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class FaceMeshGaze:
    """
    CPU-only gaze estimator using MediaPipe FaceMesh + solvePnP head pose.
    Outputs yaw/pitch in degrees and a coarse gaze label.
    """

    # MediaPipe FaceMesh landmark indices for 6-point head pose (commonly used set)
    # Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_mesh/face_mesh_landmarks.png
    IDX_NOSE_TIP = 1
    IDX_CHIN = 152
    IDX_LEFT_EYE_OUTER = 33
    IDX_RIGHT_EYE_OUTER = 263
    IDX_MOUTH_LEFT = 61
    IDX_MOUTH_RIGHT = 291

    # A rough 3D face model (in mm) for PnP, centered near the nose
    MODEL_POINTS_3D = np.array([
        [0.0,   0.0,   0.0],    # Nose tip
        [0.0, -63.6, -12.5],    # Chin
        [-43.3, 32.7, -26.0],   # Left eye outer corner
        [ 43.3, 32.7, -26.0],   # Right eye outer corner
        [-28.9,-28.9, -24.1],   # Left mouth corner
        [ 28.9,-28.9, -24.1],   # Right mouth corner
    ], dtype=np.float32)

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,   # better eye/nose precision
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _classify(self, yaw_deg: float, pitch_deg: float) -> str:
        # Thresholds tuned for classroom distance; adjust if needed
        if yaw_deg > 20:   # head turned to their left → gaze right from camera POV
            return "right"
        if yaw_deg < -20:
            return "left"
        if pitch_deg > 20:
            return "down"
        if pitch_deg < -15:
            return "up"
        return "forward"

    def estimate(self, frame_bgr, face_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Returns (yaw_deg, pitch_deg, label). None if not available.
        """
        x, y, w, h = face_bbox
        h_img, w_img = frame_bgr.shape[:2]
        crop = frame_bgr[max(0, y):min(h_img, y+h), max(0, x):min(w_img, x+w)]
        if crop.size == 0:
            return None, None, None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, None, None

        # Use the most confident/first face in the crop
        lms = res.multi_face_landmarks[0].landmark
        idxs = [
            self.IDX_NOSE_TIP, self.IDX_CHIN,
            self.IDX_LEFT_EYE_OUTER, self.IDX_RIGHT_EYE_OUTER,
            self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT
        ]
        pts_2d = []
        for idx in idxs:
            lm = lms[idx]
            pts_2d.append([lm.x * w, lm.y * h])  # local (crop) coords in pixels
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Camera intrinsics approximation
        focal = w_img  # simple guess; okay for coarse gaze states
        cam_matrix = np.array([[focal, 0, w_img / 2.0],
                               [0, focal, h_img / 2.0],
                               [0, 0, 1.0]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Solve PnP (head pose)
        ok, rvec, tvec = cv2.solvePnP(self.MODEL_POINTS_3D, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None

        R, _ = cv2.Rodrigues(rvec)
        # Extract yaw/pitch/roll from rotation matrix (y: left-right, x: up-down)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        yaw = np.degrees(np.arctan2(R[2, 0], sy))     # yaw (left/right)
        pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2]))  # pitch (up/down)
        # roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # unused now

        label = self._classify(yaw, pitch)
        return float(yaw), float(pitch), label

    def draw(self, frame_bgr, face_bbox: Tuple[int, int, int, int], yaw: Optional[float], pitch: Optional[float], label: Optional[str]):
        x, y, w, h = face_bbox
        text = "Gaze: " + (label if label else "unknown")
        y_text = y - 50 if y > 60 else y + h + 35
        cv2.putText(frame_bgr, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 0), 2)
        if yaw is not None and pitch is not None:
            # Draw a small direction arrow from face center
            cx, cy = x + w // 2, y + h // 2
            # Map yaw/pitch to a short arrow (visual intuition only)
            dx = int(np.clip(yaw, -30, 30) * 2)     # scale factor
            dy = int(np.clip(pitch, -30, 30) * 2)
            cv2.arrowedLine(frame_bgr, (cx, cy), (cx + dx, cy + dy), (200, 255, 0), 2, tipLength=0.3)
        return frame_bgr