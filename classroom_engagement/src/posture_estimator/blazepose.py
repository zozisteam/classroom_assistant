import mediapipe as mp
import cv2
import numpy as np
from collections import defaultdict, deque


class BlazePoseEstimator:
    def __init__(self):
        self.posture_history = defaultdict(lambda: deque(maxlen=10))  # smooth over last 10 frames
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Needed for single frame inference
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def smooth_posture(self, student_id: str, posture_label: str) -> str:
        history = self.posture_history[student_id]
        history.append(posture_label)

        if not history:
            return posture_label

        # Count occurrences in window
        counts = defaultdict(int)
        for label in history:
            counts[label] += 1

        # Return most frequent label
        return max(counts.items(), key=lambda x: x[1])[0]


    def classify_posture(self, landmarks, crop_size) -> str:
        if not landmarks:
            return "unknown"

        # Get landmark positions in pixels (normalized to crop)
        def get_coords(idx):
            lm = landmarks[idx]
            return np.array([lm.x * crop_size[0], lm.y * crop_size[1]])

        try:
            # Core torso landmarks
            left_shoulder = get_coords(11)
            right_shoulder = get_coords(12)
            left_hip = get_coords(23)
            right_hip = get_coords(24)
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid = (left_hip + right_hip) / 2

            # Head forward landmarks
            nose = get_coords(0)
            left_ear = get_coords(7)
            right_ear = get_coords(8)

            # Spine verticality
            spine_vec = shoulder_mid - hip_mid
            vertical_ref = np.array([0, -1])
            cos_angle = np.dot(spine_vec, vertical_ref) / (np.linalg.norm(spine_vec) + 1e-6)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            # Shoulder imbalance
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])

            # Head forward slouch: nose significantly ahead of shoulder mid
            head_vec = nose - shoulder_mid
            head_forward_dist = abs(head_vec[0])  # lateral slouch
            head_down_dist = head_vec[1]         # downward slouch

            # Heuristic rules
            if angle_deg < 15 and shoulder_diff < 30 and head_forward_dist < 30 and head_down_dist < 30:
                return "upright"
            elif angle_deg > 30 or head_forward_dist > 50 or head_down_dist > 40:
                return "slouched"
            elif shoulder_diff > 40:
                return "leaning"
            else:
                return "neutral"


        except Exception:
            return "unknown"

    def estimate_on_crop(self, frame, bbox):
        x, y, w, h = bbox
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None, None

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_crop)

        return results, (x, y)

    def draw_landmarks_on_crop(self, frame, results, offset, crop_size, student_id=None):
        if results and results.pose_landmarks:
            x_off, y_off = offset
            crop_w, crop_h = crop_size

            # Copy landmarks with corrected coordinates
            for lm in results.pose_landmarks.landmark:
                cx = int(lm.x * crop_w) + x_off
                cy = int(lm.y * crop_h) + y_off
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            # Optionally: draw skeleton connections
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = results.pose_landmarks.landmark[start_idx]
                end_lm = results.pose_landmarks.landmark[end_idx]

                x1 = int(start_lm.x * crop_w) + x_off
                y1 = int(start_lm.y * crop_h) + y_off
                x2 = int(end_lm.x * crop_w) + x_off
                y2 = int(end_lm.y * crop_h) + y_off

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            posture_label = self.classify_posture(results.pose_landmarks.landmark, crop_size)
            if student_id:
                posture_label = self.smooth_posture(student_id, posture_label)
            cx = offset[0]
            cy = offset[1] - 10 if offset[1] > 20 else offset[1] + 20
            cv2.putText(frame, f"Posture: {posture_label}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            return frame, posture_label
        return frame, None
        
