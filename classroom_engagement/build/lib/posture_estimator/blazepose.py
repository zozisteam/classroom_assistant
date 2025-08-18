import mediapipe as mp
import cv2

class BlazePoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def estimate(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        return results

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
        return frame
