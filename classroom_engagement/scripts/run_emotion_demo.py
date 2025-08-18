import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import cv2
from emotion_detector.config import EmotionDetectorConfig
from emotion_detector.analyzer import EmotionAnalyzer
from emotion_detector.visualizer import draw_detections, draw_engagement_label, pixelate_faces
from posture_estimator.blazepose import BlazePoseEstimator
from posture_estimator.detector import PersonDetector
from posture_estimator.blazepose import BlazePoseEstimator
from fusion.engagement import FusionEngine, iou_xywh
from tracking.iou_tracker import IoUTracker
from gaze_estimator.facemesh import FaceMeshGaze
#from gaze_estimator.gazelle import GazeLLE



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    config = EmotionDetectorConfig()
    analyzer = EmotionAnalyzer(config)
    person_detector = PersonDetector()
    pose_estimator = BlazePoseEstimator()
    fusion = FusionEngine(window_seconds=15.0)
    tracker = IoUTracker(iou_thresh=0.45, max_misses=15, ema=0.25)
    gaze_estimator = FaceMeshGaze()



    print("✅ Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = analyzer.analyze(frame)
        summary = analyzer.get_class_summary()
        annotated = draw_detections(frame, results, summary)

        # Detect → Track → Pose
        det_bboxes = person_detector.detect_persons(annotated)
        tracks = tracker.update(det_bboxes)  # [{'id': int, 'bbox': (x,y,w,h), ...}, ...]

        posture_records = []
        for tr in tracks:
            x, y, w, h = tr["bbox"]
            student_id = f"student_{tr['id']}"
            pose_result, offset = pose_estimator.estimate_on_crop(annotated, (x, y, w, h))
            annotated, posture_label = pose_estimator.draw_landmarks_on_crop(
                annotated, pose_result, (x, y), (w, h), student_id=student_id
            )
            posture_records.append({"id": student_id, "bbox": (x, y, w, h), "posture": posture_label})

        # Optional: draw track IDs on-screen
        for tr in tracks:
            x, y, w, h = tr["bbox"]
            cv2.putText(annotated, f"ID:{tr['id']}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 255), 2)

        # Associate each face (emotion) with nearest tracked person bbox via IoU, then fuse
        engagement_to_draw = []  # collect labels to draw after pixelation
        for face_res in results:
            face_bbox = face_res["bbox"]  # (x,y,w,h)
            # find best match
            best = None
            best_iou = 0.0
            for pr in posture_records:
                iou = iou_xywh(face_bbox, pr["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best = pr

            if best and best.get("posture") and best_iou >= 0.1:
                engagement = fusion.fuse(
                    student_id=best["id"],
                    focus_state=face_res["focus_state"],
                    focus_conf=face_res["confidence"],
                    posture=best["posture"],
                )
                engagement_to_draw.append((face_bbox, engagement))
            # --- Eye gaze estimation: Gazelle primary → FaceMesh fallback ---
           # yaw, pitch = gaze_gazelle.estimate(annotated, face_bbox)
            #if yaw is None or pitch is None:
             #   yaw, pitch, gaze_label = gaze_facemesh.estimate(annotated, face_bbox)
            #else:
                # Translate yaw/pitch into a coarse label like FaceMesh for UI consistency
             #   if yaw > 20: gaze_label = "right"
              #  elif yaw < -20: gaze_label = "left"
               # elif pitch > 15: gaze_label = "down"
                #elif pitch < -15: gaze_label = "up"
                #else: gaze_label = "forward"
            #annotated = gaze_facemesh.draw(annotated, face_bbox, yaw, pitch, gaze_label)
            # --- Eye gaze estimation: FaceMesh only ---
            yaw, pitch, gaze_label = gaze_estimator.estimate(annotated, face_bbox)
            annotated = gaze_estimator.draw(annotated, face_bbox, yaw, pitch, gaze_label)
        # === Censor faces in the DISPLAY ONLY ===
        annotated = pixelate_faces(annotated, results, pixel_size=16)

        # Draw engagement labels AFTER pixelation so they remain readable
        for bbox, label in engagement_to_draw:
            annotated = draw_engagement_label(annotated, bbox, label)
        cv2.imshow("Emotion Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
