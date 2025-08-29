import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import cv2
from emotion_detector.config import EmotionDetectorConfig
from emotion_detector.analyzer import EmotionAnalyzer
from emotion_detector.visualizer import draw_detections, draw_engagement_label, pixelate_faces
from posture_estimator.blazepose import BlazePoseEstimator
from posture_estimator.detector import PersonDetector
from fusion.engagement import FusionEngine, iou_xywh
from tracking.iou_tracker import IoUTracker
from gaze_estimator.gazelle import GazeLLE
from face_detector.retinaface_insight import RetinaFaceDetector




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
    gaze_estimator = GazeLLE(model_name="gazelle_dinov2_vitb14_inout")
    face_detector = RetinaFaceDetector(det_size=640)



    print("✅ Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect → Track people
        det_bboxes = person_detector.detect_persons(frame)
        tracks = tracker.update(det_bboxes)

        # For each tracked person, try to detect a face INSIDE their bbox (more accurate)
        face_bboxes = []
        for tr in tracks:
            x, y, w, h = tr["bbox"]
            faces_in_roi = face_detector.detect(frame, roi=(x, y, w, h))
            # Pick the largest face in this ROI (if multiple)
            if faces_in_roi:
                fx, fy, fw, fh = max(faces_in_roi, key=lambda b: b[2] * b[3])
                face_bboxes.append((fx, fy, fw, fh))

        # Run emotion on the collected face bboxes (fallback to Haar if none found)
        if face_bboxes:
            results = analyzer.analyze_with_faces(frame, face_bboxes)
        else:
            results = analyzer.analyze(frame)  # fallback

        summary = analyzer.get_class_summary()
        annotated = draw_detections(frame, results, summary)
        # --- Gazelle gaze estimation (single scene encode, multi-face) ---
        gaze_draw = []  # collect to draw after pixelation
        face_boxes = [r["bbox"] for r in results]  # (x,y,w,h) in pixels
        if face_boxes:
            gazes = gaze_estimator.estimate_many(annotated, face_boxes)  # aligned with face_boxes
            # Prepare arrows/labels per face
            for (x, y, w, h), gz in zip(face_boxes, gazes):
                cx, cy = int(x + w / 2), int(y + h / 2)
                tx, ty = gz["target_xy"]
                label = gz["label"]
                inout = gz["inout"]
                gaze_draw.append(((cx, cy), (tx, ty), label, inout))

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
        tracks_with_face = set()
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
                tracks_with_face.add(best["id"])
                engagement = fusion.fuse(
                    student_id=best["id"],
                    focus_state=face_res["focus_state"],
                    focus_conf=face_res["confidence"],
                    posture=best["posture"],
                )
                engagement_to_draw.append((face_bbox, engagement))
        # posture-only fallback when no face matched for a tracked person
        for pr in posture_records:
            if pr["id"] not in tracks_with_face:
                # No face for this person this frame → fuse with uncertain/0.0
                engagement = fusion.fuse(
                    student_id=pr["id"],
                    focus_state="uncertain",
                    focus_conf=0.0,
                    posture=pr.get("posture"),
                )
                # Draw engagement near the PERSON bbox so it's still visible
                engagement_to_draw.append((pr["bbox"], engagement))
       
        # === Censor faces in the DISPLAY ONLY ===
        annotated = pixelate_faces(annotated, results, pixel_size=16)

        # Draw Gazelle gaze: arrow + label
        for (cxy, txy, glabel, inout_prob) in gaze_draw:
            cx, cy = cxy
            tx, ty = txy
            # arrow
            cv2.arrowedLine(annotated, (cx, cy), (tx, ty), (0, 255, 255), 2, tipLength=0.3)
            # label
            text = f"Gaze: {glabel}"
            if inout_prob is not None and inout_prob < 0.5:
                text += " (out)"
            tx_text = max(10, min(annotated.shape[1] - 160, cx + 8))
            ty_text = max(20, cy - 8)
            cv2.putText(annotated, text, (tx_text, ty_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


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
