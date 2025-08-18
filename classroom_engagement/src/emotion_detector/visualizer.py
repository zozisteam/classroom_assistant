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

def pixelate_faces(frame, face_results, pixel_size: int = 16):
    """
    Pixelate (mosaic) each face bbox in-place on the frame.
    Safe for display; call at the very end of the draw pipeline so it doesn't
    interfere with inference steps that use the image content.
    """
    h, w = frame.shape[:2]
    for res in face_results:
        x, y, bw, bh = res["bbox"]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        # Downscale then upscale with nearest neighbor → mosaic
        small_w = max(1, (x2 - x1) // pixel_size)
        small_h = max(1, (y2 - y1) // pixel_size)
        tiny = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = pix
    return frame

def draw_detections(frame, results, summary):
    for res in results:
        x, y, w, h = res['bbox']
        color = FOCUS_COLORS.get(res['focus_state'], (255, 255, 255))
        label = f"{res['id']}: {res['emotion']} ({res['confidence']:.2f})"
        focus = f"Focus: {res['focus_state']}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, focus, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.rectangle(frame, (10, 10), (370, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Engagement: {summary['engagement_percentage']:.1f}%", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Class State: {summary['overall']}", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Students: {len(results)} | Detections: {summary['distribution'].get('engaged', {}).get('count', 0)}", 
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

def draw_engagement_label(frame, bbox, engagement_label: str):
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    text = f"Engagement: {engagement_label}"

    # Color by label
    color_map = {
        "attentive": (0, 220, 0),
        "struggling": (0, 0, 255),
        "distracted": (0, 165, 255),
        "off_task": (0, 140, 255),
        "unknown": (200, 200, 200),
    }
    txt_color = color_map.get(str(engagement_label).lower(), (0, 215, 255))

    # Preferred anchor above box; fallback below if near top
    y_text = y - 10
    if y_text < 25:
        y_text = y + h + 25

    # Compute background box size
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x1 = max(10, min(W - tw - 2 * pad - 10, x))
    y1 = max(10, y_text - th - 2 * pad)
    x2 = x1 + tw + 2 * pad
    y2 = y1 + th + 2 * pad

    # Solid background for readability
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    # Text
    cv2.putText(frame, text, (x1 + pad, y2 - pad), font, scale, txt_color, thickness, lineType=cv2.LINE_AA)
    return frame