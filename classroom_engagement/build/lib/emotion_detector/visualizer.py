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
