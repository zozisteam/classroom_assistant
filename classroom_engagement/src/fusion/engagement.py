from collections import defaultdict, deque
import time
from typing import Optional, Tuple, Dict

def iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / float(area_a + area_b - inter)

class FusionEngine:
    """
    Rule-based fusion + 15s rolling majority smoothing per student.
    """
    def __init__(self, window_seconds: float = 15.0):
        self.window_seconds = window_seconds
        self.history: Dict[str, deque] = defaultdict(lambda: deque())  # stores (timestamp, label)

    def _push(self, student_id: str, label: str) -> str:
        now = time.time()
        q = self.history[student_id]
        q.append((now, label))
        # drop old
        cutoff = now - self.window_seconds
        while q and q[0][0] < cutoff:
            q.popleft()
        # majority over window
        counts = defaultdict(int)
        for _, lab in q:
            counts[lab] += 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def fuse_once(self, focus_state: str, focus_conf: float, posture: Optional[str]) -> str:
        """
        Returns one of: attentive, struggling, distracted, off_task, unknown
        """
        fs = (focus_state or "unknown").lower()
        ps = (posture or "unknown").lower()

        # Prioritize clear negative cognitive states
        if fs in {"struggling", "frustrated"}:
            return "struggling"
        if fs in {"anxious"} and focus_conf >= 0.6:
            return "struggling"

        # Turned/away (if you add this later) would map to off_task
        if ps in {"turned_away"}:
            return "off_task"

        # Good posture + good affect → attentive
        if ps in {"upright", "neutral"} and fs in {"engaged", "focused"}:
            return "attentive"

        # Poor posture + poor/uncertain affect → distracted
        if ps in {"slouched", "leaning"} and fs in {"disengaged", "uncertain"}:
            return "distracted"

        # Fallbacks
        if fs in {"engaged", "focused"}:
            return "attentive"
        if fs in {"disengaged"}:
            return "distracted"

        return "unknown"

    def fuse(self, student_id: str, focus_state: str, focus_conf: float, posture: Optional[str]) -> str:
        raw = self.fuse_once(focus_state, focus_conf, posture)
        return self._push(student_id, raw)