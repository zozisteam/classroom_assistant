from typing import List, Tuple, Dict
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = Tuple[int, int, int, int]  # (x,y,w,h)

def iou_xywh(a: BBox, b: BBox) -> float:
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
    return inter / float(aw * ah + bw * bh - inter)

class _Track:
    __slots__ = ("tid", "bbox", "misses", "hits", "last_time")
    def __init__(self, tid: int, bbox: BBox):
        self.tid = tid
        self.bbox = bbox
        self.misses = 0
        self.hits = 1
        self.last_time = time.time()

class IoUTracker:
    """
    Minimal Hungarian-assignment IoU tracker (no Kalman).
    - Global optimal matching per frame via Hungarian
    - New tracks for unmatched detections
    - Remove tracks after N misses
    - Optional EMA smoothing for bbox stability
    """
    def __init__(self, iou_thresh: float = 0.3, max_misses: int = 15, ema: float = 0.2):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.ema = ema
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def _ema(self, old: BBox, new: BBox) -> BBox:
        if self.ema <= 0:
            return new
        ox, oy, ow, oh = old
        nx, ny, nw, nh = new
        a = self.ema
        return (
            int(ox * (1 - a) + nx * a),
            int(oy * (1 - a) + ny * a),
            int(ow * (1 - a) + nw * a),
            int(oh * (1 - a) + nh * a),
        )

    def update(self, detections: List[BBox]) -> List[Dict]:
        track_ids = list(self._tracks.keys())

        # Edge cases
        if len(track_ids) == 0 and len(detections) == 0:
            return []
        if len(track_ids) == 0:
            # Create tracks for all detections
            for det in detections:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = _Track(tid, det)
            return [{"id": tid, "bbox": tr.bbox, "hits": tr.hits, "misses": tr.misses}
                    for tid, tr in sorted(self._tracks.items())]
        if len(detections) == 0:
            # Age all tracks
            to_delete = []
            for ti, tr in self._tracks.items():
                tr.misses += 1
                if tr.misses > self.max_misses:
                    to_delete.append(ti)
            for ti in to_delete:
                del self._tracks[ti]
            return [{"id": tid, "bbox": tr.bbox, "hits": tr.hits, "misses": tr.misses}
                    for tid, tr in sorted(self._tracks.items())]

        # Build cost matrix = 1 - IoU (lower is better)
        T, D = len(track_ids), len(detections)
        cost = np.ones((T, D), dtype=np.float32)
        for ti_idx, tid in enumerate(track_ids):
            tbox = self._tracks[tid].bbox
            for di in range(D):
                iou = iou_xywh(tbox, detections[di])
                cost[ti_idx, di] = 1.0 - iou

        # Hungarian assignment
        rows, cols = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()
        # Accept matches only if IoU >= threshold
        for r, c in zip(rows, cols):
            tid = track_ids[r]
            det = detections[c]
            iou = 1.0 - cost[r, c]
            if iou >= self.iou_thresh:
                tr = self._tracks[tid]
                tr.bbox = self._ema(tr.bbox, det)
                tr.hits += 1
                tr.misses = 0
                tr.last_time = time.time()
                matched_tracks.add(tid)
                matched_dets.add(c)

        # Unmatched detections → new tracks
        for di in range(D):
            if di not in matched_dets:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = _Track(tid, detections[di])

        # Unmatched tracks → age & possibly delete
        to_delete = []
        for tid in track_ids:
            if tid not in matched_tracks:
                tr = self._tracks[tid]
                tr.misses += 1
                if tr.misses > self.max_misses:
                    to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

        # Emit active tracks
        out = []
        for tid, tr in self._tracks.items():
            out.append({"id": tid, "bbox": tr.bbox, "hits": tr.hits, "misses": tr.misses})
        out.sort(key=lambda d: d["id"])
        return out