# src/gaze_estimator/gazelle.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from PIL import Image


BBox = Tuple[int, int, int, int]  # (x, y, w, h)


class GazeLLE:
    """
    Wrapper around the official Gaze-LLE (Gazelle) model.

    - Loads from the installed 'gazelle' package if available, else tries PyTorch Hub.
    - Single image encode + multi-person heads: pass the full frame and list of face bboxes.
    - Outputs per-face:
        * target_xy (int,int) in image pixels (argmax of 64x64 heatmap)
        * inout (float in [0,1]) when using an *_inout* model; else None
        * label: coarse direction 'forward/left/right/up/down' or 'out-of-frame'
    - No fallbacks to other gaze models (as requested).

    Methods:
        estimate_many(frame_bgr, face_bboxes_px) -> List[Dict]
    """

    def __init__(
        self,
        model_name: str = "gazelle_dinov2_vitb14_inout",
        device: Optional[str] = None,
        forward_px_thresh: float = 0.03,  # percentage of max(H,W) below which we call 'forward'
    ) -> None:
        self.model_name = model_name
        self.device = (
            device
            if device
            else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        self.forward_px_thresh = forward_px_thresh
        self.model, self.transform = self._load_model_and_transform(model_name)
        self.model.to(self.device).eval()
        print(f"✅ Gazelle enabled: {model_name} on {self.device}")

    # ---- internal helpers ----------------------------------------------------

    def _load_model_and_transform(self, model_name: str):
        # Try official package first (you installed it with `pip install -e /path/to/gazelle`)
        try:
            from gazelle.model import get_gazelle_model  # type: ignore
            model, transform = get_gazelle_model(model_name)
            return model, transform
        except Exception as e:
            print(f"ℹ️ get_gazelle_model not available ({e}); trying torch.hub...")
        # Fallback to PyTorch Hub (still the official repo)
        model, transform = torch.hub.load("fkryan/gazelle", model_name)
        return model, transform

    @staticmethod
    def _normalize_bboxes(face_bboxes_px: List[BBox], W: int, H: int) -> List[Tuple[float, float, float, float]]:
        """Convert (x,y,w,h) pixel boxes -> normalized (xmin, ymin, xmax, ymax) in [0,1]."""
        normed = []
        for (x, y, w, h) in face_bboxes_px:
            x1 = max(0, min(W, x))
            y1 = max(0, min(H, y))
            x2 = max(0, min(W, x + w))
            y2 = max(0, min(H, y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            normed.append((x1 / W, y1 / H, x2 / W, y2 / H))
        return normed

    @staticmethod
    def _heatmap_argmax_to_xy(heatmap: torch.Tensor, W: int, H: int) -> Tuple[int, int]:
        """
        heatmap: [64,64] tensor with values in [0,1].
        Return integer pixel (x,y) in the original image coordinates.
        """
        if heatmap.ndim == 3:  # sometimes [1,64,64]
            heatmap = heatmap.squeeze(0)
        idx = torch.argmax(heatmap).item()
        h, w = heatmap.shape[-2], heatmap.shape[-1]  # expect 64,64
        y_idx, x_idx = divmod(idx, w)
        # Map center of the cell to image
        x = int(((x_idx + 0.5) / w) * W)
        y = int(((y_idx + 0.5) / h) * H)
        return x, y

    @staticmethod
    def _face_center(face_bbox: BBox) -> Tuple[int, int]:
        x, y, w, h = face_bbox
        return int(x + w / 2), int(y + h / 2)

    def _coarse_label(self, face_center: Tuple[int, int], target_xy: Tuple[int, int], inout: Optional[float], W: int, H: int) -> str:
        if inout is not None and inout < 0.5:
            return "out-of-frame"
        cx, cy = face_center
        tx, ty = target_xy
        dx, dy = tx - cx, ty - cy
        mag = np.hypot(dx, dy)
        fwd_thresh = self.forward_px_thresh * max(W, H)
        if mag < fwd_thresh:
            return "forward"
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    # ---- public API ----------------------------------------------------------

    @torch.inference_mode()
    def estimate_many(self, frame_bgr: np.ndarray, face_bboxes_px: List[BBox]) -> List[Dict]:
        """
        Args:
            frame_bgr: HxWx3 BGR frame (numpy uint8)
            face_bboxes_px: list of (x,y,w,h) in pixels for faces in this frame

        Returns (aligned with input boxes):
            List[{
                "target_xy": (x,y) int pixels,
                "inout": float|None,
                "label": str in {"forward","left","right","up","down","out-of-frame"}
            }]
        """
        H, W = frame_bgr.shape[:2]
        if not face_bboxes_px:
            return []

        # Prepare input
        img_rgb = Image.fromarray(frame_bgr[:, :, ::-1].copy())  # BGR -> RGB
        images = self.transform(img_rgb).unsqueeze(0).to(self.device)  # [1,3,448,448]
        bboxes_norm = self._normalize_bboxes(face_bboxes_px, W, H)
        if not bboxes_norm:
            return []

        model_input = {"images": images, "bboxes": [bboxes_norm]}  # list-of-lists shape for batch=1

        out = self.model(model_input)
        # out["heatmap"]: [B=1, P, 64, 64]; out.get("inout"): [B=1, P]
        heatmaps = out["heatmap"][0]  # [P,64,64]
        inout = out.get("inout", None)
        inout_row = None
        if inout is not None:
            inout_row = inout[0].detach().cpu().float().tolist()  # [P]
        results: List[Dict] = []

        # Align outputs with the provided, valid bboxes_norm
        p = heatmaps.shape[0]
        for i in range(p):
            hm = heatmaps[i].detach().cpu()
            target_xy = self._heatmap_argmax_to_xy(hm, W, H)
            inout_prob = float(inout_row[i]) if inout_row is not None else None
            # Map back to the original pixel bbox that corresponds to this i
            face_bbox_px = face_bboxes_px[i]
            face_center = self._face_center(face_bbox_px)
            label = self._coarse_label(face_center, target_xy, inout_prob, W, H)
            results.append(
                {
                    "target_xy": target_xy,
                    "inout": inout_prob,
                    "label": label,
                }
            )
        return results