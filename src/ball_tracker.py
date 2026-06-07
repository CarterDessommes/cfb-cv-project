"""ROI-based football tracking helper."""

import numpy as np

from .team_classifier import _best_device
from .yolo_utils import boxes_to_cpu_arrays


class BallTracker:
    """Tracks the single ball with ROI-cropped detection and motion coasting.

    The ball is small and sparse, so most frames only need a small crop around the
    predicted position run at a reduced ``imgsz``. A periodic full-frame pass
    re-acquires the ball and corrects drift.
    """

    def __init__(self, model_path: str, ball_conf: float = 0.3,
                 roi_size: int = 320, full_every: int = 30, max_coast: int = 3):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = _best_device()
        self.predict_kwargs = {"device": self.device, "verbose": False, "half": True}

        self.ball_conf = ball_conf
        self.roi_size = roi_size
        # imgsz must be a multiple of 32; the crop is square so one size covers it.
        self.roi_imgsz = max(32, (roi_size // 32) * 32)
        self.full_imgsz = 640
        self.full_every = full_every
        self.max_coast = max_coast
        # When the ball is fully lost, only attempt the (expensive) full-frame
        # re-acquire every Nth frame rather than every frame.
        self.lost_acquire_every = 3

        self.last_xy: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.misses = 0
        self.frames_since_full = 0
        self.lost_frames = 0

    def warmup(self, frame_shape):
        full_dummy = np.zeros(frame_shape, dtype=np.uint8)
        self.model(full_dummy, imgsz=self.full_imgsz, **self.predict_kwargs)
        roi_dummy = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
        self.model(roi_dummy, imgsz=self.roi_imgsz, **self.predict_kwargs)

    def _best_center(self, boxes_obj, ox: float = 0.0, oy: float = 0.0):
        """Center of the highest-confidence ball box, offset into full-frame coords."""
        xyxy, _, _, confs = boxes_to_cpu_arrays(boxes_obj)
        if xyxy is None or confs is None or len(confs) == 0:
            return None
        box = xyxy[int(confs.argmax())]
        return float((box[0] + box[2]) / 2) + ox, float((box[1] + box[3]) / 2) + oy

    def _detect_full(self, frame):
        self.frames_since_full = 0
        results = self.model(frame, imgsz=self.full_imgsz, conf=self.ball_conf,
                             **self.predict_kwargs)
        return self._best_center(results[0].boxes)

    def _detect_roi(self, frame, px: float, py: float):
        self.frames_since_full += 1
        h, w = frame.shape[:2]
        half = self.roi_size // 2
        x1 = int(max(0, min(w - 1, px - half)))
        y1 = int(max(0, min(h - 1, py - half)))
        x2 = int(max(0, min(w, px + half)))
        y2 = int(max(0, min(h, py + half)))
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return self._detect_full(frame)
        results = self.model(crop, imgsz=self.roi_imgsz, conf=self.ball_conf,
                             **self.predict_kwargs)
        det = self._best_center(results[0].boxes, ox=x1, oy=y1)
        if det is None:
            return self._detect_full(frame)
        return det

    def update(self, frame) -> tuple[float, float] | None:
        """Detect/predict the ball for this frame; returns (cx, cy) or None."""
        use_full = (self.last_xy is None
                    or self.misses >= self.max_coast
                    or self.frames_since_full >= self.full_every)
        if use_full:
            # Ball fully lost: throttle the full-frame re-acquire to every Nth frame.
            if self.last_xy is None:
                attempt = (self.lost_frames % self.lost_acquire_every == 0)
                self.lost_frames += 1
                if not attempt:
                    return None
            det = self._detect_full(frame)
        else:
            px = self.last_xy[0] + self.velocity[0]
            py = self.last_xy[1] + self.velocity[1]
            det = self._detect_roi(frame, px, py)

        if det is not None:
            self.lost_frames = 0
            if self.last_xy is not None:
                vx = det[0] - self.last_xy[0]
                vy = det[1] - self.last_xy[1]
                self.velocity = (0.5 * vx + 0.5 * self.velocity[0],
                                 0.5 * vy + 0.5 * self.velocity[1])
            self.last_xy = det
            self.misses = 0
        else:
            self.misses += 1
            if self.last_xy is not None and self.misses <= self.max_coast:
                # Coast on last velocity so the next ROI stays centered.
                self.last_xy = (self.last_xy[0] + self.velocity[0],
                                self.last_xy[1] + self.velocity[1])
            else:
                self.last_xy = None
                self.velocity = (0.0, 0.0)
        return self.last_xy
