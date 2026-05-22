"""
Lightweight BoT-SORT implementation.
No PyTorch - uses numpy, scipy, opencv only.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from collections import deque


class KalmanFilter:
    """Kalman filter for bounding box tracking in [cx, cy, w, h] format."""

    def __init__(self):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.dim_x = 8
        self.dim_z = 4

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x)
        self.F[:4, 4:] = np.eye(4)

        # Measurement matrix
        self.H = np.eye(self.dim_z, self.dim_x)

        # Process noise
        self.Q = np.eye(self.dim_x)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(self.dim_z) * 10.0

        # State and covariance
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 100.0

    def init(self, bbox):
        """Initialize with bbox [cx, cy, w, h]."""
        self.x[:4] = bbox
        self.x[4:] = 0
        self.P = np.eye(self.dim_x) * 100.0

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, bbox):
        """Update with measurement [cx, cy, w, h]."""
        y = bbox - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        return self.x[:4].copy()

    def get_state(self):
        """Get current bbox estimate."""
        return self.x[:4].copy()


class Track:
    """Single object track."""

    _next_id = 1

    def __init__(self, bbox, conf, cls):
        self.id = Track._next_id
        Track._next_id += 1

        self.kf = KalmanFilter()
        self.kf.init(bbox)

        self.conf = conf
        self.cls = cls
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "tentative"  # tentative -> confirmed -> lost

        # Track history for visualization
        self.history = deque(maxlen=100)
        self.history.append(bbox[:2].copy())

    def predict(self):
        """Predict next position."""
        self.age += 1
        self.time_since_update += 1
        bbox = self.kf.predict()
        return bbox

    def update(self, bbox, conf, cls):
        """Update with matched detection."""
        self.kf.update(bbox)
        self.conf = conf
        self.cls = cls
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.kf.get_state()[:2].copy())

        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"

    def mark_lost(self):
        """Mark track as lost."""
        self.state = "lost"

    @property
    def bbox(self):
        """Current bbox [cx, cy, w, h]."""
        return self.kf.get_state()

    @property
    def xyxy(self):
        """Current bbox [x1, y1, x2, y2]."""
        cx, cy, w, h = self.bbox
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    @staticmethod
    def reset_id():
        Track._next_id = 1


class CameraMotionCompensation:
    """Camera motion compensation using ORB features."""

    def __init__(self):
        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None

    def apply(self, frame, tracks):
        """Compute camera motion and apply to track states."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if self.prev_desc is not None and desc is not None and len(kp) >= 8:
            matches = self.bf.knnMatch(self.prev_desc, desc, k=2)

            # Lowe's ratio test
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) >= 8:
                src = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
                dst = np.float32([kp[m.trainIdx].pt for m in good])

                # Estimate affine transform
                H, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)

                if H is not None:
                    # Apply transform to track positions
                    for track in tracks:
                        cx, cy = track.kf.x[0], track.kf.x[1]
                        new_pos = H @ np.array([cx, cy, 1])
                        track.kf.x[0] = new_pos[0]
                        track.kf.x[1] = new_pos[1]

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_desc = desc


def iou_batch(boxes1, boxes2):
    """Compute IoU between two sets of boxes in [cx, cy, w, h] format."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Convert to xyxy
    b1 = np.array(boxes1)
    b2 = np.array(boxes2)

    b1_xyxy = np.stack([
        b1[:, 0] - b1[:, 2]/2,
        b1[:, 1] - b1[:, 3]/2,
        b1[:, 0] + b1[:, 2]/2,
        b1[:, 1] + b1[:, 3]/2
    ], axis=1)

    b2_xyxy = np.stack([
        b2[:, 0] - b2[:, 2]/2,
        b2[:, 1] - b2[:, 3]/2,
        b2[:, 0] + b2[:, 2]/2,
        b2[:, 1] + b2[:, 3]/2
    ], axis=1)

    # Intersection
    inter_x1 = np.maximum(b1_xyxy[:, None, 0], b2_xyxy[None, :, 0])
    inter_y1 = np.maximum(b1_xyxy[:, None, 1], b2_xyxy[None, :, 1])
    inter_x2 = np.minimum(b1_xyxy[:, None, 2], b2_xyxy[None, :, 2])
    inter_y2 = np.minimum(b1_xyxy[:, None, 3], b2_xyxy[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area1 = b1[:, 2] * b1[:, 3]
    area2 = b2[:, 2] * b2[:, 3]
    union = area1[:, None] + area2[None, :] - inter_area

    return inter_area / np.maximum(union, 1e-6)


class BoTSORT:
    """BoT-SORT multi-object tracker."""

    def __init__(
        self,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.7,
        use_cmc=True,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        self.tracks = []
        self.lost_tracks = []
        self.frame_id = 0

        self.cmc = CameraMotionCompensation() if use_cmc else None

    def update(self, detections, frame=None):
        """
        Update tracker with detections.

        Args:
            detections: np.array of shape (N, 6) [x1, y1, x2, y2, conf, cls]
            frame: BGR image for camera motion compensation

        Returns:
            np.array of shape (M, 5) [x1, y1, x2, y2, track_id]
        """
        self.frame_id += 1

        # Camera motion compensation
        if self.cmc is not None and frame is not None:
            self.cmc.apply(frame, self.tracks + self.lost_tracks)

        # Convert detections from xyxy to cxcywh
        dets_cxcywh = []
        confs = []
        classes = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            dets_cxcywh.append([cx, cy, w, h])
            confs.append(conf)
            classes.append(int(cls))

        dets_cxcywh = np.array(dets_cxcywh) if dets_cxcywh else np.empty((0, 4))
        confs = np.array(confs)
        classes = np.array(classes)

        # Split detections by confidence
        if len(confs) > 0:
            high_mask = confs >= self.track_high_thresh
            low_mask = (confs >= self.track_low_thresh) & ~high_mask
        else:
            high_mask = np.array([], dtype=bool)
            low_mask = np.array([], dtype=bool)

        dets_high = dets_cxcywh[high_mask]
        dets_low = dets_cxcywh[low_mask]
        confs_high = confs[high_mask]
        confs_low = confs[low_mask]
        classes_high = classes[high_mask]
        classes_low = classes[low_mask]

        # Predict all tracks
        for track in self.tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        # === First association: confirmed tracks with high-conf detections ===
        confirmed = [t for t in self.tracks if t.state == "confirmed"]
        unmatched_tracks_1 = list(range(len(confirmed)))
        unmatched_dets_1 = list(range(len(dets_high)))

        if len(confirmed) > 0 and len(dets_high) > 0:
            track_boxes = np.array([t.bbox for t in confirmed])
            iou_matrix = iou_batch(track_boxes, dets_high)
            cost = 1 - iou_matrix

            row_idx, col_idx = linear_sum_assignment(cost)

            matched_1 = []
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < 1 - self.match_thresh:
                    matched_1.append((r, c))
                    unmatched_tracks_1.remove(r)
                    unmatched_dets_1.remove(c)

            for r, c in matched_1:
                confirmed[r].update(dets_high[c], confs_high[c], classes_high[c])

        # === Second association: remaining tracks with low-conf detections ===
        remaining_tracks = [confirmed[i] for i in unmatched_tracks_1]
        unmatched_tracks_2 = list(range(len(remaining_tracks)))

        if len(remaining_tracks) > 0 and len(dets_low) > 0:
            track_boxes = np.array([t.bbox for t in remaining_tracks])
            iou_matrix = iou_batch(track_boxes, dets_low)
            cost = 1 - iou_matrix

            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < 1 - self.match_thresh:
                    remaining_tracks[r].update(dets_low[c], confs_low[c], classes_low[c])
                    unmatched_tracks_2.remove(r)

        # === Third association: tentative tracks with remaining high-conf ===
        tentative = [t for t in self.tracks if t.state == "tentative"]

        if len(tentative) > 0 and len(unmatched_dets_1) > 0:
            det_indices = list(unmatched_dets_1)
            remaining_dets = dets_high[det_indices]
            track_boxes = np.array([t.bbox for t in tentative])
            iou_matrix = iou_batch(track_boxes, remaining_dets)
            cost = 1 - iou_matrix

            row_idx, col_idx = linear_sum_assignment(cost)

            matched_det_indices = set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < 1 - self.match_thresh:
                    orig_idx = det_indices[c]
                    tentative[r].update(
                        dets_high[orig_idx],
                        confs_high[orig_idx],
                        classes_high[orig_idx]
                    )
                    matched_det_indices.add(orig_idx)

            unmatched_dets_1 = [i for i in unmatched_dets_1 if i not in matched_det_indices]

        # === Fourth association: lost tracks with remaining high-conf ===
        if len(self.lost_tracks) > 0 and len(unmatched_dets_1) > 0:
            det_indices = list(unmatched_dets_1)
            remaining_dets = dets_high[det_indices]
            track_boxes = np.array([t.bbox for t in self.lost_tracks])
            iou_matrix = iou_batch(track_boxes, remaining_dets)
            cost = 1 - iou_matrix

            row_idx, col_idx = linear_sum_assignment(cost)

            reactivated = []
            matched_det_indices = set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < 1 - self.match_thresh:
                    orig_idx = det_indices[c]
                    self.lost_tracks[r].update(
                        dets_high[orig_idx],
                        confs_high[orig_idx],
                        classes_high[orig_idx]
                    )
                    self.lost_tracks[r].state = "confirmed"
                    reactivated.append(r)
                    matched_det_indices.add(orig_idx)

            unmatched_dets_1 = [i for i in unmatched_dets_1 if i not in matched_det_indices]

            for r in sorted(reactivated, reverse=True):
                self.tracks.append(self.lost_tracks.pop(r))

        # === Create new tracks from remaining high-conf detections ===
        for i in unmatched_dets_1:
            if confs_high[i] >= self.new_track_thresh:
                new_track = Track(dets_high[i], confs_high[i], classes_high[i])
                self.tracks.append(new_track)

        # === Handle unmatched tracks ===
        # Move unmatched confirmed tracks to lost
        still_unmatched = [confirmed[i] for i in unmatched_tracks_1 if i in unmatched_tracks_2 or i not in [j for j in range(len(remaining_tracks))]]
        for t in [confirmed[i] for i in unmatched_tracks_1]:
            if t.time_since_update > 0:
                t.mark_lost()
                self.lost_tracks.append(t)
                self.tracks.remove(t)

        # Remove dead tentative tracks
        self.tracks = [t for t in self.tracks if not (t.state == "tentative" and t.time_since_update > 2)]

        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.track_buffer]

        # === Output confirmed tracks ===
        outputs = []
        for track in self.tracks:
            if track.state == "confirmed" and track.time_since_update == 0:
                xyxy = track.xyxy
                outputs.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], track.id])

        return np.array(outputs) if outputs else np.empty((0, 5))

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.lost_tracks = []
        self.frame_id = 0
        Track.reset_id()
