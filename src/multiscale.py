"""Multi-resolution player detection.

Runs the detector at two resolutions and merges the two detection sets with
class-aware IoU NMS *before* tracking, so players bunched at the line of
scrimmage (missed at the low primary resolution) are caught by the higher
second pass. The merged boxes are fed to a manually-driven BoT-SORT instance so
track IDs stay stable for all downstream consumers (team clustering, jersey OCR,
homography projection, ball-carrier NN — all keyed on track_id).

This is the opt-in path used when a second resolution is configured; the
single-pass `model.track()` path is unchanged and remains the default.
"""

import numpy as np


def merge_detections(detector, frame, conf, device, imgsz_list, nms_iou):
    """Detect at each resolution in `imgsz_list` and merge via class-aware NMS.

    Returns a float32 (N, 6) array of [x1, y1, x2, y2, conf, cls]. Ultralytics
    rescales predict boxes back to the original frame coords, so detections from
    the different resolutions share one coordinate space and merge directly.
    """
    import torch
    from torchvision.ops import batched_nms

    rows = []
    for sz in imgsz_list:
        result = detector.predict(
            frame, imgsz=sz, conf=conf, device=device, half=True, verbose=False
        )[0]
        boxes = result.boxes.cpu().numpy()
        if len(boxes):
            rows.append(np.concatenate(
                [boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]], axis=1
            ))

    if not rows:
        return np.zeros((0, 6), dtype=np.float32)

    merged = np.concatenate(rows, axis=0).astype(np.float32)
    # Class-aware NMS (batched_nms offsets boxes per class internally) so a
    # referee box never suppresses an overlapping player box. CPU is fine — the
    # candidate count is in the tens.
    keep = batched_nms(
        torch.from_numpy(merged[:, :4]),
        torch.from_numpy(merged[:, 4]),
        torch.from_numpy(merged[:, 5]),
        nms_iou,
    ).numpy()
    return merged[keep]


class MultiScaleTracker:
    """A persisted BoT-SORT instance driven directly with merged detections.

    Mirrors Ultralytics' own `on_predict_postprocess_end`: build a numpy `Boxes`
    from the merged detections, call `tracker.update`, and strip the trailing
    det-index column. One instance per video reproduces `persist=True` ID
    continuity. Uses a `with_reid: false` config because native-feature Re-ID is
    only available on the `.track()` path (the merged path has no native feats).
    """

    def __init__(self, tracker_config_path):
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
        from ultralytics.trackers.bot_sort import BOTSORT

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(str(tracker_config_path))))
        self.tracker = BOTSORT(args=cfg)

    def update(self, merged, frame):
        """Track merged (N, 6) detections; return (M, 7) [x1,y1,x2,y2,id,conf,cls]."""
        from ultralytics.engine.results import Boxes

        boxes = Boxes(merged.astype(np.float32), frame.shape[:2])  # orig_shape = (h, w)
        tracks = self.tracker.update(boxes, frame, None)
        if len(tracks) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        return np.asarray(tracks)[:, :7]
