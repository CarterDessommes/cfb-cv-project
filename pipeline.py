"""
Full field-state pipeline: player location + team + jersey number.

Usage:
    python pipeline.py <video> [--det PATH] [--ocr PATH] [--ball PATH] [--out FILE]
                              [--conf N] [--imgsz N] [--no-ball] [--ball-11m]
                              [--det-every N] [--ocr-every N] [--ball-every N]

Defaults:
    --det        weights/player-best.pt
    --ocr        weights/jersey_best.pt
    --ball       weights/ball-best.pt (YOLO 12M)
    --ball-11m   use weights/ball-yolo11m.pt (YOLO 11M) instead
    --conf       0.4
    --imgsz      480 (YOLO inference resolution for player detection; must be a
                      multiple of 32. Smaller = faster but lower accuracy — one of
                      the two biggest compute levers for player detection)
    --imgsz2     960 (dual-resolution detection: the detector runs a second pass
                      at this resolution and the two detection sets are merged
                      with class-aware IoU NMS before tracking, to catch players
                      bunched at the line of scrimmage. Set 0 to disable (single
                      pass, ~20-28% faster but lower recall). Doubles detector
                      cost on detection frames.)
    --nms-iou    0.6 (IoU threshold for the dual-resolution merge; higher keeps
                      adjacent (overlapping) players separate)
    --det-every  1   (run player detection+tracking every Nth frame; set >1 to
                      reuse prior tracks between detections. The detector net is
                      the dominant per-frame cost, so this is an opt-in speed knob.)
    --ocr-every  10  (run jersey OCR every Nth frame; reuse cached numbers between)
    --ball-every 2   (run ball tracker every Nth frame; reuse last position between.
                      ROI passes are cheap, so 1 (every frame) is now affordable)
    --ball-roi   320 (square px window cropped around the predicted ball; the crop is
                      run at a reduced imgsz, so smaller = faster)
    --ball-full-every 30 (force a full-frame re-acquire every N ball passes to fix drift)
"""

import sys
import time
import cv2
import numpy as np

from team_classifier import TeamClassifier, _best_device
from field_mapper import (
    project_players,
    build_field_canvas,
    field_to_canvas_point,
    load_homographies,
    draw_field_overlay,
    _homography_for_frame,
    CANVAS_SCALE,
)
from yolo_utils import boxes_to_cpu_arrays
from tracker import get_tracker_config_path
from ball_tracker import BallTracker
from multiscale import merge_detections, MultiScaleTracker

_NUMBER_HISTORY: dict[int, list[tuple[str, float]]] = {}   # track_id -> recent (pred, conf) reads
_NUMBER_LOCKED:  dict[int, str] = {}                        # track_id -> permanently decided number
_VOTE_WINDOW = 75  # last 75 accepted OCR reads (with --ocr-every 10 that spans ~25 s @ 30 fps)
_LOCK_VOTES  = 8   # confident reads agreeing before a tracklet's number is permanently locked


def _stable_number(track_id: int, pred: str, conf: float) -> str:
    if track_id in _NUMBER_LOCKED:
        return _NUMBER_LOCKED[track_id]
    history = _NUMBER_HISTORY.setdefault(track_id, [])
    history.append((pred, conf))
    if len(history) > _VOTE_WINDOW:
        history.pop(0)
    weights: dict[str, float] = {}
    for p, c in history:
        weights[p] = weights.get(p, 0.0) + c
    best = max(weights, key=weights.__getitem__)
    if sum(1 for p, _ in history if p == best) >= _LOCK_VOTES:
        _NUMBER_LOCKED[track_id] = best
    return best


def _due(frame_num: int, every: int) -> bool:
    """True on frames where a throttled model should run (frames 1, 1+every, ...)."""
    return every <= 1 or frame_num % every == 1


COLORS = {
    "offense": (0, 200, 255),
    "defense": (255, 100,   0),
    "unknown": (128, 128, 128),
}

_MIN_CROP_PX = 10
_OCR_CONF_THRESHOLD = 0.45  # min classifier confidence to accept a read
_BLUR_THRESHOLD     = 20    # min Laplacian variance; below = too blurry to read
_CROP_BOTTOM        = 0.95  # keep from box top down to this fraction of player height
_CROP_WIDTH_PAD     = 0.15  # widen past the bbox left/right by this fraction


def _number_crop(frame, box):
    """Generous crop covering (nearly) the whole player so the chest number is
    never clipped by an off-center number or out-flung arms. It's a wide net."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = y2 - y1, x2 - x1
    ty1 = max(0, y1)
    ty2 = min(frame.shape[0], y1 + int(h * _CROP_BOTTOM))
    sx1 = max(0, x1 - int(w * _CROP_WIDTH_PAD))
    sx2 = min(frame.shape[1], x2 + int(w * _CROP_WIDTH_PAD))
    return frame[ty1:ty2, sx1:sx2]


def _crops(frame, boxes):
    crops, indices = [], []
    for i, box in enumerate(boxes):
        crop = _number_crop(frame, box)
        if crop.shape[0] >= _MIN_CROP_PX and crop.shape[1] >= _MIN_CROP_PX:
            crops.append(np.ascontiguousarray(crop))
            indices.append(i)
    return crops, indices


def _nearest_player_to_ball(ball_xy, boxes: list) -> int | None:
    """Return the track id of the visible player nearest the detected ball."""
    if ball_xy is None or not boxes:
        return None

    bx, by = float(ball_xy[0]), float(ball_xy[1])
    nearest_track_id = None
    nearest_dist = None
    for box in boxes:
        if len(box) < 5:
            continue
        cx = (float(box[0]) + float(box[2])) / 2
        cy = (float(box[1]) + float(box[3])) / 2
        dist = np.hypot(bx - cx, by - cy)
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_track_id = int(box[4])
    return nearest_track_id


class JerseyOCR:
    def __init__(self, model_path: str):
        from ultralytics import YOLO

        self.model  = YOLO(model_path)
        self.device = _best_device()
        self.predict_kwargs = {"device": self.device, "verbose": False}

    def warmup(self):
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        self.model([dummy], **self.predict_kwargs)

    def predict(self, crops: list) -> list[tuple]:
        """One (top1, top5, blur_var) tuple per crop:
          top1     = (label, conf) if not blurry and conf >= _OCR_CONF_THRESHOLD, else None
          top5     = [(label, conf), ...] (5 entries), or [] when the blur gate fails
          blur_var = Laplacian variance of the crop
        A crop failing the blur gate yields (None, [], blur_var); a crop whose best
        guess is below the confidence gate yields (None, top5, blur_var)."""
        if not crops:
            return []
        results = self.model(crops, **self.predict_kwargs)
        out = []
        for crop, r in zip(crops, results):
            blur_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            if blur_var < _BLUR_THRESHOLD:
                out.append((None, [], blur_var))
                continue
            top5_idx  = r.probs.top5
            top5_conf = r.probs.top5conf.cpu().numpy()
            top5      = [(r.names[i], float(c)) for i, c in zip(top5_idx, top5_conf)]
            top1_label, top1_conf = top5[0]
            top1 = (top1_label, top1_conf) if top1_conf >= _OCR_CONF_THRESHOLD else None
            out.append((top1, top5, blur_var))
        return out


_DEBUG_PANEL_H = 200   # height (px) of the bottom OCR debug strip
_DEBUG_ENTRY_W = 150   # width (px) per crop entry within the strip


def _build_ocr_panel(crops, ocr_results, track_ids, panel_w: int) -> np.ndarray:
    """Bottom debug strip: each crop thumbnail + its top-5 predictions, laid out
    left-to-right. Sized to span the full width of the main display so it can be
    vstacked below it."""
    panel = np.full((_DEBUG_PANEL_H, panel_w, 3), 28, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    x = 6
    for crop, (top1, top5, blur_var), tid in zip(crops, ocr_results, track_ids):
        if x + _DEBUG_ENTRY_W > panel_w:
            break
        cv2.putText(panel, f"#{tid} blur={blur_var:.0f}", (x, 14), font, 0.38, (160, 160, 160), 1)

        thumb_w = _DEBUG_ENTRY_W - 12
        h, w    = crop.shape[:2]
        scale   = thumb_w / max(w, 1)
        thumb_h = min(max(1, int(h * scale)), 90)
        thumb   = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        panel[20:20 + thumb_h, x:x + thumb_w] = thumb

        ty = 20 + thumb_h + 14
        if not top5:
            cv2.putText(panel, "BLURRY", (x, ty), font, 0.4, (80, 80, 200), 1)
        else:
            for rank, (label, conf) in enumerate(top5):
                is_winner = (rank == 0 and top1 is not None)
                color = (0, 210, 100) if is_winner else (130, 130, 130)
                cv2.putText(panel, f"{rank+1}. {label}: {conf:.2f}", (x, ty), font, 0.36, color, 1)
                ty += 14
                if ty > _DEBUG_PANEL_H - 4:
                    break
        x += _DEBUG_ENTRY_W

    if not crops:
        cv2.putText(panel, "No crops this frame", (6, 20), font, 0.4, (100, 100, 100), 1)
    return panel


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1)
    return float(ordered[idx])


def run_pipeline_benchmark(video_path, frame_numbers, det_model_path="weights/player-best.pt",
                           ocr_model_path="weights/jersey_best.pt",
                           ball_model_path="weights/ball-best.pt",
                           homography_path="homographies.npz", conf=0.4,
                           ocr_every=10, ball_every=1, det_imgsz=480, det_every=1,
                           det_imgsz2=960, nms_iou=0.6):
    """Run the field-state pipeline headlessly and return predictions + timings.

    `frame_numbers` are zero-based video frame indices, matching OpenCV and the
    exported benchmark PNG filenames.
    """
    from ultralytics import YOLO

    target_frames = sorted({int(n) for n in frame_numbers if int(n) >= 0})
    if not target_frames:
        raise ValueError("frame_numbers must include at least one frame index")

    _NUMBER_HISTORY.clear()
    _NUMBER_LOCKED.clear()
    load_start = time.perf_counter()
    detector      = YOLO(det_model_path)
    device        = _best_device()
    classifier    = TeamClassifier()
    ocr           = JerseyOCR(ocr_model_path)
    ball_detector = BallTracker(ball_model_path) if ball_model_path else None
    homography    = load_homographies(homography_path)
    # Opt-in dual-resolution detection: merge a second higher-res pass via NMS,
    # then drive a manually-managed BoT-SORT (Re-ID off) for stable track IDs.
    mst = MultiScaleTracker(get_tracker_config_path(multiscale=True)) if det_imgsz2 else None
    model_load_seconds = time.perf_counter() - load_start

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame_index = min(max(target_frames), total - 1) if total > 0 else max(target_frames)
    target_set = set(target_frames)

    fitted = False
    ball_xy: tuple[float, float] | None = None
    boxes: list = []   # persisted across frames where detection is skipped
    cluster_by_id: dict[int, int] = {}
    number_by_id: dict[int, str] = {}
    frame_latencies: list[float] = []
    predictions_by_frame: dict[int, dict] = {}

    process_start = time.perf_counter()
    frame_num = 0
    while cap.isOpened() and frame_num <= max_frame_index:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame_index = frame_num - 1
        frame_start = time.perf_counter()

        # Player detection+tracking is the dominant per-frame cost, so run it on a
        # stride and reuse the prior tracks between (persist=True keeps the IDs).
        if _due(frame_num, det_every):
            if mst is not None:
                merged = merge_detections(
                    detector, frame, conf, device, [det_imgsz, det_imgsz2], nms_iou
                )
                tracks = mst.update(merged, frame)
                boxes = [[t[0], t[1], t[2], t[3], int(t[4]), int(t[6])]
                         for t in tracks if int(t[6]) == 0]
            else:
                results = detector.track(
                    frame, persist=True, conf=conf, verbose=False,
                    device=device, half=True, imgsz=det_imgsz,
                )

                boxes = []
                xyxy, ids, clss, _ = boxes_to_cpu_arrays(results[0].boxes)
                if xyxy is not None and ids is not None and clss is not None:
                    for box, tid, cls in zip(xyxy, ids, clss):
                        if cls == 0:
                            boxes.append([*box, tid, cls])

        # ── Ball: run before fit so ball_xy is available for offense labeling ──
        if ball_detector and _due(frame_num, ball_every):
            ball_xy = ball_detector.update(frame)

        if boxes and not fitted:
            fitted = classifier.fit(frame, boxes, ball_xy=ball_xy)

        if fitted and boxes:
            new = [b for b in boxes if int(b[4]) not in cluster_by_id]
            if new:
                for b, c in zip(new, classifier.assign_clusters(frame, new)):
                    if c >= 0:
                        cluster_by_id[int(b[4])] = c

        def _clusters():
            return ([cluster_by_id.get(int(b[4]), -1) for b in boxes]
                    if fitted else [-1] * len(boxes))

        def _labels(clusters):
            return ([classifier.cluster_to_label(c) for c in clusters]
                    if fitted else ["unknown"] * len(boxes))

        team_clusters = _clusters()
        team_labels   = _labels(team_clusters)

        if _due(frame_num, ocr_every):
            crops, valid_idx = _crops(frame, boxes)
            ocr_results = ocr.predict(crops)
            for j, (top1, top5, blur_var) in enumerate(ocr_results):
                if top1 is None:   # blurry or low-confidence read; don't dilute the vote
                    continue
                label, read_conf = top1   # NB: not `conf` — that's the detector threshold
                tid = int(boxes[valid_idx[j]][4])
                number_by_id[tid] = _stable_number(tid, label, read_conf)

        field_state = []
        for box, team, cluster in zip(boxes, team_labels, team_clusters):
            track_id = int(box[4])
            field_state.append({
                "track_id": track_id,
                "bbox": [int(v) for v in box[:4]],
                "team": team,
                "_team_cluster": int(cluster),
                "number": number_by_id.get(track_id, "?"),
            })

        detections_for_mapper = [
            {"track_id": p["track_id"], "bbox": p["bbox"], "class": 0}
            for p in field_state
        ]
        field_points = project_players(
            frame_num=frame_num - 1,
            detections=detections_for_mapper,
            homography=homography,
        )
        fp_by_id = {p["track_id"]: p for p in field_points}
        for p in field_state:
            fp = fp_by_id.get(p["track_id"])
            p["field_x"] = fp["field_x"] if fp else None
            p["field_y"] = fp["field_y"] if fp else None

        frame_latencies.append(time.perf_counter() - frame_start)
        if frame_index in target_set:
            predictions_by_frame[frame_index] = {
                "frame_number": frame_index,
                "players": field_state,
                "ball": {"center": [float(ball_xy[0]), float(ball_xy[1])]} if ball_xy else None,
            }

    cap.release()
    process_seconds = time.perf_counter() - process_start
    processed_frames = frame_num
    predictions = [predictions_by_frame[n] for n in target_frames if n in predictions_by_frame]
    for frame_prediction in predictions:
        for player in frame_prediction["players"]:
            player["team"] = classifier.cluster_to_label(player.pop("_team_cluster", -1))

    return {
        "video_path": str(video_path),
        "target_frames": target_frames,
        "processed_frames": processed_frames,
        "total_frames": total,
        "predictions": predictions,
        "timings": {
            "model_load_seconds": model_load_seconds,
            "processing_seconds": process_seconds,
            "total_seconds": model_load_seconds + process_seconds,
            "fps": processed_frames / process_seconds if process_seconds > 0 else 0.0,
            "mean_frame_seconds": float(np.mean(frame_latencies)) if frame_latencies else 0.0,
            "p95_frame_seconds": _p95(frame_latencies),
        },
    }


def run_pipeline(video_path, det_model_path, ocr_model_path, ball_model_path=None,
                 output_path=None, conf=0.4, ocr_warning=False,
                 ocr_every=10, ball_every=2, ball_roi=320, ball_full_every=30,
                 det_imgsz=480, det_every=1, det_imgsz2=960, nms_iou=0.6,
                 show_grid=False, homography_path="homographies.npz"):
    from ultralytics import YOLO

    _NUMBER_HISTORY.clear()
    _NUMBER_LOCKED.clear()
    _TRAIL: dict[int, list] = {}   # track_id -> list of (cx, cy) canvas points
    _TRAIL_MAX = 45
    detector       = YOLO(det_model_path)
    tracker_config = get_tracker_config_path()
    # Opt-in dual-resolution detection: merge a second higher-res pass via NMS,
    # then drive a manually-managed BoT-SORT (Re-ID off) for stable track IDs.
    mst = MultiScaleTracker(get_tracker_config_path(multiscale=True)) if det_imgsz2 else None
    device        = _best_device()
    classifier    = TeamClassifier()
    ocr           = JerseyOCR(ocr_model_path)
    ball_tracker  = (BallTracker(ball_model_path, roi_size=ball_roi,
                                 full_every=ball_full_every)
                     if ball_model_path else None)
    fitted        = False

    cap    = cv2.VideoCapture(video_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width > 0 and height > 0:
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        detector(dummy_frame, device=device, half=True, verbose=False, imgsz=det_imgsz)
        if det_imgsz2:
            detector(dummy_frame, device=device, half=True, verbose=False, imgsz=det_imgsz2)
        if ball_tracker:
            ball_tracker.warmup(dummy_frame.shape)
    ocr.warmup()

    writer = None

    window = "Field State"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Static top-down field background — built once, copied each frame
    base_canvas = build_field_canvas(scale=CANVAS_SCALE)

    # Load homographies once; indexed per frame in memory inside the loop.
    homography = load_homographies(homography_path)

    frame_num = 0
    ball_xy: tuple[float, float] | None = None   # persisted across throttled frames
    boxes: list = []                             # persisted across det-skipped frames
    cluster_by_id: dict[int, int] = {}           # track_id -> team cluster (0/1)
    number_by_id:  dict[int, str] = {}           # track_id -> stable jersey number
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # ── Player detection+tracking: strided, reuse prior tracks between ────
        # The detector net is the dominant per-frame cost, so run it every Nth
        # frame and coast on the previous tracks. persist=True keeps BoT-SORT IDs
        # stable across the skipped frames (see tracker.py for the same pattern).
        if _due(frame_num, det_every):
            if mst is not None:
                merged = merge_detections(
                    detector, frame, conf, device, [det_imgsz, det_imgsz2], nms_iou
                )
                tracks = mst.update(merged, frame)
                boxes = [[t[0], t[1], t[2], t[3], int(t[4]), int(t[6])]
                         for t in tracks if int(t[6]) == 0]  # players only
            else:
                results = detector.track(
                    frame, tracker=tracker_config, persist=True, conf=conf, verbose=False,
                    device=device, half=True, imgsz=det_imgsz,
                )

                boxes = []
                xyxy, ids, clss, _ = boxes_to_cpu_arrays(results[0].boxes)
                if xyxy is not None and ids is not None and clss is not None:
                    for box, tid, cls in zip(xyxy, ids, clss):
                        if cls == 0:  # players only
                            boxes.append([*box, tid, cls])

        # ── Ball: run before fit so ball_xy is available for offense labeling ──
        if ball_tracker and _due(frame_num, ball_every):
            ball_xy = ball_tracker.update(frame)

        if boxes and not fitted:
            fitted = classifier.fit(frame, boxes, ball_xy=ball_xy)

        # ── Team: classify only newly-seen track_ids; cache cluster by id ─────
        # A player's team is fixed for the life of its track, so SigLIP runs once
        # per track (event-driven) instead of every frame.
        if fitted and boxes:
            new = [b for b in boxes if int(b[4]) not in cluster_by_id]
            if new:
                for b, c in zip(new, classifier.assign_clusters(frame, new)):
                    if c >= 0:
                        cluster_by_id[int(b[4])] = c

        def _clusters():
            return ([cluster_by_id.get(int(b[4]), -1) for b in boxes]
                    if fitted else [-1] * len(boxes))

        def _labels(clusters):
            return ([classifier.cluster_to_label(c) for c in clusters]
                    if fitted else ["unknown"] * len(boxes))

        team_clusters = _clusters()
        team_labels   = _labels(team_clusters)
        ball_carrier_track_id = _nearest_player_to_ball(ball_xy, boxes)

        # ── Jersey OCR: throttled, cached by track_id with _stable_number ─────
        if _due(frame_num, ocr_every):
            crops, valid_idx = _crops(frame, boxes)
            ocr_results = ocr.predict(crops)
            for j, (top1, top5, blur_var) in enumerate(ocr_results):
                if top1 is None:   # blurry or low-confidence read; don't dilute the vote
                    continue
                label, read_conf = top1   # NB: not `conf` — that's the detector threshold
                tid = int(boxes[valid_idx[j]][4])
                number_by_id[tid] = _stable_number(tid, label, read_conf)

        field_state = []
        detections_for_mapper = []
        for i, (box, team) in enumerate(zip(boxes, team_labels)):
            track_id = int(box[4])
            number   = number_by_id.get(track_id, "?")
            bbox     = [int(v) for v in box[:4]]
            field_state.append({
                "track_id": track_id,
                "bbox":     bbox,
                "team":     team,
                "number":   number,
                "is_ball_carrier": track_id == ball_carrier_track_id,
            })
            detections_for_mapper.append({
                "track_id": track_id,
                "bbox":     bbox,
                "class":    0,
            })

            x1, y1, x2, y2 = bbox
            color = COLORS[team]
            num_str = f" #{number}" if number != "?" else ""   # blank when unreadable
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{team.upper()}{num_str}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── Grid overlay: project field lines onto broadcast frame ──────────
        if show_grid:
            H_cur = _homography_for_frame(homography, frame_num - 1)
            if H_cur is not None:
                draw_field_overlay(frame, H_cur)

        # ── Field mapping: project players to top-down field coords ─────────
        field_points = project_players(
            frame_num=frame_num - 1,   # frame_num is 1-indexed; npz is 0-indexed
            detections=detections_for_mapper,
            homography=homography,
        )
        # Add field coords back to field_state
        for p in field_state:
            p["field_x"] = None
            p["field_y"] = None
        for p, fp in zip(field_state, field_points):
            p["field_x"] = fp["field_x"]
            p["field_y"] = fp["field_y"]

        # ── Top-down canvas ───────────────────────────────────────────────────
        canvas = base_canvas.copy()
        # Color dots by team
        team_color_map = {
            "offense": (0, 200, 255),
            "defense": (255, 100, 0),
            "unknown": (128, 128, 128),
        }
        for p, fp in zip(field_state, field_points):
            if not fp["in_bounds"]:
                continue
            cx, cy = field_to_canvas_point(fp["field_x"], fp["field_y"], CANVAS_SCALE)
            color = team_color_map.get(p["team"], (128, 128, 128))

            trail = _TRAIL.setdefault(p["track_id"], [])
            trail.append((cx, cy))
            if len(trail) > _TRAIL_MAX:
                trail.pop(0)
            if len(trail) > 1:
                cv2.polylines(canvas, [np.array(trail, dtype=np.int32)], False, color, 1)

            cv2.circle(canvas, (cx, cy), 7, color, -1)
            cv2.circle(canvas, (cx, cy), 7, (255, 255, 255), 1)
            if p["is_ball_carrier"]:
                cv2.circle(canvas, (cx, cy), 11, (0, 255, 255), 2)
                cv2.circle(canvas, (cx, cy), 9, (0, 0, 0), 1)
                cv2.putText(canvas, "C", (cx - 4, cy - 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            number = p["number"]
            if number not in ("?", "unreadable"):   # unknown/unreadable render as a blank circle
                (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.putText(canvas, number, (cx - tw // 2, cy + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Resize canvas to match frame height and show side by side
        target_h = frame.shape[0]
        sf = target_h / canvas.shape[0]
        canvas_resized = cv2.resize(canvas, (int(canvas.shape[1] * sf), target_h))
        combined = np.hstack([frame, canvas_resized])

        if ball_xy:
            bx, by = int(ball_xy[0]), int(ball_xy[1])
            cv2.circle(combined, (bx, by), 8, (0, 255, 0), 2)
            cv2.putText(combined, "ball", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.putText(combined, f"Frame {frame_num}/{total}  players={len(field_state)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if ocr_warning:
            cv2.putText(combined, "WARNING: jersey # accuracy is low (experimental)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── OCR debug strip (--debug): stacked below, never touches main display ─
        if debug:
            dbg_crops, dbg_idx = _crops(frame, boxes)
            dbg_results = ocr.predict(dbg_crops)
            dbg_tids    = [int(boxes[i][4]) for i in dbg_idx]
            panel = _build_ocr_panel(dbg_crops, dbg_results, dbg_tids, combined.shape[1])
            combined = np.vstack([combined, panel])

        if output_path and writer is None:
            h_out, w_out = combined.shape[:2]
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_out, h_out))
        if writer:
            writer.write(combined)

        cv2.imshow(window, combined)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

        if frame_num % 10 == 0:
            print(f"\rFrame {frame_num}/{total}: {len(field_state)} players", end="")

    print(f"\nDone. Processed {frame_num} frames. Press any key to close.")
    cap.release()
    if writer:
        writer.release()

    # Save final top-down route map
    if output_path:
        import os
        base = os.path.splitext(output_path)[0]
        route_path = base + "_routes.png"
        cv2.imwrite(route_path, canvas)
        print(f"Saved route map: {route_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video> [--det PATH] [--ocr PATH] [--out FILE] "
              "[--conf N] [--imgsz N] [--imgsz2 N] [--nms-iou F] [--det-every N] "
              "[--ocr-every N] [--ball-every N] "
              "[--ball-roi N] [--ball-full-every N] [--ball-11m] [--no-ball]")
        sys.exit(1)

    video       = sys.argv[1]
    det_model   = "weights/player-best.pt"
    ocr_model   = "weights/jersey_best.pt"
    ball_model  = "weights/ball-best.pt"
    out_path    = None
    conf        = 0.4
    det_imgsz   = 480
    det_imgsz2  = 960
    nms_iou     = 0.6
    det_every   = 1
    ocr_warning = False
    ocr_every   = 10
    ball_every  = 2
    ball_roi    = 320
    ball_full_every = 30
    debug       = False
    show_grid   = False
    homography_path = "homographies.npz"

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--det" and i + 1 < len(sys.argv):
            det_model = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--ocr" and i + 1 < len(sys.argv):
            ocr_model = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--ball" and i + 1 < len(sys.argv):
            ball_model = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--no-ball":
            ball_model = None; i += 1
        elif sys.argv[i] == "--ball-11m":
            ball_model = "weights/ball-yolo11m.pt"; i += 1
        elif sys.argv[i] == "--out" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--conf" and i + 1 < len(sys.argv):
            conf = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--imgsz" and i + 1 < len(sys.argv):
            det_imgsz = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--imgsz2" and i + 1 < len(sys.argv):
            det_imgsz2 = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--nms-iou" and i + 1 < len(sys.argv):
            nms_iou = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--det-every" and i + 1 < len(sys.argv):
            det_every = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--ocr-every" and i + 1 < len(sys.argv):
            ocr_every = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--ball-every" and i + 1 < len(sys.argv):
            ball_every = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--ball-roi" and i + 1 < len(sys.argv):
            ball_roi = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--ball-full-every" and i + 1 < len(sys.argv):
            ball_full_every = int(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--ocr-warning":
            ocr_warning = True; i += 1
        elif sys.argv[i] == "--debug":
            debug = True; i += 1
        elif sys.argv[i] == "--show-grid":
            show_grid = True; i += 1
        elif sys.argv[i] == "--homography" and i + 1 < len(sys.argv):
            homography_path = sys.argv[i + 1]; i += 2
        else:
            i += 1

    run_pipeline(video, det_model, ocr_model, ball_model, out_path, conf, ocr_warning,
                 ocr_every=ocr_every, ball_every=ball_every,
                 ball_roi=ball_roi, ball_full_every=ball_full_every,
                 det_imgsz=det_imgsz, det_every=det_every,
                 det_imgsz2=det_imgsz2, nms_iou=nms_iou,
                 show_grid=show_grid, homography_path=homography_path)
