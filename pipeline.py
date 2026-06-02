"""
Full field-state pipeline: player location + team + jersey number.

Usage:
    python pipeline.py <video> [--det PATH] [--ocr PATH] [--ball PATH] [--out FILE] [--conf N] [--no-ball]

Defaults:
    --det        weights/player-best.pt
    --ocr        weights/best.pt
    --ball       weights/ball-best.pt
    --conf       0.4
    --pose-every 3    (run pose every N frames)
"""

import sys
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

from team_classifier import TeamClassifier, _best_device
from field_mapper import project_players, build_field_canvas, CANVAS_SCALE

_NUMBER_HISTORY: dict[int, list[tuple[str, float]]] = {}
_NUMBER_LOCKED:  dict[int, str] = {}
_VOTE_WINDOW = 75  # ~2.5 s at 30 fps
_LOCK_VOTES  = 8   # confident reads needed to permanently lock a tracklet's number


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


COLORS = {
    "offense": (0, 200, 255),
    "defense": (255, 100,   0),
    "unknown": (128, 128, 128),
}


def _put_text(img, text, pos, scale=0.5, color=(255, 255, 255), thickness=1):
    """Draw text with a solid black background rectangle for legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + baseline + 1), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

_MIN_CROP_PX = 10
_OCR_CONF_THRESHOLD = 0.35
_BLUR_THRESHOLD = 20
_SHOULDER_L, _SHOULDER_R = 5, 6
_HIP_L,      _HIP_R      = 11, 12
_KP_CONF_MIN = 0.3
_SIDE_FACING_THRESHOLD = 0.25  # shoulder span / bbox width; below = side-on, skip OCR


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def _pose_crop(frame, kps, box, fw):
    """Return (crop, rect) using pose for vertical placement and bbox for horizontal width."""
    ls, rs = kps[_SHOULDER_L], kps[_SHOULDER_R]
    lh, rh = kps[_HIP_L],      kps[_HIP_R]
    if min(ls[2], rs[2], lh[2], rh[2]) < _KP_CONF_MIN:
        return None, None
    shoulder_y = (ls[1] + rs[1]) / 2
    hip_y      = (lh[1] + rh[1]) / 2
    torso_h    = hip_y - shoulder_y
    if torso_h < 5:
        return None, None
    # Vertical: pose-guided — just below collar to ~65% of torso
    ty1 = int(shoulder_y + torso_h * 0.05)
    ty2 = int(shoulder_y + torso_h * 0.65)
    # Horizontal: central ~55% of the player bbox (avoids arms while wider than shoulder joints)
    bx1, bx2 = int(box[0]), int(box[2])
    cx        = (bx1 + bx2) // 2
    half_w    = int((bx2 - bx1) * 0.28)
    sx1       = max(0, cx - half_w)
    sx2       = min(fw, cx + half_w)
    crop = frame[ty1:ty2, sx1:sx2]
    if crop.shape[0] < _MIN_CROP_PX or crop.shape[1] < _MIN_CROP_PX:
        return None, None
    return crop, (sx1, ty1, sx2, ty2)


def _is_forward_facing(kps, box) -> bool:
    """False when the player is side-on and the jersey number is unreadable."""
    ls, rs = kps[_SHOULDER_L], kps[_SHOULDER_R]
    if ls[2] < _KP_CONF_MIN or rs[2] < _KP_CONF_MIN:
        return True  # can't determine orientation — don't filter
    shoulder_span = abs(rs[0] - ls[0])
    bbox_w = max(float(box[2]) - float(box[0]), 1.0)
    return (shoulder_span / bbox_w) > _SIDE_FACING_THRESHOLD


# Saved/inference crop geometry. Both dataset creation (label_crops.py) and
# inference (_crops below) use _number_crop so the model sees identical inputs.
# The narrow _pose_crop above is used only for *gating* (orientation/size),
# never for the crop we actually feed to the jersey reader.
_CROP_BOTTOM    = 0.95   # keep from box top down to this fraction of player height
_CROP_WIDTH_PAD = 0.15   # widen past the bbox left/right by this fraction


def _number_crop(frame, box):
    """Generous crop covering (nearly) the whole player so the chest number is
    never clipped by keypoint error, an off-center number, or out-flung arms."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = y2 - y1, x2 - x1
    ty1 = max(0, y1)
    ty2 = min(frame.shape[0], y1 + int(h * _CROP_BOTTOM))
    sx1 = max(0, x1 - int(w * _CROP_WIDTH_PAD))
    sx2 = min(frame.shape[1], x2 + int(w * _CROP_WIDTH_PAD))
    return frame[ty1:ty2, sx1:sx2]


def _crops(frame, boxes, pose_kps_list=None):
    crops, indices, rects = [], [], []
    for i, box in enumerate(boxes):
        crop = _number_crop(frame, box)
        if crop.shape[0] >= _MIN_CROP_PX and crop.shape[1] >= _MIN_CROP_PX:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crops.append(np.ascontiguousarray(crop))
            indices.append(i)
            rects.append((x1, y1, x2, y2))
    return crops, indices, rects


class BallDetector:
    def __init__(self, model_path: str):
        self.model  = YOLO(model_path)
        self.device = _best_device()

    def detect(self, frame) -> tuple[float, float] | None:
        """Returns (cx, cy) pixel coords of the highest-confidence ball, or None."""
        results = self.model(frame, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        idx = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[idx].cpu().numpy()
        return float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)


_OCR_PANEL_W     = 300   # width of the right-hand debug panel (px)
_OCR_PANEL_CROP_W = 260  # width each crop thumbnail is scaled to


def _build_ocr_panel(crops, ocr_results, track_ids, frame_h: int) -> np.ndarray:
    """Right-side debug panel: each crop + its top-5 predictions."""
    panel = np.full((frame_h, _OCR_PANEL_W, 3), 28, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    y     = 6

    for crop, result, tid in zip(crops, ocr_results, track_ids):
        top1, top5, blur_var = result
        entry_h = 0

        # header
        hdr = f"Track #{tid}   blur={blur_var:.0f}"
        cv2.putText(panel, hdr, (4, y + 13), font, 0.38, (160, 160, 160), 1)
        y += 18;  entry_h += 18

        # thumbnail
        h, w    = crop.shape[:2]
        scale   = _OCR_PANEL_CROP_W / max(w, 1)
        thumb_h = max(1, int(h * scale))
        thumb_h = min(thumb_h, frame_h // 6)
        thumb   = cv2.resize(crop, (_OCR_PANEL_CROP_W, thumb_h), interpolation=cv2.INTER_LANCZOS4)
        if y + thumb_h <= frame_h:
            panel[y:y+thumb_h, 4:4+_OCR_PANEL_CROP_W] = thumb
        y += thumb_h + 3;  entry_h += thumb_h + 3

        # top-5 guesses
        if not top5:
            cv2.putText(panel, "  BLURRY (filtered)", (4, y + 12), font, 0.37, (80, 80, 200), 1)
            y += 15;  entry_h += 15
        else:
            for rank, (label, conf) in enumerate(top5):
                is_winner = (rank == 0 and top1 is not None)
                color = (0, 210, 100) if is_winner else (130, 130, 130)
                cv2.putText(panel, f"  {rank+1}. {label}: {conf:.3f}",
                            (4, y + 12), font, 0.37, color, 1)
                y += 15;  entry_h += 15

        # divider
        y += 4
        if y < frame_h:
            cv2.line(panel, (4, y), (_OCR_PANEL_W - 4, y), (55, 55, 55), 1)
        y += 5

        if y >= frame_h - 30:
            break

    if not crops:
        cv2.putText(panel, "No crops this frame", (4, 20), font, 0.4, (100, 100, 100), 1)

    return panel


class JerseyOCR:
    def __init__(self, model_path: str):
        self.model  = YOLO(model_path)
        self.device = _best_device()

    def predict(self, crops: list) -> list[tuple]:
        """
        Returns one 3-tuple per crop: (top1, top5, blur_var)
          top1     = (label, conf) if conf >= _OCR_CONF_THRESHOLD, else None
          top5     = [(label, conf), ...] — always 5 entries when not blurry
          blur_var = Laplacian variance of the crop
        When a crop fails the blur gate, top1=None and top5=[].
        """
        if not crops:
            return []
        results = self.model(crops, device=self.device, verbose=False)
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


class PoseEstimator:
    def __init__(self, model_path: str):
        self.model  = YOLO(model_path)
        self.device = _best_device()

    def get_keypoints_for_boxes(self, frame, boxes) -> list:
        """Match pose detections to tracked player boxes by IoU. Returns list[kps | None]."""
        if not boxes:
            return []
        results = self.model(frame, device=self.device, verbose=False)
        r = results[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return [None] * len(boxes)
        pose_xyxy = r.boxes.xyxy.cpu().numpy()
        kps_data  = r.keypoints.data.cpu().numpy()
        out = []
        for box in boxes:
            best_iou, best_kps = 0.0, None
            for j, pbox in enumerate(pose_xyxy):
                iou = _box_iou(box[:4], pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_kps = kps_data[j]
            out.append(best_kps if best_iou > 0.3 else None)
        return out


def run_pipeline(video_path, det_model_path, ocr_model_path, ball_model_path=None,
                 pose_model_path="weights/yolo11n-pose.pt",
                 output_path=None, conf=0.4, ocr_warning=False, pose_every=3):
    _NUMBER_HISTORY.clear()
    _NUMBER_LOCKED.clear()

    device        = _best_device()
    detector      = YOLO(det_model_path)
    classifier    = TeamClassifier()
    ball_detector = BallDetector(ball_model_path) if ball_model_path else None
    pose_est      = PoseEstimator(pose_model_path) if pose_model_path else None
    jersey_reader = JerseyOCR(ocr_model_path)

    fitted = False

    cap    = cv2.VideoCapture(video_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    window = "Field State"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_num       = 0
    cached_pose_kps = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = detector.track(
            frame, persist=True, conf=conf, verbose=False,
            device=device, half=True,
        )

        boxes = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            ids  = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, tid, cls in zip(xyxy, ids, clss):
                if cls == 0:  # players only
                    boxes.append([*box, tid, cls])

        if boxes and not fitted:
            fitted = classifier.fit(frame, boxes)

        team_labels = classifier.classify(frame, boxes) if fitted and boxes else ["unknown"] * len(boxes)

        ball_xy = ball_detector.detect(frame) if ball_detector else None
        if fitted and boxes:
            flipped = classifier.update_offense_from_ball(ball_xy, boxes, team_labels)
            if flipped:
                team_labels = classifier.classify(frame, boxes)

        # ── Pose estimation (every pose_every frames) ────────────────────
        if frame_num % pose_every == 0 or cached_pose_kps is None:
            cached_pose_kps = pose_est.get_keypoints_for_boxes(frame, boxes) if pose_est else None
        pose_kps = cached_pose_kps

        crops, valid_idx, _ = _crops(frame, boxes, pose_kps)

        # Orientation gate: skip side-facing players
        gated_crops, gated_box_idx = [], []
        for j, crop in enumerate(crops):
            box_i = valid_idx[j]
            kps = pose_kps[box_i] if pose_kps and box_i < len(pose_kps) else None
            if kps is not None and not _is_forward_facing(kps, boxes[box_i]):
                continue
            gated_crops.append(crop)
            gated_box_idx.append(box_i)

        # ── Jersey number reading ────────────────────────────────────────
        ocr_results = jersey_reader.predict(gated_crops)
        ocr_number_map: dict[int, tuple[str, float]] = {}
        for j, (top1, top5, blur_var) in enumerate(ocr_results):
            if top1 is not None:
                ocr_number_map[gated_box_idx[j]] = top1

        field_state = []
        for i, (box, team) in enumerate(zip(boxes, team_labels)):
            track_id = int(box[4])
            raw = ocr_number_map.get(i)
            number = _stable_number(track_id, *raw) if raw is not None else "?"
            field_state.append({
                "track_id": track_id,
                "bbox":     [int(v) for v in box[:4]],
                "team":     team,
                "number":   number,
            })

            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            color = COLORS[team]
            locked = "L" if track_id in _NUMBER_LOCKED else ""
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _put_text(frame, f"{team[:3].upper()} #{number}{locked}",
                      (x1, max(y1 - 6, 12)), scale=0.55, color=color, thickness=2)

        # ── Field mapping: project players to top-down field coords ─────────
        detections_for_mapper = [
            {"track_id": p["track_id"], "bbox": p["bbox"], "class": 0}
            for p in field_state
        ]
        field_points = project_players(
            frame_num=frame_num - 1,   # frame_num is 1-indexed; npz is 0-indexed
            detections=detections_for_mapper,
            homography_path="homographies.npz",
        )
        # Add field coords back to field_state
        fp_by_id = {p["track_id"]: p for p in field_points}
        for p in field_state:
            fp = fp_by_id.get(p["track_id"])
            p["field_x"] = fp["field_x"] if fp else None
            p["field_y"] = fp["field_y"] if fp else None

        # ── OCR debug panel ───────────────────────────────────────────────────
        debug_tids  = [int(boxes[i][4]) for i in gated_box_idx]
        debug_panel = _build_ocr_panel(gated_crops, ocr_results, debug_tids, frame.shape[0])
        combined    = np.hstack([frame, debug_panel])

        if ball_xy:
            bx, by = int(ball_xy[0]), int(ball_xy[1])
            cv2.circle(combined, (bx, by), 8, (0, 255, 0), 2)
            cv2.putText(combined, "ball", (bx + 10, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.putText(combined, f"Frame {frame_num}/{total}  players={len(field_state)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if ocr_warning:
            cv2.putText(combined, "WARNING: jersey # accuracy is low (experimental)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video> [--det PATH] [--ocr PATH] [--out FILE] [--conf N] [--pose-every N]")
        sys.exit(1)

    video       = sys.argv[1]
    det_model   = "weights/player-best.pt"
    ocr_model   = "weights/best.pt"
    ball_model  = "weights/ball-best.pt"
    pose_model  = "weights/yolo11n-pose.pt"
    out_path    = None
    conf        = 0.4
    ocr_warning = False
    pose_every  = 3

    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if   a == "--det"        and i + 1 < len(sys.argv): det_model   = sys.argv[i+1]; i += 2
        elif a == "--ocr"        and i + 1 < len(sys.argv): ocr_model   = sys.argv[i+1]; i += 2
        elif a == "--ball"       and i + 1 < len(sys.argv): ball_model  = sys.argv[i+1]; i += 2
        elif a == "--no-ball":                               ball_model  = None;          i += 1
        elif a == "--pose"       and i + 1 < len(sys.argv): pose_model  = sys.argv[i+1]; i += 2
        elif a == "--no-pose":                               pose_model  = None;          i += 1
        elif a == "--out"        and i + 1 < len(sys.argv): out_path    = sys.argv[i+1]; i += 2
        elif a == "--conf"       and i + 1 < len(sys.argv): conf        = float(sys.argv[i+1]); i += 2
        elif a == "--ocr-warning":                           ocr_warning = True;          i += 1
        elif a == "--pose-every" and i + 1 < len(sys.argv): pose_every  = int(sys.argv[i+1]); i += 2
        else: i += 1

    run_pipeline(video, det_model, ocr_model,
                 ball_model_path=ball_model,
                 pose_model_path=pose_model,
                 output_path=out_path,
                 conf=conf,
                 ocr_warning=ocr_warning,
                 pose_every=pose_every)
