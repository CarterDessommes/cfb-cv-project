"""
Full field-state pipeline: player location + team + jersey number.

Usage:
    python pipeline.py <video> [--det PATH] [--ocr PATH] [--ball PATH] [--out FILE] [--conf N] [--no-ball]
    python pipeline.py <video> --vlm [--vlm-every N] [--pose-every N] ...

Defaults:
    --det        weights/player-best.pt
    --ocr        weights/jersey_ocr.pt   (ignored when --vlm is set)
    --ball       weights/ball-best.pt
    --conf       0.4
    --pose-every 3    (run pose every N frames)
    --vlm-every  30   (query VLM per track every N frames, VLM mode only)
    --vlm-model  ID   (HuggingFace model ID, default Qwen/Qwen2.5-VL-3B-Instruct)
"""

import re
import sys
import os
import cv2
import shutil
import tempfile
import numpy as np
from collections import Counter
from ultralytics import YOLO

from team_classifier import TeamClassifier, _best_device
from field_mapper import project_players, build_field_canvas, CANVAS_SCALE

_NUMBER_HISTORY: dict[int, list[tuple[str, float]]] = {}
_NUMBER_LOCKED:  dict[int, str] = {}
_VOTE_WINDOW = 75  # ~2.5 s at 30 fps
_LOCK_VOTES  = 8   # confident reads needed to permanently lock a tracklet's number

# VLM-mode state
# Each entry is a list of predictions (str digits or None = unreadable) in order.
# None entries count against confidence so consistently-occluded players show "?".
_VLM_READS:      dict[int, list[str | None]] = {}
_VLM_CONFIRM     = 3    # consecutive identical non-None reads required to lock
_VLM_WINDOW      = 10   # rolling window size for confidence calculation
_VLM_MIN_READS   = 2    # minimum reads before displaying anything
_VLM_CONF_RATIO  = 0.5  # top prediction must appear in >= this fraction of the window

_VLM_BATCH_PROMPT = (
    "Each image above shows a football player. "
    "What is the jersey number printed on each player's chest? "
    "Note: '0' and '00' are valid football jersey numbers — do not treat them as empty or missing. "
    "Reply with exactly {n} lines — one per image in order — containing ONLY the jersey number digits "
    "(e.g. '0', '42', or '7'), or the letter X if no number is clearly visible. No other text."
)


def _parse_vlm_batch(response: str, n: int) -> list[str | None]:
    """Parse a structured VLM response (one answer per line) into n results."""
    results: list[str | None] = []
    for raw in response.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip list numbering only when the separator is followed by more content
        # (e.g. "1. 42", "2: 7"). A bare "0", "0.", or "42." is the answer itself.
        m = re.match(r"^\d+[.:\)]\s+(.+)$", line)
        if m:
            line = m.group(1).strip()
        digits = "".join(c for c in line if c.isdigit())
        results.append(digits if digits else None)
        if len(results) >= n:
            break
    results += [None] * max(0, n - len(results))
    return results[:n]


def _vlm_record(track_id: int, pred: str | None) -> None:
    """Record one VLM read (None = unreadable) and lock the track if confident."""
    if track_id in _NUMBER_LOCKED:
        return
    reads = _VLM_READS.setdefault(track_id, [])
    reads.append(pred)
    tail = reads[-_VLM_CONFIRM:]
    if len(tail) == _VLM_CONFIRM and len(set(tail)) == 1 and tail[0] is not None:
        _NUMBER_LOCKED[track_id] = tail[0]


def _vlm_display(track_id: int) -> str:
    """
    Return the best current number for display, or '?' if not yet confident.
    Never modifies state — safe to call every frame.
    """
    if track_id in _NUMBER_LOCKED:
        return _NUMBER_LOCKED[track_id]
    window = _VLM_READS.get(track_id, [])[-_VLM_WINDOW:]
    if len(window) < _VLM_MIN_READS:
        return "?"
    non_none = [r for r in window if r is not None]
    if not non_none:
        return "?"
    top, count = Counter(non_none).most_common(1)[0]
    if count / len(window) < _VLM_CONF_RATIO:
        return "?"
    return top


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
_OCR_CONF_THRESHOLD = 0.6
_BLUR_THRESHOLD = 50
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


def _max_overlap(box, all_boxes, self_idx) -> float:
    """Max fraction of `box`'s area covered by any other box in all_boxes."""
    ax1, ay1, ax2, ay2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    best = 0.0
    for k, other in enumerate(all_boxes):
        if k == self_idx:
            continue
        bx1, by1, bx2, by2 = float(other[0]), float(other[1]), float(other[2]), float(other[3])
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        best = max(best, inter / a_area)
    return best


_OVERLAP_THRESHOLD = 0.15  # fall back to tight crop above this


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


def _crops(frame, boxes, pose_kps_list=None, full_box=False):
    crops, indices, rects = [], [], []
    fw = frame.shape[1]
    for i, box in enumerate(boxes):
        kps = pose_kps_list[i] if pose_kps_list is not None and i < len(pose_kps_list) else None

        crop, rect = None, None

        if full_box:
            overlap = _max_overlap(box[:4], [b[:4] for b in boxes], i)
            if overlap < _OVERLAP_THRESHOLD:
                # Isolated enough — give VLM the full player
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                crop = frame[y1:y2, x1:x2]
                rect = (x1, y1, x2, y2)
            # else: fall through to pose-guided crop below

        if crop is None:
            # Pose-guided crop (or heuristic fallback)
            if kps is not None:
                crop, rect = _pose_crop(frame, kps, box, fw)
            if crop is None:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                h      = y2 - y1
                ty1    = y1 + int(h * 0.15)
                ty2    = y1 + int(h * 0.45)
                crop_h = ty2 - ty1
                cx     = (x1 + x2) // 2
                sx1    = max(0, cx - crop_h // 2)
                sx2    = min(fw, sx1 + crop_h)
                crop   = frame[ty1:ty2, sx1:sx2]
                rect   = (sx1, ty1, sx2, ty2)

        if crop.shape[0] >= _MIN_CROP_PX and crop.shape[1] >= _MIN_CROP_PX:
            crops.append(crop)
            indices.append(i)
            rects.append(rect)
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


class JerseyOCR:
    def __init__(self, model_path: str):
        self.model  = YOLO(model_path)
        self.device = _best_device()

    def predict(self, crops: list) -> list[tuple[str, float] | None]:
        if not crops:
            return []
        results = self.model(crops, device=self.device, verbose=False)
        out = []
        for crop, r in zip(crops, results):
            if cv2.Laplacian(crop, cv2.CV_64F).var() < _BLUR_THRESHOLD:
                out.append(None)
                continue
            conf = float(r.probs.top1conf)
            out.append((r.names[r.probs.top1], conf) if conf >= _OCR_CONF_THRESHOLD else None)
        return out


class VLMJerseyReader:
    """
    Lazy-loaded Qwen2.5-VL reader for jersey numbers.

    All pending crops for a frame are sent in a SINGLE forward pass so the model
    is called at most once per frame regardless of how many players need querying.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self._model_id  = model_id
        self._model     = None
        self._processor = None
        self._pvi_fn    = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError:
            print("ERROR: VLM deps missing. Run:\n  pip install transformers accelerate qwen-vl-utils")
            sys.exit(1)
        print(f"Loading {self._model_id} …", flush=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_id, torch_dtype="auto", device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._pvi_fn    = process_vision_info
        print("VLM ready.", flush=True)

    def predict_batch(self, crops: list) -> list[str | None]:
        """
        Run VLM on a list of BGR crops in one forward pass.
        Returns digit strings (e.g. '42') or None for blurry / unreadable crops.
        """
        if not crops:
            return []

        # Pre-filter blurry crops — only pass sharp ones to the model
        sharp = [cv2.Laplacian(c, cv2.CV_64F).var() >= _BLUR_THRESHOLD for c in crops]
        sharp_crops = [c for c, ok in zip(crops, sharp) if ok]
        if not sharp_crops:
            return [None] * len(crops)

        self._load()

        n      = len(sharp_crops)
        tmpdir = tempfile.mkdtemp(prefix="vlm_jersey_")
        try:
            # Write all sharp crops to a temp dir, build a multi-image message
            content: list[dict] = []
            for i, crop in enumerate(sharp_crops):
                path = os.path.join(tmpdir, f"c{i}.png")
                cv2.imwrite(path, crop)
                content.append({"type": "image", "image": f"file://{os.path.abspath(path)}"})

            content.append({
                "type": "text",
                "text": _VLM_BATCH_PROMPT.format(n=n),
            })

            messages  = [{"role": "user", "content": content}]
            text      = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self._pvi_fn(messages)
            inputs = self._processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self._model.device)

            ids = self._model.generate(**inputs, max_new_tokens=n * 8)
            response = self._processor.batch_decode(
                ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        parsed = _parse_vlm_batch(response, n)

        # Re-expand to original crop count, inserting None for blurry slots
        out: list[str | None] = []
        it = iter(parsed)
        for ok in sharp:
            out.append(next(it) if ok else None)
        return out

    def predict(self, crop) -> str | None:
        """Single-crop convenience wrapper (used by label_crops.py)."""
        return self.predict_batch([crop])[0]


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
                 output_path=None, conf=0.4, ocr_warning=False,
                 use_vlm=False, pose_every=3, vlm_every=12,
                 vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    _NUMBER_HISTORY.clear()
    _NUMBER_LOCKED.clear()
    _VLM_READS.clear()

    device        = _best_device()
    detector      = YOLO(det_model_path)
    classifier    = TeamClassifier()
    ball_detector = BallDetector(ball_model_path) if ball_model_path else None
    pose_est      = PoseEstimator(pose_model_path) if pose_model_path else None

    if use_vlm:
        jersey_reader: VLMJerseyReader | JerseyOCR = VLMJerseyReader(vlm_model_id)
        _vlm_last_frame: dict[int, int] = {}
    else:
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

        crops, valid_idx, _ = _crops(frame, boxes, pose_kps, full_box=use_vlm)

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
        if use_vlm:
            # Collect all crops due for a VLM query this frame, then run ONE
            # batched forward pass instead of one per player.
            _VLM_MIN_PX = 80
            pending_crops: list = []
            pending_tids:  list[int] = []
            for crop, box_i in zip(gated_crops, gated_box_idx):
                tid = int(boxes[box_i][4])
                if tid in _NUMBER_LOCKED:
                    continue
                if frame_num - _vlm_last_frame.get(tid, -vlm_every) < vlm_every:
                    continue
                if crop.shape[0] < _VLM_MIN_PX or crop.shape[1] < _VLM_MIN_PX:
                    _vlm_last_frame[tid] = frame_num  # don't retry tiny crops immediately
                    continue
                pending_crops.append(crop)
                pending_tids.append(tid)

            if pending_crops:
                preds = jersey_reader.predict_batch(pending_crops)
                # Uniqueness filter: duplicate predictions in one batch = VLM confused
                pred_counts = Counter(p for p in preds if p is not None)
                for tid, pred in zip(pending_tids, preds):
                    _vlm_last_frame[tid] = frame_num
                    # Treat duplicate predictions as unreadable for this frame
                    clean_pred = pred if (pred is not None and pred_counts[pred] == 1) else None
                    _vlm_record(tid, clean_pred)
        else:
            # OCR path: batch inference (fast YOLO classifier)
            ocr_preds = jersey_reader.predict(gated_crops)
            ocr_number_map: dict[int, tuple[str, float]] = {}
            for j, pred in enumerate(ocr_preds):
                if pred is not None:
                    ocr_number_map[gated_box_idx[j]] = pred

        field_state = []
        for i, (box, team) in enumerate(zip(boxes, team_labels)):
            track_id = int(box[4])
            if use_vlm:
                number = _vlm_display(track_id)
            else:
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

        # ── Top-down canvas ───────────────────────────────────────────────────
        canvas = build_field_canvas(scale=CANVAS_SCALE)
        # Color dots by team
        team_color_map = {
            "offense": (0, 200, 255),
            "defense": (255, 100, 0),
            "unknown": (128, 128, 128),
        }
        for p, fp in zip(field_state, field_points):
            if not fp["in_bounds"]:
                continue
            cx = int(fp["field_x"] * CANVAS_SCALE)
            cy = int(fp["field_y"] * CANVAS_SCALE)
            color = team_color_map.get(p["team"], (128, 128, 128))
            cv2.circle(canvas, (cx, cy), 7, color, -1)
            cv2.circle(canvas, (cx, cy), 7, (255, 255, 255), 1)
            cv2.putText(canvas, str(p["track_id"]), (cx + 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

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
        print("Usage: python pipeline.py <video> [--det PATH] [--ocr PATH] [--out FILE] [--conf N] [--vlm] [--vlm-every N] [--pose-every N]")
        sys.exit(1)

    video         = sys.argv[1]
    det_model     = "weights/player-best.pt"
    ocr_model     = "weights/jersey_ocr.pt"
    ball_model    = "weights/ball-best.pt"
    pose_model    = "weights/yolo11n-pose.pt"
    out_path      = None
    conf          = 0.4
    ocr_warning   = False
    use_vlm       = False
    pose_every    = 3
    vlm_every     = 12
    vlm_model_id  = "Qwen/Qwen2.5-VL-3B-Instruct"

    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if   a == "--det"        and i + 1 < len(sys.argv): det_model    = sys.argv[i+1]; i += 2
        elif a == "--ocr"        and i + 1 < len(sys.argv): ocr_model    = sys.argv[i+1]; i += 2
        elif a == "--ball"       and i + 1 < len(sys.argv): ball_model   = sys.argv[i+1]; i += 2
        elif a == "--no-ball":                               ball_model   = None;          i += 1
        elif a == "--pose"       and i + 1 < len(sys.argv): pose_model   = sys.argv[i+1]; i += 2
        elif a == "--no-pose":                               pose_model   = None;          i += 1
        elif a == "--out"        and i + 1 < len(sys.argv): out_path     = sys.argv[i+1]; i += 2
        elif a == "--conf"       and i + 1 < len(sys.argv): conf         = float(sys.argv[i+1]); i += 2
        elif a == "--ocr-warning":                           ocr_warning  = True;          i += 1
        elif a == "--vlm":                                   use_vlm      = True;          i += 1
        elif a == "--pose-every" and i + 1 < len(sys.argv): pose_every   = int(sys.argv[i+1]); i += 2
        elif a == "--vlm-every"  and i + 1 < len(sys.argv): vlm_every    = int(sys.argv[i+1]); i += 2
        elif a == "--vlm-model"  and i + 1 < len(sys.argv): vlm_model_id = sys.argv[i+1]; i += 2
        else: i += 1

    run_pipeline(video, det_model, ocr_model,
                 ball_model_path=ball_model,
                 pose_model_path=pose_model,
                 output_path=out_path,
                 conf=conf,
                 ocr_warning=ocr_warning,
                 use_vlm=use_vlm,
                 pose_every=pose_every,
                 vlm_every=vlm_every,
                 vlm_model_id=vlm_model_id)
