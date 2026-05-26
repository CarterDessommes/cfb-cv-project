"""
Full field-state pipeline: player location + team + jersey number.

Usage:
    python pipeline.py <video> [--det PATH] [--ocr PATH] [--ball PATH] [--out FILE] [--conf N] [--no-ball]

Defaults:
    --det  weights/player-best.pt
    --ocr  weights/jersey_ocr.pt
    --ball weights/ball-best.pt
    --conf 0.4
"""

import sys
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

from team_classifier import TeamClassifier, _best_device
from field_mapper import project_players, build_field_canvas, CANVAS_SCALE

_NUMBER_HISTORY: dict[int, list[str]] = {}
_VOTE_WINDOW = 15


def _stable_number(track_id: int, prediction: str) -> str:
    history = _NUMBER_HISTORY.setdefault(track_id, [])
    history.append(prediction)
    if len(history) > _VOTE_WINDOW:
        history.pop(0)
    return Counter(history).most_common(1)[0][0]


COLORS = {
    "offense": (0, 200, 255),
    "defense": (255, 100,   0),
    "unknown": (128, 128, 128),
}

_MIN_CROP_PX = 10


def _crops(frame, boxes):
    crops, indices = [], []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Slice the torso band (10%-50% of height) where the number lives
        h = y2 - y1
        ty1 = y1 + int(h * 0.10)
        ty2 = y1 + int(h * 0.50)
        crop = frame[ty1:ty2, x1:x2]
        if crop.shape[0] >= _MIN_CROP_PX and crop.shape[1] >= _MIN_CROP_PX:
            # Grayscale → 3-channel so YOLO classifier still gets RGB input shape
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            crops.append(crop)
            indices.append(i)
    return crops, indices


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

    def predict(self, crops: list) -> list[str]:
        if not crops:
            return []
        results = self.model(crops, device=self.device, verbose=False)
        return [r.names[r.probs.top1] for r in results]


def run_pipeline(video_path, det_model_path, ocr_model_path, ball_model_path=None,
                 output_path=None, conf=0.4, ocr_warning=False):
    _NUMBER_HISTORY.clear()
    detector      = YOLO(det_model_path)
    classifier    = TeamClassifier()
    ocr           = JerseyOCR(ocr_model_path)
    ball_detector = BallDetector(ball_model_path) if ball_model_path else None
    fitted        = False

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

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = detector.track(
            frame, persist=True, conf=conf, verbose=False,
            device=_best_device(), half=True,
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
            classifier.update_offense_from_ball(ball_xy, boxes, team_labels)
            # Re-classify after a potential label flip so this frame reflects the update
            team_labels = classifier.classify(frame, boxes)

        crops, valid_idx = _crops(frame, boxes)
        ocr_preds  = ocr.predict(crops)
        number_map = {valid_idx[j]: ocr_preds[j] for j in range(len(ocr_preds))}

        field_state = []
        for i, (box, team) in enumerate(zip(boxes, team_labels)):
            track_id = int(box[4])
            raw      = number_map.get(i, "?")
            number   = _stable_number(track_id, raw) if raw != "?" else "?"
            field_state.append({
                "track_id": track_id,
                "bbox":     [int(v) for v in box[:4]],
                "team":     team,
                "number":   number,
            })

            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            color = COLORS[team]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{team.upper()} #{number}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
        print("Usage: python pipeline.py <video> [--det PATH] [--ocr PATH] [--out FILE] [--conf N]")
        sys.exit(1)

    video       = sys.argv[1]
    det_model   = "weights/player-best.pt"
    ocr_model   = "weights/jersey_ocr.pt"
    ball_model  = "weights/ball-best.pt"
    out_path    = None
    conf        = 0.4
    ocr_warning = False

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
        elif sys.argv[i] == "--out" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--conf" and i + 1 < len(sys.argv):
            conf = float(sys.argv[i + 1]); i += 2
        elif sys.argv[i] == "--ocr-warning":
            ocr_warning = True; i += 1
        else:
            i += 1

    run_pipeline(video, det_model, ocr_model, ball_model, out_path, conf, ocr_warning)
