"""
Full field-state pipeline: player location + team + jersey number.

Usage:
    python pipeline.py <video> [--det PATH] [--ocr PATH] [--out FILE] [--conf N]

Defaults:
    --det  weights/best.pt
    --ocr  weights/jersey_ocr.pt
    --conf 0.4
"""

import sys
import cv2
from ultralytics import YOLO

from team_classifier import TeamClassifier, _best_device


COLORS = {
    "team_a":  (0, 200, 255),
    "team_b":  (255, 100,   0),
    "unknown": (128, 128, 128),
}

_MIN_CROP_PX = 10


def _crops(frame, boxes):
    crops, indices = [], []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] >= _MIN_CROP_PX and crop.shape[1] >= _MIN_CROP_PX:
            crops.append(crop)
            indices.append(i)
    return crops, indices


class JerseyOCR:
    def __init__(self, model_path: str):
        self.model  = YOLO(model_path)
        self.device = _best_device()

    def predict(self, crops: list) -> list[str]:
        if not crops:
            return []
        results = self.model(crops, device=self.device, verbose=False)
        return [r.names[r.probs.top1] for r in results]


def run_pipeline(video_path, det_model_path, ocr_model_path, output_path=None, conf=0.4):
    detector   = YOLO(det_model_path)
    classifier = TeamClassifier()
    ocr        = JerseyOCR(ocr_model_path)
    fitted     = False

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

        crops, valid_idx = _crops(frame, boxes)
        ocr_preds  = ocr.predict(crops)
        number_map = {valid_idx[j]: ocr_preds[j] for j in range(len(ocr_preds))}

        field_state = []
        for i, (box, team) in enumerate(zip(boxes, team_labels)):
            number = number_map.get(i, "?")
            field_state.append({
                "track_id": int(box[4]),
                "bbox":     [int(v) for v in box[:4]],
                "team":     team,
                "number":   number,
            })

            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            color = COLORS[team]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{team[5:].upper()} #{number}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Frame {frame_num}/{total}  players={len(field_state)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

        if frame_num % 10 == 0:
            print(f"\rFrame {frame_num}/{total}: {len(field_state)} players", end="")

    print(f"\nDone. Processed {frame_num} frames.")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video> [--det PATH] [--ocr PATH] [--out FILE] [--conf N]")
        sys.exit(1)

    video     = sys.argv[1]
    det_model = "weights/best.pt"
    ocr_model = "weights/jersey_ocr.pt"
    out_path  = None
    conf      = 0.4

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--det" and i + 1 < len(sys.argv):
            det_model = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--ocr" and i + 1 < len(sys.argv):
            ocr_model = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--out" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--conf" and i + 1 < len(sys.argv):
            conf = float(sys.argv[i + 1]); i += 2
        else:
            i += 1

    run_pipeline(video, det_model, ocr_model, out_path, conf)
