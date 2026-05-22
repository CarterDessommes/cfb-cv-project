"""
BoT-SORT tracker with local YOLO model.
Fast - no API calls, runs entirely on your machine.
"""

from botsort import BoTSORT
from ultralytics import YOLO
import numpy as np
import cv2
import sys
import os


def create_tracker():
    """Initialize BoT-SORT tracker."""
    return BoTSORT(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.7,
        use_cmc=True,
    )


def load_model(model_path):
    """Load YOLO model."""
    print(f"Loading model: {model_path}")
    return YOLO(model_path)


def run_detection(model, frame, confidence=0.4):
    """Run YOLO detection. Returns (N, 6) array [x1, y1, x2, y2, conf, cls]."""
    results = model(frame, verbose=False, conf=confidence)

    dets = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                dets.append([x1, y1, x2, y2, conf, cls])

    return np.array(dets) if dets else np.empty((0, 6))


def track_video(video_path, model_path, output_path=None, confidence=0.4, show=True):
    """Run BoT-SORT tracking on video."""

    model = load_model(model_path)
    tracker = create_tracker()

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    colors = {}
    frame_num = 0
    window_name = "BoT-SORT Tracking"

    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Processing {total_frames} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Run detection
        dets = run_detection(model, frame, confidence)

        # Update tracker
        tracks = tracker.update(dets, frame) if len(dets) > 0 else np.empty((0, 5))

        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5].astype(int)

            if track_id not in colors:
                colors[track_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                )
            color = colors[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Frame: {frame_num}/{total_frames} | Tracks: {len(tracks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow(window_name, frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        if frame_num % 10 == 0:
            print(f"\rFrame {frame_num}/{total_frames}: {len(tracks)} tracks", end="")

    print(f"\nDone! Processed {frame_num} frames.")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tracker.py <video_path> [options]")
        print()
        print("Options:")
        print("  --model PATH   Path to YOLO weights (default: weights/best.pt)")
        print("  --out FILE     Save output video")
        print("  --conf N       Detection confidence 0.0-1.0 (default: 0.4)")
        print()
        print("Examples:")
        print('  python tracker.py "test media/videos/bijon_run.mp4"')
        print('  python tracker.py "test media/videos/bijon_run.mp4" --out tracked.mp4')
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = "weights/best.pt"
    output_path = None
    confidence = 0.4

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--out" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--conf" and i + 1 < len(sys.argv):
            confidence = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    track_video(video_path, model_path, output_path, confidence)
