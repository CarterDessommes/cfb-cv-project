"""
BoT-SORT tracker with local YOLO model and Re-ID support.
Uses Ultralytics built-in BoT-SORT with appearance embeddings for
consistent track IDs across occlusions.
"""

from ultralytics import YOLO
import numpy as np
import cv2
import sys
import os


def load_model(model_path):
    """Load YOLO model."""
    print(f"Loading model: {model_path}")
    return YOLO(model_path)


def get_tracker_config_path():
    """Get path to botsort.yaml config file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "botsort.yaml")


def track_video(video_path, model_path, output_path=None, confidence=0.4, show=True):
    """Run BoT-SORT tracking on video with Re-ID support."""

    model = load_model(model_path)
    tracker_config = get_tracker_config_path()
    print(f"Using tracker config: {tracker_config}")

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
    tracks = []  # Persist tracks across skipped frames
    window_name = "BoT-SORT Tracking (Re-ID)"

    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Processing {total_frames} frames (every other frame)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Only run detection on every other frame
        if frame_num % 2 == 1:
            # Run detection + tracking with Re-ID using Ultralytics built-in BoT-SORT
            # persist=True maintains track state across frames
            results = model.track(
                frame,
                tracker=tracker_config,
                persist=True,
                conf=confidence,
                verbose=False,
                device='mps',    # Use Apple Silicon GPU
                half=True,       # FP16 inference (2x faster)
            )

            # Extract tracks from results
            tracks = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                # Separate by class and keep top N by confidence
                # Class 0 = player (max 22), Class 1 = referee (max 7)
                max_per_class = {0: 22, 1: 7}

                class_detections = {}
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                    if cls not in class_detections:
                        class_detections[cls] = []
                    class_detections[cls].append((conf, [*box, track_id, cls]))

                # Sort by confidence and keep top N for each class
                for cls, detections in class_detections.items():
                    detections.sort(key=lambda x: x[0], reverse=True)
                    max_count = max_per_class.get(cls, 50)  # default 50 for unknown classes
                    for conf, track_data in detections[:max_count]:
                        tracks.append(track_data)
        # On skipped frames, reuse previous tracks

        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2, track_id, cls = [int(v) for v in track]

            if track_id not in colors:
                colors[track_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                )
            color = colors[track_id]

            label = "R" if cls == 1 else "P"  # R=referee, P=player
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}{track_id}", (x1, y1 - 10),
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
