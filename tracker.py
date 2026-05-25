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

# Constants
BALL_TRACK_ID_OFFSET = 1000
DEFAULT_BALL_MODEL = "weights/ball-best.pt"


def load_model(model_path):
    """Load YOLO model."""
    print(f"Loading model: {model_path}")
    return YOLO(model_path)


def get_tracker_config_path(ball=False):
    """Get path to botsort.yaml config file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_name = "botsort_ball.yaml" if ball else "botsort.yaml"
    return os.path.join(script_dir, config_name)


def track_video(video_path, model_path, output_path=None, confidence=0.4, show=True,
                ball_model_path=None, ball_confidence=0.3, track_ball=True):
    """Run BoT-SORT tracking on video with Re-ID support and optional ball tracking."""

    # Load player/referee model
    model = load_model(model_path)
    tracker_config = get_tracker_config_path()
    print(f"Using player tracker config: {tracker_config}")

    # Load ball model if enabled
    ball_model = None
    ball_tracker_config = None
    if track_ball and ball_model_path:
        if os.path.exists(ball_model_path):
            ball_model = load_model(ball_model_path)
            ball_tracker_config = get_tracker_config_path(ball=True)
            print(f"Using ball tracker config: {ball_tracker_config}")
        else:
            print(f"Warning: Ball model not found at {ball_model_path}, disabling ball tracking")
            track_ball = False

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
    tracks = []  # Persist player/referee tracks across skipped frames
    ball_tracks = []  # Persist ball tracks across skipped frames
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

            # Run ball model tracking if enabled
            ball_tracks = []
            if ball_model is not None:
                ball_results = ball_model.track(
                    frame,
                    tracker=ball_tracker_config,
                    persist=True,
                    conf=ball_confidence,
                    verbose=False,
                    device='mps',
                    half=True,
                )

                # Debug: check what we got
                boxes_obj = ball_results[0].boxes
                if boxes_obj is not None and len(boxes_obj) > 0:
                    num_detections = len(boxes_obj)
                    has_ids = boxes_obj.id is not None
                    confs_debug = boxes_obj.conf.cpu().numpy()
                    if frame_num <= 10 or frame_num % 50 == 0:
                        print(f"\n[DEBUG] Frame {frame_num}: {num_detections} ball detections, confs={confs_debug.round(3)}, has_ids={has_ids}")

                    if has_ids:
                        boxes = boxes_obj.xyxy.cpu().numpy()
                        track_ids = boxes_obj.id.cpu().numpy().astype(int)
                        confs = boxes_obj.conf.cpu().numpy()

                        # Get all ball detections sorted by confidence
                        ball_detections = []
                        for box, track_id, conf in zip(boxes, track_ids, confs):
                            # Offset track ID to avoid collision with player IDs
                            offset_id = track_id + BALL_TRACK_ID_OFFSET
                            ball_detections.append((conf, [*box, offset_id, 2]))  # cls=2 for ball

                        # Keep only the highest confidence ball (max 1)
                        if ball_detections:
                            ball_detections.sort(key=lambda x: x[0], reverse=True)
                            ball_tracks.append(ball_detections[0][1])
                    else:
                        # No track IDs yet - use raw detections without tracking
                        boxes = boxes_obj.xyxy.cpu().numpy()
                        confs = boxes_obj.conf.cpu().numpy()

                        # Get highest confidence detection
                        best_idx = confs.argmax()
                        box = boxes[best_idx]
                        ball_tracks.append([*box, BALL_TRACK_ID_OFFSET, 2])
                elif frame_num <= 10 or frame_num % 50 == 0:
                    print(f"\n[DEBUG] Frame {frame_num}: NO ball detections")
        # On skipped frames, reuse previous tracks

        # Count players and referees
        player_count = sum(1 for t in tracks if int(t[5]) == 0)
        referee_count = sum(1 for t in tracks if int(t[5]) == 1)
        ball_detected = len(ball_tracks) > 0

        # Draw player/referee tracks
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

        # Draw ball tracks as bright yellow circles
        for track in ball_tracks:
            x1, y1, x2, y2, track_id, cls = [int(v) for v in track]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max((x2 - x1) // 2, (y2 - y1) // 2, 10)

            # Bright yellow color for ball
            ball_color = (0, 255, 255)  # BGR: yellow
            cv2.circle(frame, (center_x, center_y), radius, ball_color, 3)
            cv2.putText(frame, "BALL", (center_x - 20, center_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)

        # Status display with player/referee/ball counts
        ball_status = "YES" if ball_detected else "NO"
        status_text = f"Frame: {frame_num}/{total_frames} | P:{player_count} R:{referee_count} Ball:{ball_status}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
            ball_str = " +ball" if ball_detected else ""
            print(f"\rFrame {frame_num}/{total_frames}: {len(tracks)} tracks{ball_str}", end="")

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
        print("  --model PATH       Path to player YOLO weights (default: weights/best.pt)")
        print("  --out FILE         Save output video")
        print("  --conf N           Player detection confidence 0.0-1.0 (default: 0.4)")
        print()
        print("Ball tracking options:")
        print("  --ball-model PATH  Path to ball YOLO weights (default: weights/ball-best.pt)")
        print("  --ball-conf N      Ball detection confidence 0.0-1.0 (default: 0.3)")
        print("  --no-ball          Disable ball tracking")
        print()
        print("Examples:")
        print('  python tracker.py "test media/videos/bijon_run.mp4"')
        print('  python tracker.py "test media/videos/bijon_run.mp4" --out tracked.mp4')
        print('  python tracker.py "test media/videos/bijon_run.mp4" --no-ball')
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = "weights/best.pt"
    output_path = None
    confidence = 0.4
    ball_model_path = DEFAULT_BALL_MODEL
    ball_confidence = 0.3
    track_ball = True

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
        elif sys.argv[i] == "--ball-model" and i + 1 < len(sys.argv):
            ball_model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ball-conf" and i + 1 < len(sys.argv):
            ball_confidence = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no-ball":
            track_ball = False
            i += 1
        else:
            i += 1

    track_video(video_path, model_path, output_path, confidence,
                ball_model_path=ball_model_path, ball_confidence=ball_confidence,
                track_ball=track_ball)
