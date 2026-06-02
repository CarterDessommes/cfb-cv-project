"""
BoT-SORT tracker with local YOLO model and Re-ID support.
Uses Ultralytics built-in BoT-SORT with appearance embeddings for
consistent track IDs across occlusions.
"""

import numpy as np
import cv2
import sys
import os

from ball_tracker import BallTracker
from yolo_utils import boxes_to_cpu_arrays
from multiscale import merge_detections, MultiScaleTracker

# Constants
DEFAULT_BALL_MODEL = "weights/ball-best.pt"
DEFAULT_BALL_EVERY = 2
DEFAULT_BALL_ROI = 320
DEFAULT_BALL_FULL_EVERY = 30


def load_model(model_path):
    """Load YOLO model."""
    from ultralytics import YOLO

    print(f"Loading model: {model_path}")
    return YOLO(model_path)


def get_tracker_config_path(ball=False, multiscale=False):
    """Get path to a BoT-SORT config file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if multiscale:
        config_name = "botsort_multiscale.yaml"
    elif ball:
        config_name = "botsort_ball.yaml"
    else:
        config_name = "botsort.yaml"
    return os.path.join(script_dir, config_name)


def _due(frame_num: int, every: int) -> bool:
    """True on frames where a throttled model should run (frames 1, 1+every, ...)."""
    return every <= 1 or frame_num % every == 1


def track_video(video_path, model_path, output_path=None, confidence=0.4, show=True,
                ball_model_path=None, ball_confidence=0.3, track_ball=True,
                ball_every=DEFAULT_BALL_EVERY, ball_roi=DEFAULT_BALL_ROI,
                ball_full_every=DEFAULT_BALL_FULL_EVERY,
                imgsz=480, imgsz2=960, nms_iou=0.6):
    """Run BoT-SORT tracking on video with Re-ID support and optional ball tracking."""

    # Load player/referee model
    model = load_model(model_path)
    tracker_config = get_tracker_config_path()
    print(f"Using player tracker config: {tracker_config}")

    # Opt-in dual-resolution detection: merge a second higher-res pass via NMS,
    # then drive a manually-managed BoT-SORT (Re-ID off) for stable track IDs.
    mst = MultiScaleTracker(get_tracker_config_path(multiscale=True)) if imgsz2 else None
    if mst is not None:
        print(f"Dual-resolution detection: imgsz={imgsz}+{imgsz2}, nms_iou={nms_iou}")

    # Load ROI-based ball tracker if enabled
    ball_tracker = None
    if track_ball and ball_model_path:
        if os.path.exists(ball_model_path):
            print(f"Loading ball tracker: {ball_model_path}")
            ball_tracker = BallTracker(
                ball_model_path,
                ball_conf=ball_confidence,
                roi_size=ball_roi,
                full_every=ball_full_every,
            )
        else:
            print(f"Warning: Ball model not found at {ball_model_path}, disabling ball tracking")
            track_ball = False

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width > 0 and height > 0:
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        model(dummy_frame, device='mps', half=True, verbose=False)
        if ball_tracker is not None:
            ball_tracker.warmup(dummy_frame.shape)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    colors = {}
    frame_num = 0
    detection_num = 0
    tracks = []  # Persist player/referee tracks across skipped frames
    ball_xy: tuple[float, float] | None = None  # Persist ball center across skipped frames
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
            detection_num += 1
            if mst is not None:
                # Dual-resolution: merge two passes via NMS, then track manually.
                merged = merge_detections(
                    model, frame, confidence, 'mps', [imgsz, imgsz2], nms_iou
                )
                trk = mst.update(merged, frame)
                boxes = trk[:, :4] if len(trk) else None
                track_ids = trk[:, 4].astype(int) if len(trk) else None
                classes = trk[:, 6].astype(int) if len(trk) else None
                confs = trk[:, 5] if len(trk) else None
            else:
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
                boxes, track_ids, classes, confs = boxes_to_cpu_arrays(results[0].boxes)

            # Extract tracks
            tracks = []
            if boxes is not None and track_ids is not None and classes is not None and confs is not None:
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

            # Run ROI-based ball tracking if enabled
            if ball_tracker is not None and _due(detection_num, ball_every):
                ball_xy = ball_tracker.update(frame)
        # On skipped frames, reuse previous tracks

        # Count players and referees
        player_count = sum(1 for t in tracks if int(t[5]) == 0)
        referee_count = sum(1 for t in tracks if int(t[5]) == 1)
        ball_detected = ball_xy is not None

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

        # Draw ball center as a bright yellow circle
        if ball_xy is not None:
            center_x, center_y = int(ball_xy[0]), int(ball_xy[1])
            radius = 10
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
        print("  --model PATH       Path to player YOLO weights (default: weights/player-best.pt)")
        print("  --out FILE         Save output video")
        print("  --conf N           Player detection confidence 0.0-1.0 (default: 0.4)")
        print("  --imgsz N          Player detection resolution (default: 480)")
        print("  --imgsz2 N         Second resolution for dual-res NMS merge (default: 960, 0=off)")
        print("  --nms-iou F        IoU threshold for the dual-res merge (default: 0.6)")
        print()
        print("Ball tracking options:")
        print("  --ball-model PATH  Path to ball YOLO weights (default: weights/ball-best.pt)")
        print("  --ball-conf N      Ball detection confidence 0.0-1.0 (default: 0.3)")
        print("  --ball-every N     Run ball detector every N tracker detections (default: 2)")
        print("  --ball-roi N       Ball ROI square size in pixels (default: 320)")
        print("  --ball-full-every N  Full-frame ball re-acquire interval (default: 30)")
        print("  --no-ball          Disable ball tracking")
        print()
        print("Examples:")
        print('  python tracker.py "test media/videos/bijon_run.mp4"')
        print('  python tracker.py "test media/videos/bijon_run.mp4" --out tracked.mp4')
        print('  python tracker.py "test media/videos/bijon_run.mp4" --no-ball')
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = "weights/player-best.pt"
    output_path = None
    confidence = 0.4
    ball_model_path = DEFAULT_BALL_MODEL
    ball_confidence = 0.3
    ball_every = DEFAULT_BALL_EVERY
    ball_roi = DEFAULT_BALL_ROI
    ball_full_every = DEFAULT_BALL_FULL_EVERY
    track_ball = True
    imgsz = 480
    imgsz2 = 960
    nms_iou = 0.6

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
        elif sys.argv[i] == "--ball-every" and i + 1 < len(sys.argv):
            ball_every = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--ball-roi" and i + 1 < len(sys.argv):
            ball_roi = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--ball-full-every" and i + 1 < len(sys.argv):
            ball_full_every = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--imgsz" and i + 1 < len(sys.argv):
            imgsz = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--imgsz2" and i + 1 < len(sys.argv):
            imgsz2 = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--nms-iou" and i + 1 < len(sys.argv):
            nms_iou = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no-ball":
            track_ball = False
            i += 1
        else:
            i += 1

    track_video(video_path, model_path, output_path, confidence,
                ball_model_path=ball_model_path, ball_confidence=ball_confidence,
                track_ball=track_ball, ball_every=ball_every, ball_roi=ball_roi,
                ball_full_every=ball_full_every,
                imgsz=imgsz, imgsz2=imgsz2, nms_iou=nms_iou)
