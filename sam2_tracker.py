"""
SAM2-based player tracker.
Uses Segment Anything 2 for video object segmentation and tracking.
"""

from dotenv import load_dotenv
from roboflow import Roboflow
import numpy as np
import cv2
import sys
import os
import torch

load_dotenv()

# Lazy load SAM2 to avoid import errors if not installed
sam2_model = None
sam2_predictor = None


def load_sam2():
    """Load SAM2 model."""
    global sam2_model, sam2_predictor

    if sam2_predictor is not None:
        return sam2_predictor

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        print("SAM2 not installed. Install with:")
        print("  pip install sam2")
        print("  # Or clone from https://github.com/facebookresearch/sam2")
        sys.exit(1)

    # Use small model for speed (s=small, t=tiny, l=large, b+=base+)
    # Config path is relative to sam2 package
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    checkpoint = "sam2.1_hiera_small.pt"

    # Check if checkpoint exists, if not download
    if not os.path.exists(checkpoint):
        print(f"Downloading SAM2 checkpoint: {checkpoint}")
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{checkpoint}"
        result = os.system(f"curl -L -o {checkpoint} '{url}'")
        if result != 0:
            print("Failed to download checkpoint. Please download manually from:")
            print(f"  {url}")
            sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build predictor
    sam2_predictor = build_sam2_video_predictor(
        config_file=model_cfg,
        ckpt_path=checkpoint,
        device=device
    )
    return sam2_predictor


def get_initial_detections(frame_path, confidence=40):
    """Get player detections from Roboflow for initial frame."""
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    model = rf.workspace().project("nfl-detection-1500-eeuk7").version(1).model

    prediction = model.predict(frame_path, confidence=confidence)
    results = prediction.json()["predictions"]

    boxes = []
    for obj in results:
        x, y = obj["x"], obj["y"]
        w, h = obj["width"], obj["height"]
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        boxes.append([x1, y1, x2, y2])

    return np.array(boxes)


def extract_frames(video_path, output_dir):
    """Extract all frames from video to a directory."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"{frame_num:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_num += 1

    cap.release()
    return frame_paths


def track_video_sam2(video_path, output_path=None, confidence=40, show=True):
    """Track players using SAM2."""

    print("Extracting frames...")
    frame_dir = "/tmp/sam2_frames"
    frame_paths = extract_frames(video_path, frame_dir)
    print(f"Extracted {len(frame_paths)} frames")

    print("Loading SAM2...")
    predictor = load_sam2()

    print("Getting initial detections...")
    boxes = get_initial_detections(frame_paths[0], confidence)
    print(f"Found {len(boxes)} players")

    if len(boxes) == 0:
        print("No players detected in first frame!")
        return

    # Initialize SAM2 video predictor
    print("Initializing SAM2 tracking...")
    inference_state = predictor.init_state(video_path=frame_dir)

    # Add each detection as a tracked object
    colors = {}
    for obj_id, box in enumerate(boxes):
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=box,
        )
        colors[obj_id] = (
            np.random.randint(50, 255),
            np.random.randint(50, 255),
            np.random.randint(50, 255),
        )

    # Propagate through video
    print("Tracking through video...")
    video_segments = {}
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
        video_segments[frame_idx] = {
            "obj_ids": obj_ids,
            "masks": masks,
        }
        print(f"\rProcessed frame {frame_idx + 1}/{len(frame_paths)}", end="")

    print("\nRendering output...")

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    window_name = "SAM2 Tracking"
    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Render each frame with masks
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)

        if frame_idx in video_segments:
            obj_ids = video_segments[frame_idx]["obj_ids"]
            masks = video_segments[frame_idx]["masks"]

            for obj_id, mask in zip(obj_ids, masks):
                mask_np = mask.squeeze().cpu().numpy()

                # Create colored overlay
                color = colors.get(obj_id, (0, 255, 0))
                overlay = frame.copy()
                overlay[mask_np > 0.5] = color

                # Blend
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                # Draw contour
                mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)

                # Find center for label
                if len(contours) > 0:
                    M = cv2.moments(contours[0])
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(frame, f"ID:{obj_id}", (cx - 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{len(frame_paths)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow(window_name, frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(30) & 0xFF  # ~30fps playback
            if key == ord("q") or key == 27:
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Cleanup
    import shutil
    shutil.rmtree(frame_dir, ignore_errors=True)

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sam2_tracker.py <video_path> [--out FILE] [--conf N]")
        print("  video_path: Input video file")
        print("  --out FILE: Save output video to FILE")
        print("  --conf N:   Detection confidence 0-100 (default: 40)")
        print("\nControls: 'q' or ESC to quit, or close window")
        print("\nNote: SAM2 is slower but gives pixel-perfect segmentation masks.")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = None
    confidence = 40

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--out" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--conf" and i + 1 < len(sys.argv):
            confidence = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    track_video_sam2(video_path, output_path, confidence)
