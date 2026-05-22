"""
Train YOLOv12s on your custom dataset.
"""

from ultralytics import YOLO
import os
import sys


def train(
    dataset_dir="weights/nfl-detection-1500-eeuk7-1",
    epochs=50,
    batch_size=16,
    output_dir="weights"
):
    """Train YOLOv12s on dataset."""

    # Find data.yaml
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    if not os.path.exists(data_yaml):
        print(f"Dataset not found at {dataset_dir}")
        print("Run 'python3 download_model.py' first to download your dataset.")
        sys.exit(1)

    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()

    # Load YOLOv12s (small model - good balance of speed/accuracy)
    print("Loading YOLOv12s...")
    model = YOLO("yolo12s.pt")

    # Train
    print("Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device="mps",  # Apple Silicon
        project=output_dir,
        name="nfl_yolo12s",
    )

    # Best weights path
    best_weights = os.path.join(output_dir, "nfl_yolo12s", "weights", "best.pt")
    print(f"\nTraining complete!")
    print(f"Best weights: {best_weights}")
    print(f"\nTo use with tracker:")
    print(f'  python3 tracker.py "test media/videos/bijon_run.mp4" --model {best_weights}')


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    train(epochs=epochs, batch_size=batch_size)
