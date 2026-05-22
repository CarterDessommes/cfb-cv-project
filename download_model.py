"""
Download trained RF-DETR model weights from Roboflow for local inference.
"""

from roboflow import Roboflow
from dotenv import load_dotenv
import os
import sys

load_dotenv()


def download_model(project_id="nfl-detection-1500-eeuk7", version_num=1):
    """Download model weights from Roboflow."""

    print(f"Connecting to Roboflow...")
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace().project(project_id)
    version = project.version(version_num)

    print(f"Project: {project.name}")
    print(f"Version: {version_num}")

    # Create weights directory
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Download YOLOv12 format
    fmt = "yolov12"
    try:
        print(f"\nDownloading {fmt} format...")
        model_path = version.download(fmt, location=weights_dir)
        print(f"\nSuccess! Downloaded to: {model_path}")
        print(f"\nDataset ready for training. Run:")
        print(f"  python3 train_yolo.py")
        return model_path
    except Exception as e:
        print(f"  {fmt}: not available ({e})")

    print("\nNo downloadable format found.")
    print("You may need to export the model from the Roboflow web UI:")
    print(f"  https://universe.roboflow.com/{project_id}")
    return None


if __name__ == "__main__":
    project_id = sys.argv[1] if len(sys.argv) > 1 else "nfl-detection-1500-eeuk7"
    version_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    download_model(project_id, version_num)
