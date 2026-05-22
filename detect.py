from dotenv import load_dotenv
from roboflow import Roboflow
import os

load_dotenv()

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
model = rf.workspace().project("nfl-detection-1500-eeuk7").version(1).model


def detect_players(image_path, confidence=40):
    """Run NFL player detection on an image."""
    prediction = model.predict(image_path, confidence=confidence)
    return prediction.json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>")
        sys.exit(1)

    result = detect_players(sys.argv[1])

    print(f"Found {len(result['predictions'])} detections:\n")
    for obj in result["predictions"]:
        x, y = obj["x"], obj["y"]
        w, h = obj["width"], obj["height"]
        label = obj["class"]
        conf = obj["confidence"]
        print(f"  {label}: ({x:.0f}, {y:.0f}) {w:.0f}x{h:.0f} conf={conf:.2f}")
