from dotenv import load_dotenv
from roboflow import Roboflow
import cv2
import sys
import os

load_dotenv()

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
model = rf.workspace().project("nfl-detection-1500-eeuk7").version(1).model


def visualize(image_path, confidence=40):
    """Run detection and display results. Press any key to close."""
    prediction = model.predict(image_path, confidence=confidence)
    result = prediction.json()

    img = cv2.imread(image_path)

    for obj in result["predictions"]:
        x, y = int(obj["x"]), int(obj["y"])
        w, h = int(obj["width"]), int(obj["height"])
        label = obj["class"]
        conf = obj["confidence"]

        # Convert center coords to top-left
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        # Color by class
        color = (0, 255, 0) if label == "0" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(f"Found {len(result['predictions'])} detections. Press any key to close.")
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <image_path> [confidence]")
        sys.exit(1)

    image_path = sys.argv[1]
    confidence = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    visualize(image_path, confidence)
