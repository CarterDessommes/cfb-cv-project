"""
Train YOLOv11 jersey number classifier on:
  https://www.kaggle.com/datasets/frlemarchand/nfl-player-numbers

Each player crop image is classified as its jersey number (0-99).
Integrates with TeamClassifier: same crop in, jersey number out.

Run on Kaggle with GPU T4 x2 enabled.
After training, download: /kaggle/working/runs/jersey_ocr/weights/best.pt
"""

# !pip install ultralytics

import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


DATASET_DIR = Path("/kaggle/input/datasets/frlemarchand/nfl-player-numbers")
OUTPUT_DIR  = Path("/kaggle/working/jersey_cls")
CSV_PATH    = DATASET_DIR / "train_player_numbers.csv"

EPOCHS     = 30
BATCH_SIZE = 64
IMG_SIZE   = 224


def build_classification_dataset():
    df = pd.read_csv(CSV_PATH)

    # Stratified split so rare numbers appear in both sets
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Classes: {df['label'].nunique()}")

    for split, split_df in [("train", train_df), ("val", val_df)]:
        for _, row in split_df.iterrows():
            src = DATASET_DIR / row["filepath"]
            dst = OUTPUT_DIR / split / str(row["label"]) / row["filename"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src), str(dst))

    print(f"Dataset written to {OUTPUT_DIR}")
    return str(OUTPUT_DIR)


def train(data_dir: str):
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=data_dir,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project="/kaggle/working/runs",
        name="jersey_ocr",
    )
    print("\nTraining complete!")
    print("Best weights: /kaggle/working/runs/jersey_ocr/weights/best.pt")


if __name__ == "__main__":
    data_dir = build_classification_dataset()
    train(data_dir)
