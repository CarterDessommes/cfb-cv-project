"""
Train YOLOv11 jersey number classifier on your custom dataset.
Optimized for small datasets (~1000 images) with varying rectangular dimensions.
"""
#!pip install ultralytics
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Paths
DATASET_DIR = Path("/kaggle/input/datasets/andrewleick/labels1")
OUTPUT_DIR  = Path("/kaggle/working/jersey_cls")
CSV_PATH    = DATASET_DIR / "labels.csv"

# --- OPTIMIZED HYPERPARAMETERS FOR YOUR DATASET ---
EPOCHS     = 100
BATCH_SIZE = 32
IMG_SIZE   = 128
MIN_CLASS_SAMPLES = 4

def build_classification_dataset():
    # Wipe any previous run's data
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    images_dir = DATASET_DIR / "crops" / "crops"

    csv_filenames = set(df["filename"])
    disk_filenames = set(f.name for f in images_dir.iterdir() if f.is_file()) if images_dir.exists() else set()

    not_in_csv = disk_filenames - csv_filenames
    if not_in_csv:
        print(f"Skipping {len(not_in_csv)} crops with no CSV label.")
        df = df[df["filename"].isin(disk_filenames)]

    not_on_disk = csv_filenames - disk_filenames
    if len(not_on_disk) > 20:
        raise ValueError(f"CRITICAL: Found {len(not_on_disk)} rows in CSV missing from disk.")
    elif not_on_disk:
        df = df[df["filename"].isin(disk_filenames)]

    # Drop rare classes that can't be meaningfully learned or split
    label_counts = df["label"].value_counts()
    rare = label_counts[label_counts < MIN_CLASS_SAMPLES].index
    if len(rare) > 0:
        print(f"Dropping {len(rare)} classes with fewer than {MIN_CLASS_SAMPLES} images: {sorted(rare.tolist())}")
        df = df[~df["label"].isin(rare)]

    print(f"Dataset after filtering: {len(df)} images, {df['label'].nunique()} classes")

    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Auto-heal: drop any classes still missing from val
    train_classes = set(train_df["label"].unique())
    val_classes   = set(val_df["label"].unique())
    if train_classes != val_classes:
        missing = train_classes - val_classes
        print(f"Warning: dropping {len(missing)} classes still missing from val: {sorted(missing)}")
        train_df = train_df[~train_df["label"].isin(missing)]
        val_df   = val_df[~val_df["label"].isin(missing)]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Classes: {train_df['label'].nunique()}")

    for split, split_df in [("train", train_df), ("val", val_df)]:
        for _, row in split_df.iterrows():
            src = images_dir / row["filename"]
            dst = OUTPUT_DIR / split / str(row["label"]) / row["filename"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src), str(dst))

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
        dropout=0.1,
        patience=15,
    )
    print("\nTraining complete! Best weights saved to: /kaggle/working/runs/jersey_ocr/weights/best.pt")

if __name__ == "__main__":
    data_dir = build_classification_dataset()
    train(data_dir)