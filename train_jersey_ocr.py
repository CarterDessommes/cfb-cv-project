"""
Train YOLOv11 jersey number classifier on your custom dataset.

Optimized for small datasets (~460 images) with varying rectangular dimensions.
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
TSV_PATH    = DATASET_DIR / "labels_bijon_pass1_backup.tsv"

# --- OPTIMIZED HYPERPARAMETERS FOR YOUR DATASET ---
EPOCHS     = 100  # Increased to give the model more time to converge
BATCH_SIZE = 64   # Lowered so the model updates its weights ~11-12 times per epoch
IMG_SIZE   = 128  # Increased to prevent digit distortion/pixelation


def build_classification_dataset():
    df = pd.read_csv(TSV_PATH, sep='\t')
    images_dir = DATASET_DIR / "crops" / "crops"
    
    tsv_filenames = set(df["filename"])
    disk_filenames = set(f.name for f in images_dir.iterdir() if f.is_file()) if images_dir.exists() else set()

    # Crops on disk with no TSV entry are unlabeled (e.g. in-progress run) — skip silently.
    not_in_tsv = disk_filenames - tsv_filenames
    if not_in_tsv:
        print(f"Skipping {len(not_in_tsv)} crops with no TSV label.")
        df = df[df["filename"].isin(disk_filenames)]

    not_on_disk = tsv_filenames - disk_filenames
    if len(not_on_disk) > 20:
        raise ValueError(f"CRITICAL: Found {len(not_on_disk)} rows in TSV missing from disk.")
    elif not_on_disk:
        df = df[df["filename"].isin(disk_filenames)]

    # Safe Stratification handling
    label_counts = df["label"].value_counts()
    singletons = df[df["label"].isin(label_counts[label_counts == 1].index)]
    stratified_df = df[df["label"].isin(label_counts[label_counts > 1].index)]

    if not stratified_df.empty:
        train_df, val_df = train_test_split(
            stratified_df, test_size=0.2, random_state=42, stratify=stratified_df["label"]
        )
        train_df = pd.concat([train_df, singletons], ignore_index=True)
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Classes: {df['label'].nunique()}")

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
        # Custom additions for small datasets:
        dropout=0.1,    # Adds a small dropout layer to prevent overfitting on 460 images
        patience=15,    # Early stopping: if validation loss doesn't improve for 15 epochs, wrap up early
    )
    print("\nTraining complete! Best weights saved to: /kaggle/working/runs/jersey_ocr/weights/best.pt")


if __name__ == "__main__":
    data_dir = build_classification_dataset()
    train(data_dir)