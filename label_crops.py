"""
Phase 1 VLM labeling: extract pose-guided, forward-facing jersey crops from a
video and label them with Qwen2.5-VL.

Two modes:
  --extract-only   Run detection + pose + orientation gate. Save crops, no VLM.
                   Use this first to check crop quality before spending VLM time.
  (default)        Extract crops AND run Qwen2.5-VL to label them.
                   Outputs crops/ directory + labels.tsv for PARSeq fine-tuning.

Usage:
    python3 label_crops.py <video> --out-dir <dir>
                           [--det   PATH]   default: weights/player-best.pt
                           [--pose  PATH]   default: weights/yolo11n-pose.pt
                           [--conf  N]      player detection threshold, default 0.4
                           [--every N]      sample every N frames, default 5
                           [--extract-only]
                           [--max   N]      stop after N crops, default unlimited

Install for VLM mode:
    pip install transformers accelerate qwen-vl-utils
    # model is ~8 GB; set HF_HOME to a disk with space
"""

import sys
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO

from team_classifier import _best_device
from pipeline import (
    _pose_crop, _is_forward_facing, _box_iou,
    PoseEstimator, _BLUR_THRESHOLD,
)


def _load_vlm():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("ERROR: VLM dependencies missing. Run:\n"
              "  pip install transformers accelerate qwen-vl-utils")
        sys.exit(1)

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading {model_id} …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, process_vision_info


_VLM_PROMPT = (
    "This is a tight crop of a football player's chest showing their jersey number. "
    "What is the jersey number? "
    "Reply with ONLY the digits (e.g. '12' or '7'). "
    "If the number is not clearly readable, reply with 'unreadable'."
)


def _query_vlm(model, processor, process_vision_info, crop_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(crop_path)}"},
                {"type": "text",  "text": _VLM_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    ids = model.generate(**inputs, max_new_tokens=10)
    out = processor.batch_decode(
        ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()
    # Normalise: keep only digit characters, or return "unreadable"
    digits = "".join(c for c in out if c.isdigit())
    return digits if digits else "unreadable"


def run(video_path, out_dir, det_path, pose_path,
        det_conf, every_n, extract_only, max_crops):
    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    labels_path = os.path.join(out_dir, "labels.tsv")

    detector = YOLO(det_path)
    pose_est = PoseEstimator(pose_path)
    device   = _best_device()

    vlm_model = vlm_proc = vlm_pvi = None
    if not extract_only:
        vlm_model, vlm_proc, vlm_pvi = _load_vlm()

    cap       = cv2.VideoCapture(video_path)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    saved     = 0

    with open(labels_path, "w", newline="") as tsv_f:
        writer = csv.writer(tsv_f, delimiter="\t")
        writer.writerow(["filename", "label"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % every_n != 0:
                continue
            if max_crops and saved >= max_crops:
                break

            # ── detection ────────────────────────────────────────────────
            results = detector.track(frame, persist=True, conf=det_conf,
                                     verbose=False, device=device, half=True)
            boxes = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                ids  = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, tid, cls in zip(xyxy, ids, clss):
                    if cls == 0:
                        boxes.append([*box, tid, cls])
            if not boxes:
                continue

            # ── pose + orientation gate ───────────────────────────────────
            pose_kps = pose_est.get_keypoints_for_boxes(frame, boxes)

            for j, box in enumerate(boxes):
                if max_crops and saved >= max_crops:
                    break
                kps = pose_kps[j] if j < len(pose_kps) else None

                # Require a genuine pose-guided crop — _pose_crop does its own
                # confidence gate so check it directly rather than trusting kps != None
                crop, _ = _pose_crop(frame, kps, box, frame.shape[1]) if kps is not None else (None, None)
                if crop is None:
                    continue

                # Orientation + blur gates
                if not _is_forward_facing(kps, box):
                    continue
                blur = cv2.Laplacian(crop, cv2.CV_64F).var()
                if blur < _BLUR_THRESHOLD:
                    continue

                # Minimum size gate — small distant players won't be readable
                if crop.shape[0] < 30 or crop.shape[1] < 20:
                    continue

                tid  = int(box[4])
                fname = f"f{frame_num:06d}_t{tid:03d}.png"
                fpath = os.path.join(crops_dir, fname)
                cv2.imwrite(fpath, crop)

                if not extract_only:
                    label = _query_vlm(vlm_model, vlm_proc, vlm_pvi, fpath)
                    writer.writerow([fname, label])
                    tsv_f.flush()
                    print(f"  [{saved+1}] {fname} → {label}")
                else:
                    print(f"  [{saved+1}] {fname}  blur={blur:.0f}")

                saved += 1

            pct = frame_num / max(total, 1) * 100
            print(f"\rFrame {frame_num}/{total} ({pct:.0f}%)  crops saved: {saved}", end="")

    cap.release()
    print(f"\nDone. {saved} crops in {crops_dir}")
    if not extract_only:
        print(f"Labels → {labels_path}")
        print("\nNext step: fine-tune PARSeq on these crops.")
        print("  git clone https://github.com/baudm/parseq")
        print(f"  # then point its data config at {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 label_crops.py <video> --out-dir <dir> [options]")
        sys.exit(1)

    video        = sys.argv[1]
    out_dir      = "labeled_crops"
    det_path     = "weights/player-best.pt"
    pose_path    = "weights/yolo11n-pose.pt"
    det_conf     = 0.4
    every_n      = 5
    extract_only = False
    max_crops    = None

    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if   a == "--out-dir"      and i + 1 < len(sys.argv): out_dir      = sys.argv[i+1]; i += 2
        elif a == "--det"          and i + 1 < len(sys.argv): det_path     = sys.argv[i+1]; i += 2
        elif a == "--pose"         and i + 1 < len(sys.argv): pose_path    = sys.argv[i+1]; i += 2
        elif a == "--conf"         and i + 1 < len(sys.argv): det_conf     = float(sys.argv[i+1]); i += 2
        elif a == "--every"        and i + 1 < len(sys.argv): every_n      = int(sys.argv[i+1]); i += 2
        elif a == "--max"          and i + 1 < len(sys.argv): max_crops    = int(sys.argv[i+1]); i += 2
        elif a == "--extract-only":                            extract_only = True; i += 1
        else: i += 1

    run(video, out_dir, det_path, pose_path, det_conf, every_n, extract_only, max_crops)
