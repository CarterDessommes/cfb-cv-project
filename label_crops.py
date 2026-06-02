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
                           [--max-frames N] stop at absolute frame number N
                           [--start-frame N] seek to frame N before extracting

Install for VLM mode:
    pip install transformers accelerate qwen-vl-utils
    # model is ~8 GB; set HF_HOME to a disk with space
"""

import sys
import os
import csv
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from team_classifier import _best_device
from pipeline import (
    _pose_crop, _is_forward_facing, _box_iou, _number_crop,
    PoseEstimator,
)


# ── Dataset-diversity controls ──────────────────────────────────────────────
# A tracked player appears in every sampled frame, so without limits one or two
# players flood the set with near-identical crops while others are starved. Cap
# how many crops a single track contributes and space them out in time so the
# kept crops capture different poses/angles instead of consecutive frames.
_MAX_PER_TRACK   = 8     # most crops to keep from any single track id
_TRACK_FRAME_GAP = 20    # min raw-frame gap between two saved crops of one track

# Extraction gates are deliberately lenient: the VLM is the real readability
# filter (it returns "unreadable" on its own), so let partially blurry/occluded
# crops through rather than discarding players up front.
_EXTRACT_BLUR_MIN = 12   # well below the pipeline inference threshold (50)
_EXTRACT_MIN_H    = 24
_EXTRACT_MIN_W    = 16


def _load_vlm(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("ERROR: VLM dependencies missing. Run:\n"
              "  pip install transformers accelerate qwen-vl-utils")
        sys.exit(1)

    # Place the whole model on one device in fp16. device_map="auto" offloads
    # layers to disk on a 16 GB Mac, which both is slow and breaks the
    # multimodal merge (image-token tensor-size mismatch) on offloaded params.
    device = _best_device()
    dtype  = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading {model_id} on {device} ({dtype})…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, process_vision_info


_VLM_PROMPT = (
    "You are reading the jersey number from a tight crop of the chest (front) of an "
    "American college football player's jersey.\n"
    "\n"
    "Rules:\n"
    "1. Valid jersey numbers are 0 through 99. A number is either a single digit "
    "(0-9) or two digits (00-99). '0' and '00' are both valid and are DIFFERENT "
    "numbers — report whichever you actually see.\n"
    "2. Report ONLY the large number printed on the jersey fabric. Ignore the "
    "player's name, the team logo, the manufacturer/brand mark, any collar or "
    "sleeve lettering, and any on-screen TV graphics (score bug, clock, "
    "down-and-distance).\n"
    "3. Read left to right. If the number has two digits, both digits must be "
    "fully visible. If a fold, arm, or the crop edge hides part of a digit, or only "
    "one digit of a two-digit number is visible, the number is NOT readable.\n"
    "4. Do NOT guess. If the digits are blurred, glared, too small, or you cannot "
    "confidently distinguish similar shapes (e.g. 6 vs 8, 1 vs 7, 3 vs 8, 0 vs 6), "
    "the number is NOT readable.\n"
    "\n"
    "Output: reply with the digits ONLY and nothing else — for example: 7, 0, 42, "
    "or 00. If the number is not fully and clearly readable under the rules above, "
    "reply with exactly the word: unreadable"
)


_VLM_MAX_SIDE = 224   # resize a crop's longer side to this (multiple of 28) for Qwen


def _vlm_image(crop_path: str):
    """Load a saved crop and resize to a Qwen-safe size (both dims multiple of 28).

    Upscaling tiny crops also avoids the degenerate too-few-visual-tokens grid
    and measurably improves the model's number reads.
    """
    pil  = Image.open(crop_path).convert("RGB")
    w, h = pil.size
    sc   = _VLM_MAX_SIDE / max(w, h)
    nw   = max(28, int(round(w * sc / 28)) * 28)
    nh   = max(28, int(round(h * sc / 28)) * 28)
    return pil.resize((nw, nh))


def _query_vlm(model, processor, process_vision_info, crop_path: str) -> str:
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _vlm_image(crop_path)},
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
    except Exception as e:
        # One bad crop must never kill a long unattended run.
        print(f"\n  VLM error on {os.path.basename(crop_path)}: "
              f"{type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
        return "unreadable"
    # Normalise: keep only digit characters, or return "unreadable"
    digits = "".join(c for c in out if c.isdigit())
    return digits if digits else "unreadable"


def run(video_path, out_dir, det_path, pose_path,
        det_conf, every_n, extract_only, max_crops, max_frames=None,
        start_frame=0, vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    labels_path = os.path.join(out_dir, "labels.tsv")

    detector = YOLO(det_path)
    pose_est = PoseEstimator(pose_path)
    device   = _best_device()

    vlm_model = vlm_proc = vlm_pvi = None
    if not extract_only:
        vlm_model, vlm_proc, vlm_pvi = _load_vlm(vlm_model_id)

    cap       = cv2.VideoCapture(video_path)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    saved     = 0

    # Per-track dedup state (see _MAX_PER_TRACK / _TRACK_FRAME_GAP above).
    track_count: dict[int, int] = {}
    track_last:  dict[int, int] = {}

    end_frame = min(max_frames, total) if max_frames else total
    pbar = tqdm(total=max(end_frame - start_frame, 0), unit="f",
                desc="labeling" if not extract_only else "extracting")

    append = os.path.exists(labels_path)
    with open(labels_path, "a" if append else "w", newline="") as tsv_f:
        writer = csv.writer(tsv_f, delimiter="\t")
        if not append:
            writer.writerow(["filename", "label"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            pbar.update(1)
            if max_frames and frame_num > max_frames:
                break
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
                tid = int(box[4])

                # Dedup: skip tracks we've already sampled enough, or sampled too
                # recently, so no single player floods the dataset.
                if track_count.get(tid, 0) >= _MAX_PER_TRACK:
                    continue
                if frame_num - track_last.get(tid, -_TRACK_FRAME_GAP) < _TRACK_FRAME_GAP:
                    continue

                kps = pose_kps[j] if j < len(pose_kps) else None

                # Pose is used only for gating. When it's available, gate on the
                # narrow pose crop (tracks the on-screen number scale); when it's
                # missing, fall back to the number crop so a pose miss doesn't
                # drop the player outright — the VLM still filters readability.
                pose_c, _ = _pose_crop(frame, kps, box, frame.shape[1]) if kps is not None else (None, None)

                # Orientation gate stays lenient: _is_forward_facing already
                # passes when keypoints can't decide, and we skip it with no pose.
                if kps is not None and not _is_forward_facing(kps, box):
                    continue

                gate_crop = pose_c if pose_c is not None else _number_crop(frame, box)
                if gate_crop is None or gate_crop.size == 0:
                    continue

                # Lenient blur + size gates — keep partially blurry/occluded crops.
                blur = cv2.Laplacian(gate_crop, cv2.CV_64F).var()
                if blur < _EXTRACT_BLUR_MIN:
                    continue
                if gate_crop.shape[0] < _EXTRACT_MIN_H or gate_crop.shape[1] < _EXTRACT_MIN_W:
                    continue

                # Save a generous crop covering the whole player so pose error or
                # an off-center number can't clip the digits.
                crop = _number_crop(frame, box)
                if crop.size == 0:
                    continue
                crop = np.ascontiguousarray(crop)  # ensure cv2 can encode the view

                fname = f"f{frame_num:06d}_t{tid:03d}.png"
                fpath = os.path.join(crops_dir, fname)
                if not cv2.imwrite(fpath, crop):
                    pbar.write(f"  skip {fname}: imwrite failed (shape={crop.shape})")
                    continue

                track_count[tid] = track_count.get(tid, 0) + 1
                track_last[tid]  = frame_num

                if not extract_only:
                    label = _query_vlm(vlm_model, vlm_proc, vlm_pvi, fpath)
                    writer.writerow([fname, label])
                    tsv_f.flush()
                    pbar.write(f"  [{saved+1}] {fname} → {label}")
                else:
                    pbar.write(f"  [{saved+1}] {fname}  blur={blur:.0f}")

                saved += 1

            pbar.set_postfix(crops=saved)

    pbar.close()
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
    max_frames   = None
    start_frame  = 0
    vlm_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if   a == "--out-dir"      and i + 1 < len(sys.argv): out_dir      = sys.argv[i+1]; i += 2
        elif a == "--det"          and i + 1 < len(sys.argv): det_path     = sys.argv[i+1]; i += 2
        elif a == "--pose"         and i + 1 < len(sys.argv): pose_path    = sys.argv[i+1]; i += 2
        elif a == "--conf"         and i + 1 < len(sys.argv): det_conf     = float(sys.argv[i+1]); i += 2
        elif a == "--every"        and i + 1 < len(sys.argv): every_n      = int(sys.argv[i+1]); i += 2
        elif a == "--max"          and i + 1 < len(sys.argv): max_crops    = int(sys.argv[i+1]); i += 2
        elif a == "--max-frames"   and i + 1 < len(sys.argv): max_frames   = int(sys.argv[i+1]); i += 2
        elif a == "--start-frame"  and i + 1 < len(sys.argv): start_frame  = int(sys.argv[i+1]); i += 2
        elif a == "--vlm-model"    and i + 1 < len(sys.argv): vlm_model_id = sys.argv[i+1]; i += 2
        elif a == "--extract-only":                            extract_only = True; i += 1
        else: i += 1

    run(video, out_dir, det_path, pose_path, det_conf, every_n, extract_only,
        max_crops, max_frames, start_frame, vlm_model_id)
