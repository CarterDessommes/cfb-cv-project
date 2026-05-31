"""
Fast debug pipeline: player detection + team classification + jersey OCR only.
No ball detection, no field mapping, no homography.

Crop rectangle colors:
  Green  = pose-guided crop, player forward-facing (sent to OCR)
  Yellow = pose-guided crop, player side-facing    (skipped by orientation gate)
  Cyan   = heuristic fallback crop

Usage:
    python3 test_jersey_debug.py <video>
                                 [--det   PATH]      default: weights/player-best.pt
                                 [--ocr   PATH]      default: weights/jersey_ocr.pt
                                 [--pose  PATH]      default: weights/yolo11n-pose.pt
                                 [--no-pose]         disable pose estimator
                                 [--conf  N]         player detection threshold, default 0.4
                                 [--thresh N]        OCR confidence gate, default 0.6
                                 [--blur  N]         blur gate, default 50
                                 [--save-crops DIR]  dump every crop as PNG
                                 [--step]            pause on each frame
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

from team_classifier import TeamClassifier, _best_device
from pipeline import (
    _crops, _stable_number,
    _NUMBER_HISTORY, _NUMBER_LOCKED,
    _is_forward_facing,
    COLORS, PoseEstimator,
    _OCR_CONF_THRESHOLD, _BLUR_THRESHOLD,
)

_CROP_PANEL_W = 320
_CROP_THUMB_H = 72


def _build_crop_panel(crops, labels, frame_h):
    panel = np.zeros((frame_h, _CROP_PANEL_W, 3), dtype=np.uint8)
    y = 0
    for crop, label in zip(crops, labels):
        if y + _CROP_THUMB_H > frame_h:
            break
        th = _CROP_THUMB_H - 18
        scale = min(_CROP_PANEL_W / max(crop.shape[1], 1), th / max(crop.shape[0], 1))
        rw = max(1, int(crop.shape[1] * scale))
        rh = max(1, int(crop.shape[0] * scale))
        thumb = cv2.resize(crop, (rw, rh))
        panel[y:y + rh, :rw] = thumb
        cv2.rectangle(panel, (0, y + rh), (_CROP_PANEL_W, y + _CROP_THUMB_H), (30, 30, 30), -1)
        cv2.putText(panel, label, (4, y + rh + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        y += _CROP_THUMB_H
    return panel


def run(video_path, det_path, ocr_path, pose_path,
        det_conf, ocr_thresh, blur_thresh, save_dir, step,
        pose_every=3, ocr_every=2):
    _NUMBER_HISTORY.clear()
    _NUMBER_LOCKED.clear()

    detector   = YOLO(det_path)
    classifier = TeamClassifier()
    ocr_model  = YOLO(ocr_path)
    pose_est   = PoseEstimator(pose_path) if pose_path else None
    device     = _best_device()
    fitted     = False

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = "Jersey Debug"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_num  = 0
    crop_count = 0
    # Cached results reused on skipped frames
    cached_pose_kps: list | None = None
    cached_number_map: dict[int, tuple[str, float]] = {}
    cached_panel_data: tuple = ([], [], [], [], [])  # (crops, preds, blurs, ocr_idx, valid_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # ── detection + tracking (every frame — needed for tracking continuity)
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

        # ── team classification ──────────────────────────────────────────
        if boxes and not fitted:
            fitted = classifier.fit(frame, boxes)
        team_labels = classifier.classify(frame, boxes) if fitted and boxes else ["unknown"] * len(boxes)

        # ── pose (every pose_every frames) ───────────────────────────────
        if frame_num % pose_every == 0 or cached_pose_kps is None:
            cached_pose_kps = (
                pose_est.get_keypoints_for_boxes(frame, boxes) if pose_est else None
            )
        pose_kps = cached_pose_kps

        crops, valid_idx, rects = _crops(frame, boxes, pose_kps)

        # ── orientation gate — always recomputed (cheap, box count changes) ──
        facing: list[bool] = []
        for j in range(len(crops)):
            box_i = valid_idx[j]
            kps = pose_kps[box_i] if pose_kps and box_i < len(pose_kps) else None
            facing.append(kps is None or _is_forward_facing(kps, boxes[box_i]))

        ocr_indices = [j for j, f in enumerate(facing) if f]
        ocr_crops   = [crops[j] for j in ocr_indices]

        # ── OCR (every ocr_every frames, batched in one model call) ──────
        if frame_num % ocr_every == 0 and ocr_crops:
            raw_blurs   = [cv2.Laplacian(c, cv2.CV_64F).var() for c in ocr_crops]
            sharp_pos   = [k for k, b in enumerate(raw_blurs) if b >= blur_thresh]
            sharp_crops = [ocr_crops[k] for k in sharp_pos]

            batch_results = ocr_model(sharp_crops, device=device, verbose=False) if sharp_crops else []
            batch_preds: list[tuple[str, float] | None] = []
            for r in batch_results:
                conf = float(r.probs.top1conf)
                batch_preds.append((r.names[r.probs.top1], conf) if conf >= ocr_thresh else None)

            raw_preds: list[tuple[str, float] | None] = [None] * len(ocr_crops)
            for idx, k in enumerate(sharp_pos):
                raw_preds[k] = batch_preds[idx]

            number_map: dict[int, tuple[str, float]] = {}
            for k, j in enumerate(ocr_indices):
                if raw_preds[k] is not None:
                    number_map[valid_idx[j]] = raw_preds[k]

            cached_number_map = number_map
            cached_panel_data = (ocr_crops, raw_preds, raw_blurs, ocr_indices, valid_idx)
        else:
            number_map = cached_number_map

        # ── save crops ───────────────────────────────────────────────────
        if save_dir and frame_num % ocr_every == 0:
            sc, sp, sb, si, sv = cached_panel_data
            for k, crop in enumerate(sc):
                pred = sp[k] if k < len(sp) else None
                blur = sb[k] if k < len(sb) else 0.0
                box_i = sv[si[k]] if k < len(si) and si[k] < len(sv) else 0
                label = pred[0] if pred else ("blur" if blur < blur_thresh else "low")
                conf_str = f"{pred[1]:.2f}" if pred else "0.00"
                fname = f"f{frame_num:05d}_p{box_i}_{label}_c{conf_str}_b{blur:.0f}.png"
                cv2.imwrite(os.path.join(save_dir, fname), crop)
                crop_count += 1

        # ── draw crop rectangles ─────────────────────────────────────────
        for j, (sx1, ty1, sx2, ty2) in enumerate(rects):
            box_i     = valid_idx[j]
            used_pose = pose_kps is not None and box_i < len(pose_kps) and pose_kps[box_i] is not None
            if not used_pose:
                color = (255, 255, 0)   # cyan = heuristic
            elif facing[j]:
                color = (0, 255, 0)     # green = pose + forward-facing
            else:
                color = (0, 165, 255)   # orange = pose + side-facing (skipped)
            cv2.rectangle(frame, (sx1, ty1), (sx2, ty2), color, 1)

        # ── draw player boxes + labels ───────────────────────────────────
        for k, (box, team) in enumerate(zip(boxes, team_labels)):
            track_id = int(box[4])
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            color = COLORS[team]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            raw    = number_map.get(k)
            number = _stable_number(track_id, *raw) if raw is not None else "?"
            locked = "L" if track_id in _NUMBER_LOCKED else ""
            cv2.putText(frame, f"#{number}{locked} {team[:3].upper()}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── crop panel (use cached OCR data — stable across skipped frames) ──
        p_crops, p_preds, p_blurs, p_ocr_idx, p_valid_idx = cached_panel_data
        panel_crops, panel_labels = [], []
        for k, j in enumerate(p_ocr_idx):
            if j >= len(p_valid_idx):
                continue
            box_i = p_valid_idx[j]
            if box_i >= len(boxes):
                continue
            tid    = int(boxes[box_i][4]) if box_i < len(boxes) else -1
            pred   = p_preds[k] if k < len(p_preds) else None
            blur   = p_blurs[k] if k < len(p_blurs) else 0.0
            locked = "L" if tid in _NUMBER_LOCKED else ""
            if pred is not None:
                label_str, conf = pred
                status = f"#{label_str}{locked} c={conf:.2f} b={blur:.0f}"
            elif blur < blur_thresh:
                status = f"BLUR b={blur:.0f}"
            else:
                status = f"LOW  b={blur:.0f}"
            panel_crops.append(p_crops[k])
            panel_labels.append(f"id={tid}  {status}")

        # ── stdout ────────────────────────────────────────────────────────
        n_side = sum(1 for f in facing if not f)
        if p_preds:
            parts = []
            for k in range(len(p_preds)):
                pred = p_preds[k]
                blur = p_blurs[k] if k < len(p_blurs) else 0.0
                tag  = f"#{pred[0]}({pred[1]:.2f})" if pred else ("BLUR" if blur < blur_thresh else "LOW")
                parts.append(tag)
            print(f"Frame {frame_num:5d}: players={len(boxes)} side={n_side} ocr={parts}")

        # ── compose display ───────────────────────────────────────────────
        panel    = _build_crop_panel(panel_crops, panel_labels, frame.shape[0])
        combined = np.hstack([frame, panel])
        cv2.putText(combined,
                    f"Frame {frame_num}/{total}  thresh={ocr_thresh:.2f}  blur={blur_thresh:.0f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow(window, combined)
        if cv2.waitKey(0 if step else 1) & 0xFF in (ord("q"), 27):
            break

    print(f"\nDone. {frame_num} frames, {crop_count} crops saved.")
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_jersey_debug.py <video> [options]")
        sys.exit(1)

    video        = sys.argv[1]
    det_path     = "weights/player-best.pt"
    ocr_path     = "weights/jersey_ocr.pt"
    pose_path    = "weights/yolo11n-pose.pt"
    det_conf     = 0.4
    ocr_thresh   = _OCR_CONF_THRESHOLD
    blur_thresh  = _BLUR_THRESHOLD
    save_dir     = None
    step         = False
    pose_every   = 3
    ocr_every    = 2

    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if   a == "--det"        and i + 1 < len(sys.argv): det_path    = sys.argv[i+1]; i += 2
        elif a == "--ocr"        and i + 1 < len(sys.argv): ocr_path    = sys.argv[i+1]; i += 2
        elif a == "--pose"       and i + 1 < len(sys.argv): pose_path   = sys.argv[i+1]; i += 2
        elif a == "--no-pose":                               pose_path   = None;          i += 1
        elif a == "--conf"       and i + 1 < len(sys.argv): det_conf    = float(sys.argv[i+1]); i += 2
        elif a == "--thresh"     and i + 1 < len(sys.argv): ocr_thresh  = float(sys.argv[i+1]); i += 2
        elif a == "--blur"       and i + 1 < len(sys.argv): blur_thresh = float(sys.argv[i+1]); i += 2
        elif a == "--pose-every" and i + 1 < len(sys.argv): pose_every  = int(sys.argv[i+1]);   i += 2
        elif a == "--ocr-every"  and i + 1 < len(sys.argv): ocr_every   = int(sys.argv[i+1]);   i += 2
        elif a == "--save-crops" and i + 1 < len(sys.argv): save_dir    = sys.argv[i+1]; i += 2
        elif a == "--step":                                  step = True; i += 1
        else: i += 1

    run(video, det_path, ocr_path, pose_path,
        det_conf, ocr_thresh, blur_thresh, save_dir, step,
        pose_every=pose_every, ocr_every=ocr_every)
