"""
Compute per-frame homography matrices using annotated keyframes
+ Lucas-Kanade optical flow tracking between keyframes.

Output: .npz containing 'H' (N x 3 x 3) and 'valid' (N bool mask).

Usage:
    python3 compute_homographies.py annotations.json "test media/videos/bijon_run.mp4" --out homographies.npz
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from field_schema import FIELD_LANDMARKS


def frame_idx_from_name(name: str) -> int:
    """frame_00126.jpg -> 126"""
    m = re.search(r"frame_(\d+)", name)
    if m is None:
        raise ValueError(f"Could not parse frame index from {name}")
    return int(m.group(1))


def compute_h_from_keypoints(kp_dict: dict[int, tuple[float, float]]) -> np.ndarray | None:
    """Compute homography from {schema_id: (x_pix, y_pix)}. Returns None if <4 points."""
    if len(kp_dict) < 4:
        return None

    img_pts = np.array([kp_dict[k] for k in sorted(kp_dict)], dtype=np.float32)
    field_pts = np.array([FIELD_LANDMARKS[k] for k in sorted(kp_dict)], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, field_pts, cv2.RANSAC, 5.0)
    return H


def track_keypoints(prev_gray: np.ndarray, curr_gray: np.ndarray,
                    prev_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LK forward tracking. Returns (next_pts, status_mask)."""
    if len(prev_pts) == 0:
        return np.empty((0, 2), dtype=np.float32), np.array([], dtype=bool)

    prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    return next_pts.reshape(-1, 2), (status.flatten() == 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-frame homographies")
    parser.add_argument("annotations_json", help="Output of parse_roboflow.py")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--out", default="homographies.npz", help="Output .npz")
    args = parser.parse_args()

    # Load annotations -> map frame_idx -> {kp_id: (x, y)}
    with open(args.annotations_json) as f:
        annotations = json.load(f)

    keyframes: dict[int, dict[int, tuple[float, float]]] = {}
    for fname, kpts in annotations.items():
        idx = frame_idx_from_name(fname)
        keyframes[idx] = {int(k): tuple(v) for k, v in kpts.items()}

    keyframe_indices = sorted(keyframes.keys())
    print(f"Loaded {len(keyframe_indices)} keyframes at indices: {keyframe_indices}")

    # Read all frames as grayscale (fine for ~270 frames at broadcast resolution)
    cap = cv2.VideoCapture(args.video)
    grays: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    total = len(grays)
    print(f"Loaded {total} frames from video")

    homographies: list[np.ndarray | None] = []

    for frame_idx in range(total):
        # Find the nearest keyframe <= this frame (or use first keyframe if none before)
        keyframes_before = [k for k in keyframe_indices if k <= frame_idx]
        anchor = keyframes_before[-1] if keyframes_before else keyframe_indices[0]

        if frame_idx == anchor:
            H = compute_h_from_keypoints(keyframes[anchor])
            homographies.append(H)
            continue

        # Track keypoints from anchor to current frame via LK
        kp_ids = sorted(keyframes[anchor].keys())
        curr_pts = np.array([keyframes[anchor][k] for k in kp_ids], dtype=np.float32)
        curr_ids = list(kp_ids)

        # Before first keyframe — use first keyframe's H
        if frame_idx < anchor:
            H = compute_h_from_keypoints(keyframes[anchor])
            homographies.append(H)
            continue

        # Forward tracking frame by frame from anchor to current
        for i in range(anchor, frame_idx):
            next_pts, status = track_keypoints(grays[i], grays[i + 1], curr_pts)
            curr_pts = next_pts[status]
            curr_ids = [curr_ids[j] for j, ok in enumerate(status) if ok]
            if len(curr_pts) < 4:
                break

        if len(curr_pts) >= 4:
            kp_dict = {curr_ids[i]: tuple(curr_pts[i]) for i in range(len(curr_pts))}
            H = compute_h_from_keypoints(kp_dict)
        else:
            # Track failure — fall back to anchor keyframe's H (stale but valid)
            H = compute_h_from_keypoints(keyframes[anchor])

        homographies.append(H)

    # Save
    valid_mask = np.array([h is not None for h in homographies], dtype=bool)
    H_array = np.array([h if h is not None else np.eye(3, dtype=np.float32)
                        for h in homographies], dtype=np.float32)
    np.savez(args.out, H=H_array, valid=valid_mask)
    print(f"Saved {valid_mask.sum()}/{total} valid homographies to {args.out}")


if __name__ == "__main__":
    main()
