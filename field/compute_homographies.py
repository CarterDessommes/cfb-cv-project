"""
Compute per-frame homography matrices from annotated keyframes.
Output: .npz containing 'H' (N×3×3) and 'valid' (N bool mask).

python3 field/compute_homographies.py field/annotations.json video.mp4 --out field/homographies.npz --interp --field-refine

Modes:
  --interp: linear interpolation between keyframes + optional Hough correction
  --legacy: LK-track annotated keypoints from nearest anchor
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from .field_schema import FIELD_LANDMARKS
from .field_geometry_auto import split_line_families

FIELD_WIDTH  = 53.33
FIELD_LENGTH = 100.0


# Keyframe homography

def frame_idx_from_name(name: str) -> int:
    m = re.search(r"frame_(\d+)", name)
    if m is None:
        raise ValueError(f"Cannot parse frame index from {name!r}")
    return int(m.group(1))


def compute_h_from_keypoints(kp_dict: dict[int, tuple[float, float]]) -> np.ndarray | None:
    """Homography from {schema_id: (x_pix, y_pix)}.  None if less than 4 points."""
    if len(kp_dict) < 4:
        return None
    img_pts   = np.array([kp_dict[k]        for k in sorted(kp_dict)], dtype=np.float32)
    field_pts = np.array([FIELD_LANDMARKS[k] for k in sorted(kp_dict)], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, field_pts, cv2.RANSAC, 5.0)
    return H


# Field mask (green grass + white lines only — excludes scoreboard / crowd)

def _field_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask: green grass OR white field lines, bottom 15 excluded."""
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 35, 35]),  np.array([85, 255, 255]))
    white = cv2.inRange(hsv, np.array([0,  0,  160]), np.array([180, 55, 255]))
    mask  = cv2.bitwise_or(green, white)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask[int(bgr.shape[0] * 0.85):] = 0   # scoreboard strip
    return mask


# Field-line detection: far sideline + yard lines

def _sideline_pts(mask: np.ndarray, which: str = "far",
                  step: int = 30) -> list[tuple[float, float]]:
    """
    Sample the far (top) or near (bottom) sideline from the green field mask.
    'far'  -> topmost green row per column
    'near' -> bottommost green row per column
    """
    h, w = mask.shape
    pts = []
    for x in range(step, w - step, step):
        rows = np.where(mask[:, x] > 0)[0]
        if len(rows) < 10:
            continue
        if which == "far":
            row = int(rows[0])
            if h * 0.05 < row < h * 0.65:
                pts.append((float(x), float(row)))
        else:
            row = int(rows[-1])
            if h * 0.35 < row < h * 0.95:
                pts.append((float(x), float(row)))
    return pts


def detect_far_sideline_pts(bgr: np.ndarray, mask: np.ndarray, step: int = 30):
    return _sideline_pts(mask, "far",  step)


def detect_near_sideline_pts(bgr: np.ndarray, mask: np.ndarray, step: int = 30):
    return _sideline_pts(mask, "near", step)


HASH_FIELD_YS = (23.58, 29.75)   # far hash line, near hash line


def _find_hash_marks_on_segment(
    segment: tuple,
    white_mask: np.ndarray,
    n_samples: int = 50,
    min_extent: int = 5,
    search_width: int = 60,
) -> list[tuple[float, float]]:
    """
    Sample along a nearly-vertical yard-line segment and find the rows where
    white pixels extend significantly left/right (= hash marks crossing the line).
    Returns up to 2 pixel (u, v) positions sorted top-to-bottom.
    maps to HASH_FIELD_YS[0] and HASH_FIELD_YS[1] respectively by the caller.
    """
    x1, y1, x2, y2 = segment
    h, w = white_mask.shape
    extents: list[tuple[float, float, float]] = []   # (extent, u, v)
    for t in np.linspace(0.05, 0.95, n_samples):
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        ipx, ipy = int(round(px)), int(round(py))
        if not (0 <= ipx < w and 0 <= ipy < h):
            continue
        ext = 0
        for dx in range(1, search_width + 1):
            if ipx - dx >= 0 and white_mask[ipy, ipx - dx]:
                ext += 1
            else:
                break
        for dx in range(1, search_width + 1):
            if ipx + dx < w and white_mask[ipy, ipx + dx]:
                ext += 1
            else:
                break
        extents.append((float(ext), px, py))

    candidates = [(e, px, py) for e, px, py in extents if e >= min_extent]
    if len(candidates) < 2:
        return []

    ys = [py for _, _, py in candidates]
    mid_y = (min(ys) + max(ys)) / 2.0
    result: list[tuple[float, float]] = []
    for group in [
        [(e, px, py) for e, px, py in candidates if py < mid_y],
        [(e, px, py) for e, px, py in candidates if py >= mid_y],
    ]:
        if not group:
            continue
        best_e = max(e for e, _, _ in group)
        thresh = best_e * 0.7
        good = [(px, py) for e, px, py in group if e >= thresh]
        if good:
            result.append((float(np.mean([p for p, _ in good])),
                           float(np.mean([p for _, p in good]))))
    return result  # [(u_top, v_top), (u_bot, v_bot)]


def refine_H_with_field_lines(bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Detect the far/near sidelines, yard lines, and hash mark crossings and use them as additional constraints to correct the current H.

    Far sideline points  -> field_y = 0
    Near sideline points -> field_y = FIELD_WIDTH
    Yard-line points     -> field_x = N*10
    Hash mark crossings  -> (field_x=N, field_y=23.58 or 29.75)  <- hard y anchor
    """
    h_img, w_img = bgr.shape[:2]

    mask = _field_mask(bgr)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return H

    img_pts:   list[list[float]] = []
    field_pts: list[list[float]] = []

    # Far sideline -> field_y = 0, near sideline -> field_y = FIELD_WIDTH
    for sideline_pts, field_y_target in [
        (detect_far_sideline_pts(bgr,  mask), 0.0),
        (detect_near_sideline_pts(bgr, mask), FIELD_WIDTH),
    ]:
        for u, v in sideline_pts:
            fp = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float64), H)[0][0]
            if abs(float(fp[1]) - field_y_target) > 8.0:
                continue
            field_x = float(np.clip(fp[0], 0.0, FIELD_LENGTH))
            img_pts.append([u, v])
            field_pts.append([field_x, field_y_target])

    # Yard lines -> field_x = N*10
    gray_m = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.bitwise_and(gray_m, gray_m, mask=mask)
    edges  = cv2.Canny(cv2.GaussianBlur(gray_m, (5, 5), 0), 50, 150)
    raw    = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                              minLineLength=w_img * 0.12, maxLineGap=25)

    matched_segs: dict[int, tuple] = {}   # yard_x → matched segment (for hash marks)

    if raw is not None and len(raw) >= 2:
        all_segs = [tuple(map(float, r[0])) for r in raw]
        yard_segs, _ = split_line_families(all_segs)

        predicted: dict[int, tuple[float, float]] = {}
        for yard_x in range(0, 101, 10):
            pt = cv2.perspectiveTransform(
                np.array([[[float(yard_x), FIELD_WIDTH / 2]]], dtype=np.float64), H_inv
            )[0][0]
            if -w_img * 0.5 <= pt[0] <= w_img * 1.5 and -h_img * 0.2 <= pt[1] <= h_img * 1.2:
                predicted[yard_x] = (float(pt[0]), float(pt[1]))

        seg_mids = [((x1+x2)/2, (y1+y2)/2, seg) for seg in yard_segs for x1,y1,x2,y2 in [seg]]
        threshold = w_img * 0.07
        used: set[int] = set()

        for yard_x, (px, py) in sorted(predicted.items()):
            best_d, best_i = float("inf"), -1
            for i, (mx, my, _) in enumerate(seg_mids):
                d = np.hypot(mx - px, my - py)
                if d < best_d and i not in used:
                    best_d, best_i = d, i
            if best_d > threshold or best_i < 0:
                continue
            used.add(best_i)
            x1, y1, x2, y2 = seg_mids[best_i][2]
            matched_segs[yard_x] = (x1, y1, x2, y2)
            for t in (0.1, 0.25, 0.5, 0.75, 0.9):
                sx = x1 + t * (x2 - x1)
                sy = y1 + t * (y2 - y1)
                fp = cv2.perspectiveTransform(np.array([[[sx, sy]]], dtype=np.float64), H)[0][0]
                field_y = float(np.clip(fp[1], 0.0, FIELD_WIDTH))
                img_pts.append([sx, sy])
                field_pts.append([float(yard_x), field_y])

    # Hash mark crossings -> hard (field_x, 23.58 / 29.75) anchors
    # These give a y-axis constraint even when only one yard line is visible
    has_hash = False
    if matched_segs:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        white_only = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        for yard_x, seg in matched_segs.items():
            hm_pts = _find_hash_marks_on_segment(seg, white_only)
            if len(hm_pts) == 2:
                for (hu, hv), hy in zip(sorted(hm_pts, key=lambda p: p[1]), HASH_FIELD_YS):
                    img_pts.append([hu, hv])
                    field_pts.append([float(yard_x), hy])
                has_hash = True

    if len(img_pts) < 4:
        return H

    H_ref, mask_r = cv2.findHomography(
        np.array(img_pts,   dtype=np.float32),
        np.array(field_pts, dtype=np.float32),
        cv2.RANSAC, 5.0,
    )
    if H_ref is None or (mask_r is not None and int(mask_r.sum()) < 3):
        return H

    # Cap blend: 25 % normally, 40 % when hash marks anchor the y-axis.
    # Keeping this low prevents noisy per-frame detection from causing jitter
    # temporal smoothing in the caller handles larger accumulated drift.
    max_alpha = 0.40 if has_hash else 0.25
    alpha = min(max_alpha, int(mask_r.sum()) / 40.0) if mask_r is not None else 0.15
    return (1.0 - alpha) * H + alpha * H_ref


# Camera motion detection

def _motion_score(bgr_prev: np.ndarray, bgr_curr: np.ndarray) -> float:
    """
    Median absolute pixel change between two grayscale frames.
    Still camera (player motion only): median around 0–1, most pixels unchanged.
    Camera pan/zoom: everything shifts -> median jumps to 5–30+.
    Using median so it makes it robust to player movement which only affects a minority of pixels.
    """
    g_prev = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2GRAY)
    g_curr = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2GRAY)
    return float(np.median(cv2.absdiff(g_prev, g_curr)))


# Temporal smoothing

def temporal_smooth_H(
    homographies: list,
    window: int = 9,
) -> list:
    """
    Gaussian-weighted sliding-window smooth over the H array.
    Neighboring frames stabilize each othe
    window=9 means plus or minus 4 frames contribute, if you increase there is more smoothing.
    """
    n = len(homographies)
    half = window // 2
    sigma = window / 4.0
    offsets = np.arange(-half, half + 1)
    weights = np.exp(-0.5 * offsets ** 2 / sigma ** 2)

    smoothed: list = []
    for i in range(n):
        H_acc: np.ndarray | None = None
        w_sum = 0.0
        for di, w in zip(offsets, weights):
            j = int(i + di)
            if 0 <= j < n and homographies[j] is not None:
                H_acc = w * homographies[j] if H_acc is None else H_acc + w * homographies[j]
                w_sum += w
        if H_acc is not None and w_sum > 0:
            smoothed.append(H_acc / w_sum)
        else:
            smoothed.append(homographies[i])
    return smoothed


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-frame homographies")
    parser.add_argument("annotations_json")
    parser.add_argument("video")
    parser.add_argument("--out",           default="field/homographies.npz")
    parser.add_argument("--field-refine",  action="store_true",
                        help="Apply sideline + yard-line Hough correction each frame")
    parser.add_argument("--smooth",        type=int, default=9, metavar="N",
                        help="Gaussian temporal smoothing window in frames (default 9, 0=off)")
    parser.add_argument("--motion-thresh", type=float, default=2.0, metavar="T",
                        help="Median pixel change to trigger field-refine (default 2.0, 0=always)")
    args = parser.parse_args()

    with open(args.annotations_json) as f:
        annotations = json.load(f)

    keyframes: dict[int, dict[int, tuple[float, float]]] = {}
    for fname, kpts in annotations.items():
        idx = frame_idx_from_name(fname)
        keyframes[idx] = {int(k): tuple(v) for k, v in kpts.items()}

    keyframe_indices = sorted(keyframes.keys())
    print(f"Loaded {len(keyframe_indices)} keyframes: {keyframe_indices}")

    cap   = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames")

    print(f"Mode: interpolation{' + field-refine' if args.field_refine else ''}")

    kf_sorted = sorted(keyframes.keys())
    kf_H = {k: compute_h_from_keypoints(keyframes[k]) for k in kf_sorted}
    kf_sorted = [k for k in kf_sorted if kf_H[k] is not None]
    last_kf = kf_sorted[-1] if kf_sorted else -1

    homographies = []
    frame_idx = 0
    H_carry:  np.ndarray | None = None
    bgr_prev: np.ndarray | None = None
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        before = [k for k in kf_sorted if k <= frame_idx]
        after  = [k for k in kf_sorted if k >= frame_idx]

        if not before and not after:
            H = None
        elif not before:
            H = kf_H[after[0]].copy()
        elif not after:
            H = kf_H[before[-1]].copy()
        elif before[-1] == frame_idx:
            H = kf_H[before[-1]].copy()
        else:
            lo, hi = before[-1], after[0]
            t = (frame_idx - lo) / (hi - lo)
            H = (1.0 - t) * kf_H[lo] + t * kf_H[hi]

        if H is not None and args.field_refine:
            moved = (bgr_prev is None or args.motion_thresh <= 0 or
                     _motion_score(bgr_prev, bgr) >= args.motion_thresh)
            if moved:
                base = H_carry if (frame_idx > last_kf and H_carry is not None) else H
                H = refine_H_with_field_lines(bgr, base)

        H_carry  = H
        bgr_prev = bgr
        homographies.append(H)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"\rFrame {frame_idx}/{total}", end="", flush=True)

    cap.release()
    print()
    if args.smooth > 0:
        print(f"Temporal smoothing (window={args.smooth})…")
        homographies = temporal_smooth_H(homographies, window=args.smooth)
    valid_mask = np.array([h is not None for h in homographies], dtype=bool)
    H_array = np.array(
        [h if h is not None else np.eye(3, dtype=np.float32) for h in homographies],
        dtype=np.float32,
    )
    np.savez(args.out, H=H_array, valid=valid_mask)
    print(f"Saved {valid_mask.sum()}/{total} valid homographies → {args.out}")


if __name__ == "__main__":
    main()
