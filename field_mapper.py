"""
field_mapper.py — project player bounding boxes to field coordinates.

Field coordinate system:
  X  0 → 100   yards, goal line to goal line
  Y  0 → 53.33 yards, left sideline to right sideline
  Hash marks at y = 23.58, 29.75
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Detection = Dict  # {"track_id": int, "bbox": [x1,y1,x2,y2], "class": int}
FieldPoint = Dict  # {"track_id": int, "field_x": float, "field_y": float, ...}

FIELD_LENGTH = 100.0   # yards
FIELD_WIDTH  = 53.33   # yards
CANVAS_SCALE = 8       # pixels per yard → 800 × 427 px canvas


# -- Homography loading --

def load_homography_index(summary_path: str | Path) -> List[Dict]:
    """Load calibrated H matrices from manual_homography/summary.json."""
    data = json.loads(Path(summary_path).read_text())
    index = []
    for entry in data.get("frames", []):
        H_list = entry.get("homography_matrix")
        if H_list is None:
            continue
        stem = Path(entry.get("frame_path", "frame_0")).stem
        try:
            frame_num = int(stem.split("_")[-1])
        except ValueError:
            frame_num = 0
        index.append({"frame_num": frame_num, "H": np.array(H_list, dtype=np.float64)})
    index.sort(key=lambda e: e["frame_num"])
    return index


def get_homography_for_frame(
    frame_num: int,
    index: List[Dict],
    interpolate: bool = True,
) -> Optional[np.ndarray]:
    """Return H for frame_num from a summary.json index, with optional linear interpolation."""
    if not index:
        return None
    if frame_num < index[0]["frame_num"]:
        return index[0]["H"].copy()
    if frame_num > index[-1]["frame_num"]:
        return index[-1]["H"].copy()
    for entry in index:
        if entry["frame_num"] == frame_num:
            return entry["H"].copy()
    for i in range(len(index) - 1):
        lo, hi = index[i], index[i + 1]
        if lo["frame_num"] <= frame_num <= hi["frame_num"]:
            if not interpolate:
                d_lo = abs(frame_num - lo["frame_num"])
                d_hi = abs(frame_num - hi["frame_num"])
                return (lo if d_lo <= d_hi else hi)["H"].copy()
            t = (frame_num - lo["frame_num"]) / (hi["frame_num"] - lo["frame_num"])
            return (1.0 - t) * lo["H"] + t * hi["H"]
    return index[-1]["H"].copy()


def load_homography_npz(npz_path: str | Path) -> tuple:
    """Load per-frame H matrices from homographies.npz. Returns (H_array [N,3,3], valid [N bool])."""
    data = np.load(npz_path)
    return data["H"], data["valid"]


def get_homography_npz(frame_num: int, H_array: np.ndarray, valid: np.ndarray) -> Optional[np.ndarray]:
    """Return H for frame_num from npz arrays (clamped). Returns None if frame is invalid."""
    frame_num = max(0, min(frame_num, len(H_array) - 1))
    if not valid[frame_num]:
        return None
    return H_array[frame_num].copy()


# -- Foot-point extraction + VP correction --

def bbox_foot(bbox: List[float]) -> Tuple[float, float]:
    """Return bottom-center of bbox [x1, y1, x2, y2] as the player's foot pixel."""
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float(y2))


def vp_ray_foot(
    foot_px: Tuple[float, float],
    vp: Tuple[float, float],
    snap_to_yard_lines: bool = False,
    yard_line_xs_px: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Optionally refine a foot pixel by snapping it along the VP depth ray to the
    nearest detected yard-line column. Returns foot_px unchanged if no snap targets.
    """
    if vp is None:
        return foot_px
    fu, fv = foot_px
    vpu, vpv = vp
    dx, dy = vpu - fu, vpv - fv
    length = math.hypot(dx, dy)
    if length < 1.0:
        return foot_px
    dx /= length
    dy /= length

    if not snap_to_yard_lines or not yard_line_xs_px:
        return foot_px

    if abs(dx) < 1e-3:
        return foot_px

    best_t, best_dist = None, float("inf")
    for x_yard in yard_line_xs_px:
        t = (x_yard - fu) / dx
        if t < -50 or t > 1000:
            continue
        if abs(t) < best_dist:
            best_dist = abs(t)
            best_t = t

    if best_t is None:
        return foot_px
    return (fu + best_t * dx, fv + best_t * dy)


# -- Core projection --

def project_to_field(
    pixel_points: List[Tuple[float, float]],
    H: np.ndarray,
) -> List[Tuple[float, float]]:
    """Apply homography H to image pixel points. Returns list of (x_yards, y_yards)."""
    if not pixel_points:
        return []
    pts = np.array(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return [(float(p[0][0]), float(p[0][1])) for p in transformed]


def project_players(
    frame_num: int,
    detections: List[Detection],
    homography_path: str | Path = "homographies.npz",
    vp: Optional[Tuple[float, float]] = None,
    snap_to_yard_lines: bool = False,
    yard_line_xs_px: Optional[List[float]] = None,
) -> List[FieldPoint]:
    """
    Project all detected players in a frame to field coordinates.

    Returns list of FieldPoint dicts:
      {"track_id", "class", "foot_px": [u,v], "field_x", "field_y", "in_bounds"}
    """
    homography_path = Path(homography_path)
    if homography_path.suffix == ".npz":
        H_array, valid = load_homography_npz(homography_path)
        H = get_homography_npz(frame_num, H_array, valid)
    else:
        index = load_homography_index(homography_path)
        H = get_homography_for_frame(frame_num, index)

    if H is None:
        return []

    foot_pixels: List[Tuple[float, float]] = []
    for det in detections:
        raw_foot = bbox_foot(det.get("bbox", [0, 0, 1, 1]))
        foot_pixels.append(vp_ray_foot(
            raw_foot, vp,
            snap_to_yard_lines=snap_to_yard_lines,
            yard_line_xs_px=yard_line_xs_px,
        ))

    field_coords = project_to_field(foot_pixels, H)

    results: List[FieldPoint] = []
    for det, foot_px, (fx, fy) in zip(detections, foot_pixels, field_coords):
        in_bounds = (0.0 <= fx <= FIELD_LENGTH) and (0.0 <= fy <= FIELD_WIDTH)
        results.append({
            "track_id":  int(det.get("track_id", -1)),
            "class":     int(det.get("class",    0)),
            "foot_px":   [float(foot_px[0]), float(foot_px[1])],
            "field_x":   round(fx, 2),
            "field_y":   round(fy, 2),
            "in_bounds": in_bounds,
        })
    return results


# -- Top-down canvas rendering --

def build_field_canvas(scale: int = CANVAS_SCALE) -> np.ndarray:
    """Create a blank top-down football field canvas (BGR image)."""
    h = int(FIELD_WIDTH  * scale)
    w = int(FIELD_LENGTH * scale)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (34, 139, 34)  # green

    for x_yd in range(0, int(FIELD_LENGTH) + 1, 10):
        cv2.line(canvas, (int(x_yd * scale), 0), (int(x_yd * scale), h - 1), (255, 255, 255), 1)
    for x_yd in range(5, int(FIELD_LENGTH), 10):
        cv2.line(canvas, (int(x_yd * scale), 0), (int(x_yd * scale), h - 1), (180, 180, 180), 1)

    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)

    for y_yd in [23.58, 29.75]:
        y_px = int(y_yd * scale)
        for x_yd in range(0, int(FIELD_LENGTH) + 1, 5):
            x_px = int(x_yd * scale)
            cv2.line(canvas, (x_px - 3, y_px), (x_px + 3, y_px), (200, 200, 200), 1)

    label_map = {0: "G", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50",
                 60: "40", 70: "30", 80: "20", 90: "10", 100: "G"}
    for x_yd, label in label_map.items():
        cv2.putText(canvas, label, (int(x_yd * scale) - 8, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return canvas


_COLORS = {0: (0, 200, 255), 1: (0, 128, 255)}
_OUT_COLOR = (80, 80, 80)


def draw_players_on_canvas(
    canvas: np.ndarray,
    players: List[FieldPoint],
    scale: int = CANVAS_SCALE,
    draw_ids: bool = True,
) -> np.ndarray:
    """Draw projected player positions onto a top-down field canvas."""
    out = canvas.copy()
    for p in players:
        cx = int(p["field_x"] * scale)
        cy = int(p["field_y"] * scale)
        color = _COLORS.get(p.get("class", 0), _COLORS[0]) if p.get("in_bounds", True) else _OUT_COLOR
        cv2.circle(out, (cx, cy), 6, color, -1)
        cv2.circle(out, (cx, cy), 6, (255, 255, 255), 1)
        if draw_ids and p.get("track_id", -1) >= 0:
            cv2.putText(out, str(p["track_id"]), (cx + 7, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out


def overlay_broadcast_frame(
    broadcast_frame: np.ndarray,
    players: List[FieldPoint],
    vp: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Draw foot dots and optional VP depth rays on the broadcast frame for debugging."""
    out = broadcast_frame.copy()
    for p in players:
        fu, fv = int(p["foot_px"][0]), int(p["foot_px"][1])
        color = _COLORS.get(p.get("class", 0), _COLORS[0])
        cv2.circle(out, (fu, fv), 5, color, -1)
        cv2.circle(out, (fu, fv), 5, (255, 255, 255), 1)
        if vp is not None:
            vpu, vpv = vp
            dx, dy = vpu - fu, vpv - fv
            L = math.hypot(dx, dy)
            if L > 0:
                ray_len = min(40.0, L * 0.2)
                cv2.arrowedLine(out, (fu, fv),
                                (int(fu + (dx / L) * ray_len), int(fv + (dy / L) * ray_len)),
                                (0, 255, 255), 1, tipLength=0.3)
        if p.get("track_id", -1) >= 0:
            cv2.putText(out, str(p["track_id"]), (fu + 6, fv - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if vp is not None:
        vpu, vpv = int(vp[0]), int(vp[1])
        cv2.drawMarker(out, (vpu, vpv), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(out, "VP", (vpu + 8, vpv - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out
