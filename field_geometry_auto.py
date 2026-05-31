"""
Automatic field geometry estimation.
Hough-based line detection, line-family split (yard lines vs sideline-like lines), vanishing point estimation from pairwise intersections
Run: python3 field_geometry_auto.py --frame calibration_frames/frame_00000.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Line = Tuple[float, float, float, float]
Point = Tuple[float, float]


def detect_lines_hough(frame: np.ndarray) -> List[Line]:
    """Detect candidate field lines via Canny + probabilistic Hough."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=max(80, frame.shape[1] // 12),
        maxLineGap=20,
    )

    if lines is None:
        return []

    return [tuple(map(float, row[0])) for row in lines]


def line_angle_deg(line: Line) -> float:
    x1, y1, x2, y2 = line
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    angle = (angle + 180.0) % 180.0
    return float(angle)


def split_line_families(lines: Sequence[Line]) -> Tuple[List[Line], List[Line]]:
    """Split lines into two orientation families using 1D k-means on angle."""
    if len(lines) < 2:
        return list(lines), []

    angles = np.array([[line_angle_deg(line)] for line in lines], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    compactness, labels, centers = cv2.kmeans(
        angles,
        K=2,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    del compactness
    centers = centers.flatten()

    families = {0: [], 1: []}
    for line, label in zip(lines, labels.flatten()):
        families[int(label)].append(line)

    # Yard lines are usually the steeper family in broadcast view.
    if centers[0] > centers[1]:
        yard_label, side_label = 0, 1
    else:
        yard_label, side_label = 1, 0

    return families[yard_label], families[side_label]


def line_to_abc(line: Line) -> Tuple[float, float, float]:
    """Convert segment endpoints to infinite line ax + by + c = 0."""
    x1, y1, x2, y2 = line
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def intersection(l1: Line, l2: Line) -> Optional[Point]:
    a1, b1, c1 = line_to_abc(l1)
    a2, b2, c2 = line_to_abc(l2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return float(x), float(y)


def estimate_vanishing_point(lines: Sequence[Line], frame_shape: Tuple[int, int, int]) -> Optional[Point]:
    """Estimate VP from robust median of pairwise intersections."""
    if len(lines) < 2:
        return None

    height, width = frame_shape[:2]
    intersections: List[Point] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = intersection(lines[i], lines[j])
            if pt is None:
                continue
            x, y = pt
            # Keep plausible region (allow outside frame margin)
            if -2 * width <= x <= 3 * width and -2 * height <= y <= 3 * height:
                intersections.append(pt)

    if not intersections:
        return None

    pts = np.array(intersections, dtype=np.float32)
    vp = np.median(pts, axis=0)
    return float(vp[0]), float(vp[1])


def draw_lines(image: np.ndarray, lines: Sequence[Line], color: Tuple[int, int, int], thickness: int = 2) -> None:
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def build_geometry(frame: np.ndarray) -> Dict:
    all_lines = detect_lines_hough(frame)
    yard_lines, side_lines = split_line_families(all_lines)
    vp = estimate_vanishing_point(side_lines if len(side_lines) >= 2 else yard_lines, frame.shape)

    return {
        "num_all_lines": len(all_lines),
        "num_yard_lines": len(yard_lines),
        "num_sideline_family_lines": len(side_lines),
        "all_lines": [list(line) for line in all_lines],
        "yard_lines": [list(line) for line in yard_lines],
        "sideline_family_lines": [list(line) for line in side_lines],
        "vanishing_point": list(vp) if vp is not None else None,
    }


def save_outputs(frame_path: Path, output_dir: Path, geom: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Could not open frame: {frame_path}")

    overlay = frame.copy()
    all_lines = [tuple(line) for line in geom["all_lines"]]
    yard_lines = [tuple(line) for line in geom["yard_lines"]]
    side_lines = [tuple(line) for line in geom["sideline_family_lines"]]

    draw_lines(overlay, all_lines, (80, 80, 80), 1)
    draw_lines(overlay, yard_lines, (0, 255, 255), 2)      # yellow
    draw_lines(overlay, side_lines, (255, 0, 0), 2)        # blue

    vp = geom.get("vanishing_point")
    if vp is not None:
        vx, vy = int(vp[0]), int(vp[1])
        cv2.circle(overlay, (vx, vy), 8, (0, 0, 255), -1)
        cv2.putText(overlay, "VP", (vx + 8, vy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    alpha = 0.8
    vis = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    image_out = output_dir / f"{frame_path.stem}_geometry.png"
    json_out = output_dir / f"{frame_path.stem}_geometry.json"
    cv2.imwrite(str(image_out), vis)
    json_out.write_text(json.dumps(geom, indent=2))

    print(f"Saved: {image_out}")
    print(f"Saved: {json_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic field geometry estimation")
    parser.add_argument("--frame", required=True, help="Path to image frame")
    parser.add_argument("--out-dir", default="outputs/field_geometry", help="Output directory")
    args = parser.parse_args()

    frame_path = Path(args.frame)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Could not open frame: {frame_path}")

    geom = build_geometry(frame)
    print(
        f"Detected lines: total={geom['num_all_lines']} "
        f"yard={geom['num_yard_lines']} sideline_family={geom['num_sideline_family_lines']}"
    )
    print(f"Vanishing point: {geom['vanishing_point']}")

    save_outputs(frame_path, Path(args.out_dir), geom)


if __name__ == "__main__":
    main()
