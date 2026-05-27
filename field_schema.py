"""
Keypoint schema for football field landmarks.
NFL field: 100 yds x 53.33 yds (excluding endzones).
Shared source of truth for keypoint IDs across the project.

Schema: id = yard_idx * 4 + rail_idx
  rail 0 = near_sideline (y = 0)
  rail 1 = near_hash     (y = 23.58)
  rail 2 = far_hash      (y = 29.75)
  rail 3 = far_sideline  (y = 53.33)
  yard_idx 0..10 -> yards 0, 10, 20, ..., 100
"""
from __future__ import annotations

import numpy as np

# Rail y-coordinates (sideline -> hash -> hash -> sideline)
RAILS: list[float] = [0.0, 23.58, 29.75, 53.33]
RAIL_NAMES: list[str] = ["near_sideline", "near_hash", "far_hash", "far_sideline"]

# Yard line x-coordinates (goal line to goal line, every 10 yds)
YARD_LINES: list[int] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Build the keypoint list: 44 entries, indexed by ID 0..43
KEYPOINTS: list[dict] = []
for i, x in enumerate(YARD_LINES):
    for j, (y, rail_name) in enumerate(zip(RAILS, RAIL_NAMES)):
        kp_id = i * len(RAILS) + j
        KEYPOINTS.append({
            "id": kp_id,
            "x": float(x),
            "y": float(y),
            "name": f"yd{x:03d}_{rail_name}",
        })

# Numpy array of (x, y) field coords, indexed by keypoint ID — used by findHomography
FIELD_LANDMARKS: np.ndarray = np.array(
    [[kp["x"], kp["y"]] for kp in KEYPOINTS], dtype=np.float32
)

# Flip pairs for horizontal-image augmentation: kp i flips to FLIP_PAIRS[i]
# After horizontal image flip, x_field -> 100 - x_field; rail (y) unchanged.
FLIP_PAIRS: dict[int, int] = {}
n_yards = len(YARD_LINES)
n_rails = len(RAILS)
for i in range(n_yards):
    for j in range(n_rails):
        orig_id = i * n_rails + j
        flipped_id = (n_yards - 1 - i) * n_rails + j
        FLIP_PAIRS[orig_id] = flipped_id


if __name__ == "__main__":
    print(f"Total keypoints: {len(KEYPOINTS)}")
    print("\nFirst 8 entries:")
    for kp in KEYPOINTS[:8]:
        print(f"  ID {kp['id']:2d}: {kp['name']:30s} -> ({kp['x']:>5.2f}, {kp['y']:>5.2f})")
    print(f"  ... ({len(KEYPOINTS) - 8} more)")
    print(f"\nFIELD_LANDMARKS shape: {FIELD_LANDMARKS.shape}")
    print(f"\nSample flip pairs:")
    for orig in [0, 5, 22, 43]:
        print(f"  {orig} ({KEYPOINTS[orig]['name']}) <-> {FLIP_PAIRS[orig]} ({KEYPOINTS[FLIP_PAIRS[orig]]['name']})")
