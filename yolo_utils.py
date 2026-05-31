"""Small helpers for Ultralytics result handling."""

import numpy as np


def _as_numpy(value, dtype=None):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numpy"):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    return arr.astype(dtype) if dtype is not None else arr


def boxes_to_cpu_arrays(boxes_obj):
    """Move an Ultralytics Boxes object to CPU once and expose NumPy arrays."""
    if boxes_obj is None or len(boxes_obj) == 0:
        return None, None, None, None

    boxes_cpu = boxes_obj.cpu()
    xyxy = _as_numpy(boxes_cpu.xyxy)
    ids = _as_numpy(boxes_cpu.id, dtype=int)
    cls = _as_numpy(boxes_cpu.cls, dtype=int)
    conf = _as_numpy(boxes_cpu.conf)
    return xyxy, ids, cls, conf
