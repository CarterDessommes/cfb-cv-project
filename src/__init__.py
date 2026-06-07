"""Core library code for the CFB CV pipeline."""

from .tracker import get_tracker_config_path, load_model, track_video
from .ball_tracker import BallTracker
from .team_classifier import TeamClassifier, _best_device
from .field_mapper import (
    project_players,
    build_field_canvas,
    field_to_canvas_point,
    load_homographies,
    draw_field_overlay,
    CANVAS_SCALE,
)
from .multiscale import merge_detections, MultiScaleTracker
from .yolo_utils import boxes_to_cpu_arrays, due

__all__ = [
    "get_tracker_config_path",
    "load_model",
    "track_video",
    "BallTracker",
    "TeamClassifier",
    "_best_device",
    "project_players",
    "build_field_canvas",
    "field_to_canvas_point",
    "load_homographies",
    "draw_field_overlay",
    "CANVAS_SCALE",
    "merge_detections",
    "MultiScaleTracker",
    "boxes_to_cpu_arrays",
    "due",
]
