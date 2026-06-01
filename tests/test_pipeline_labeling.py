import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pipeline


def test_nearest_player_to_ball_returns_closest_track_id():
    boxes = [
        [0, 0, 20, 20, 10, 0],
        [40, 0, 60, 20, 20, 0],
        [100, 0, 120, 20, 30, 0],
    ]

    assert pipeline._nearest_player_to_ball((55, 12), boxes) == 20


def test_nearest_player_to_ball_handles_missing_ball_or_boxes():
    assert pipeline._nearest_player_to_ball(None, [[0, 0, 20, 20, 10, 0]]) is None
    assert pipeline._nearest_player_to_ball((10, 10), []) is None


def test_pipeline_uses_cluster_assignment_cache_path():
    tree = ast.parse(inspect.getsource(pipeline.run_pipeline))
    classify_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "classify"
    ]
    assign_cluster_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assign_clusters"
    ]

    assert len(classify_calls) == 0
    assert len(assign_cluster_calls) == 1


def test_pipeline_marks_ball_carrier_without_ball_trail():
    source = inspect.getsource(pipeline.run_pipeline)
    tree = ast.parse(source)

    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_nearest_player_to_ball"
    ]
    carrier_flags = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and node.value == "is_ball_carrier"
    ]

    assert "_BALL_TRAIL" not in source
    assert "ball_xy[0] / frame.shape[1] * canvas.shape[1]" not in source
    assert len(helper_calls) == 1
    assert len(carrier_flags) >= 2
    assert 'cv2.putText(canvas, "C"' in source
