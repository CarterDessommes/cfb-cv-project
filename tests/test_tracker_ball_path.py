import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_tree(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def test_tracker_uses_shared_roi_ball_tracker():
    tracker_path = ROOT / "src" / "tracker.py"
    tracker_source = tracker_path.read_text()
    track_video = _function_tree(tracker_path, "track_video")

    track_calls = [
        node
        for node in ast.walk(track_video)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "track"
    ]
    ball_tracker_calls = [
        node
        for node in ast.walk(track_video)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "BallTracker"
    ]
    update_calls = [
        node
        for node in ast.walk(track_video)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
    ]

    assert "from .ball_tracker import BallTracker" in tracker_source
    assert len(track_calls) == 1
    assert len(ball_tracker_calls) == 1
    assert len(update_calls) == 2  # MultiScaleTracker.update + BallTracker.update
