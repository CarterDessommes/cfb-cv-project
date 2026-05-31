import ast
import inspect

import pipeline
from pipeline import _swap_offense_defense


def test_swap_offense_defense_preserves_unknowns():
    assert _swap_offense_defense(["offense", "defense", "unknown"]) == [
        "defense",
        "offense",
        "unknown",
    ]


def test_pipeline_only_classifies_once_per_frame():
    tree = ast.parse(inspect.getsource(pipeline.run_pipeline))
    classify_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "classify"
    ]

    assert len(classify_calls) == 1
