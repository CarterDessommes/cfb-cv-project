import ast
import inspect

import pipeline


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
